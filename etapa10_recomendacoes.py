# ============================================================
# ETAPA 10 - GERAÇÃO DE RECOMENDAÇÕES (VERSÃO MODIFICADA)
# ------------------------------------------------------------
# Objetivo:
#   - Ler o CSV de variáveis relevantes (seja p-valor ou importância).
#   - Ler o CSV de coeficientes do OLS (para obter a direção).
#   - Combinar os dois: gerar recomendações para as variáveis
#     mais importantes, usando a direção do OLS.
# ============================================================

import logging
import os
import pandas as pd
from pathlib import Path

# ==========================================================
# FUNÇÃO AUXILIAR: Encontrar o CSV de variáveis relevantes
# ==========================================================
def _encontrar_csv_relevancia():
    """Encontra o CSV correto (RF ou OLS) gerado pela Etapa 9."""
    caminho_rf = Path("resultados/tabelas/variaveis_importancia_rf.csv")
    caminho_ols = Path("resultados/tabelas/variaveis_significativas_ols.csv")
    
    if caminho_rf.exists():
        return caminho_rf, "rf"
    if caminho_ols.exists():
        return caminho_ols, "ols"
    
    raise FileNotFoundError("Nenhum arquivo de relevância (RF ou OLS) da Etapa 9 foi encontrado.")

# ==========================================================
# FUNÇÃO PRINCIPAL – GERAÇÃO DE RECOMENDAÇÕES
# ==========================================================
def gerar_recomendacoes(cenario, modelo_ols, respostas=None):
    """
    Gera recomendações de políticas públicas combinando:
    1. Importância (do melhor modelo, ex: RF)
    2. Direção (dos coeficientes do OLS)
    """

    etapa = "ETAPA 10 - Geração de Recomendações"
    logging.info(f"({etapa}) - Iniciando recomendações baseadas nos achados...")

    try:
        # ======================================================
        # 1. Contextualização do cenário (sem alteração)
        # ======================================================
        pais = cenario.get("pais", "Desconhecido")
        disciplina = cenario.get("disciplina", "Desconhecida")
        tema = cenario.get("tema", "Não definido")

        # ======================================================
        # 2. Carregar os resultados dos modelos
        # ======================================================
        
        # 2.1. Carrega as variáveis RELEVANTES (do melhor modelo, via Etapa 9)
        caminho_csv_relev, tipo_modelo = _encontrar_csv_relevancia()
        df_relevancia = pd.read_csv(caminho_csv_relev)
        
        variaveis_relevantes = []
        if tipo_modelo == "rf":
            # Pega as 5 variáveis mais importantes (ou com importância > 0.05)
            df_relevancia = df_relevancia.sort_values(by="Importancia", ascending=False)
            variaveis_relevantes = df_relevancia.head(5)["Variável"].tolist()
        else: # ols
            # Pega variáveis com P-valor <= 0.05
            variaveis_relevantes = df_relevancia[df_relevancia["P-valor"] <= 0.05]["Variável"].tolist()

        # 2.2. Carrega os coeficientes do OLS (para DIREÇÃO)
        caminho_ols_coefs = Path("resultados/tabelas/modelo_ols_resultados.csv")
        if not caminho_ols_coefs.exists():
            raise FileNotFoundError("Arquivo 'modelo_ols_resultados.csv' não encontrado. Execute a Etapa 7.")
        
        df_ols_coefs = pd.read_csv(caminho_ols_coefs)

        # ======================================================
        # 3. Geração automática de recomendações (Lógica Híbrida)
        # ======================================================
        recomendacoes = []

        if not variaveis_relevantes:
            logging.warning(f"({etapa}) - Nenhuma variável estatisticamente relevante encontrada para gerar recomendações.")
        else:
            logging.info(f"({etapa}) - Gerando recomendações para: {variaveis_relevantes}")

        for var_nome_original in variaveis_relevantes:
            # Encontrar o coeficiente OLS correspondente para esta variável
            coef_row = df_ols_coefs[df_ols_coefs["original"] == var_nome_original]
            
            if coef_row.empty:
                # Caso especial se a variável for o Intercept (não deve acontecer)
                continue

            coef = coef_row["coeficiente"].values[0]
            
            if coef > 0:
                direcao = "positivamente"
                acao = "fortalecer e investir"
            else:
                direcao = "negativamente"
                acao = "mitigar e revisar"

            recomendacoes.append(
                f"A variável '{var_nome_original}' foi identificada como um fator chave. "
                f"Os dados sugerem que ela impacta {direcao} o(a) {tema.lower()}. "
                f"Recomenda-se {acao} políticas públicas relacionadas a este aspecto."
            )

        # ======================================================
        # 4. Recomendações complementares (sem alteração)
        # ======================================================
        if respostas is not None:
            try:
                # Tenta encontrar o alvo (pode falhar se o nome mudou, mas é seguro)
                alvo = [c for c in respostas.columns if "autoeficacia_norm" in c or "bem_estar_norm" in c]
                if alvo:
                    media = respostas[alvo[0]].mean(skipna=True)
                    if media < 0.3: # (Assumindo escala 0-1)
                        recomendacoes.append(
                            "Os índices médios de autoeficácia/bem-estar estão baixos. "
                            "Sugere-se implementar programas de apoio psicossocial e melhoria das condições de trabalho."
                        )
                    elif media < 0.6:
                         recomendacoes.append(
                            "Os índices de autoeficácia/bem-estar estão em nível intermediário. "
                            "Recomenda-se ampliar ações de reconhecimento e desenvolvimento profissional."
                        )
            except Exception:
                pass # Ignora falhas aqui

        # ======================================================
        # 5. Recomendações gerais (sem alteração)
        # ======================================================
        recomendacoes.extend([
            f"Investir na formação inicial e continuada de professores de {disciplina.lower()} no {pais}, "
            "com foco no desenvolvimento de competências pedagógicas e socioemocionais.",
            "Criar indicadores nacionais de bem-estar e autoeficácia docente e integrá-los aos processos de avaliação educacional."
        ])

        # ======================================================
        # 6. Salvamento dos resultados (sem alteração)
        # ======================================================
        os.makedirs("resultados/textos", exist_ok=True)
        caminho_txt = "resultados/textos/recomendacoes_politicas_publicas.txt"

        with open(caminho_txt, "w", encoding="utf-8") as f:
            f.write(f"=== Recomendações de Políticas Públicas ({tema} - {pais}) ===\n\n")
            if not recomendacoes:
                f.write("Nenhuma recomendação específica pôde ser gerada com base nos resultados estatísticos (nenhum fator relevante encontrado).\n")
            for i, rec in enumerate(recomendacoes, 1):
                f.write(f"{i}. {rec}\n")

        logging.info(f"({etapa}) - Relatório salvo em '{caminho_txt}'.")
        logging.info(f"({etapa}) - Recomendações geradas com sucesso.")

        return recomendacoes

    except Exception as e:
        logging.error(
            f"[ERRO FATAL] Falha na Etapa 10 - Geração de Recomendações: {e}",
            exc_info=True
        )
        raise