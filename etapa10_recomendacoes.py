# ============================================================
# ETAPA 10 - GERAÇÃO DE RECOMENDAÇÕES DE POLÍTICAS PÚBLICAS
# ------------------------------------------------------------
# Objetivo:
#   - Traduzir os resultados quantitativos em diretrizes qualitativas;
#   - Relacionar variáveis significativas a possíveis ações e políticas;
#   - Produzir um relatório de recomendações estruturado.
#
# Entradas:
#   - variaveis_significativas: DataFrame gerado na Etapa 09
#
# Saídas:
#   - Relatório salvo em resultados/tabelas/recomendacoes_politicas.txt
# ============================================================

import logging
import os
import pandas as pd

# ==========================================================
# FUNÇÃO PRINCIPAL – GERAÇÃO DE RECOMENDAÇÕES
# ==========================================================
def gerar_recomendacoes(cenario, modelo_ols, respostas=None):
    """
    Gera recomendações de políticas públicas e estratégias de formação docente
    com base nas variáveis significativas e no contexto analisado.

    Parâmetros
    ----------
    cenario : dict
        Informações do contexto de análise (país, disciplina, público e tema).
    modelo_ols : statsmodels.regression.linear_model.RegressionResultsWrapper
        Modelo OLS ajustado na Etapa 7, usado para interpretar coeficientes.
    respostas : pandas.DataFrame, opcional
        Conjunto de dados processados, usado para cálculos de apoio.

    Retorna
    -------
    recomendacoes : list
        Lista de recomendações interpretativas e práticas.
    """

    logging.info("ETAPA 10 - Geração de Recomendações) - Iniciando recomendações baseadas nos achados...")

    try:
        # ======================================================
        # 1. Contextualização do cenário
        # ======================================================
        pais = cenario.get("pais", "Desconhecido")
        disciplina = cenario.get("disciplina", "Desconhecida")
        tema = cenario.get("tema", "Não definido")

        # ======================================================
        # 2. Interpretação das variáveis significativas
        # ------------------------------------------------------
        # Seleciona apenas as variáveis com p-valor <= 0.05
        # ======================================================
        resultados = pd.DataFrame({
            "Variável": modelo_ols.params.index,
            "Coeficiente": modelo_ols.params.values,
            "P-valor": modelo_ols.pvalues.values
        })

        variaveis_relevantes = resultados[resultados["P-valor"] <= 0.05]

        # ======================================================
        # 3. Geração automática de recomendações interpretativas
        # ======================================================
        recomendacoes = []

        for _, row in variaveis_relevantes.iterrows():
            var = row["Variável"]
            coef = row["Coeficiente"]

            if coef > 0:
                direcao = "aumenta"
            else:
                direcao = "reduz"

            recomendacoes.append(
                f"A variável '{var}' {direcao} significativamente o nível de {tema.lower()}. "
                f"Recomenda-se fortalecer políticas que valorizem este aspecto entre os docentes."
            )

        # ======================================================
        # 4. Recomendações complementares com base em dados
        # ======================================================
        if respostas is not None and "indice_bem_estar" in respostas.columns:
            media = respostas["indice_bem_estar"].mean()
            if media < 0.3:
                recomendacoes.append(
                    "Os índices médios de bem-estar docente encontram-se abaixo do ideal. "
                    "Sugere-se implementar programas de apoio psicossocial e incentivos à qualidade de vida docente."
                )
            elif media < 0.6:
                recomendacoes.append(
                    "O bem-estar docente apresenta nível intermediário. "
                    "Recomenda-se ampliar ações de formação continuada e reconhecimento profissional."
                )
            else:
                recomendacoes.append(
                    "Os docentes demonstram níveis elevados de bem-estar. "
                    "É importante consolidar essas práticas por meio de políticas de manutenção e incentivo à permanência na carreira."
                )

        # ======================================================
        # 5. Recomendações gerais com base no contexto
        # ======================================================
        recomendacoes.extend([
            f"Investir na formação inicial e continuada de professores de {disciplina.lower()} no {pais}, "
            "com foco no desenvolvimento de competências pedagógicas e socioemocionais.",
            "Criar indicadores nacionais de bem-estar docente e integrá-los aos processos de avaliação educacional.",
            "Promover programas de cooperação entre escolas para fortalecer o apoio entre pares e o senso de comunidade docente.",
            "Fomentar a pesquisa aplicada sobre condições de trabalho e saúde mental de professores.",
            "Integrar as dimensões de bem-estar e engajamento docente às políticas curriculares e formativas."
        ])

        # ======================================================
        # 6. Salvamento dos resultados
        # ======================================================
        os.makedirs("resultados/textos", exist_ok=True)
        caminho_txt = "resultados/textos/recomendacoes_politicas_publicas.txt"

        with open(caminho_txt, "w", encoding="utf-8") as f:
            f.write("=== Recomendações de Políticas Públicas ===\n\n")
            for i, rec in enumerate(recomendacoes, 1):
                f.write(f"{i}. {rec}\n")

        logging.info(f"ETAPA 10 - Geração de Recomendações) - Relatório salvo em '{caminho_txt}'.")
        logging.info("ETAPA 10 - Geração de Recomendações) - Recomendações geradas com sucesso.")

        return recomendacoes

    except Exception as e:
        logging.error(
            f"[ERRO FATAL] Falha na Etapa 10 - Geração de Recomendações: {e}",
            exc_info=True
        )
        raise
