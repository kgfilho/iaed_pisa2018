# ============================================================
# ETAPA 09 - REFINAMENTO DO CONHECIMENTO (VERSÃO MODIFICADA)
# ------------------------------------------------------------
# - (MODIFICADO) No Passo 6, adiciona a média da variável alvo
#   (ex: 0.981) ao arquivo 'melhor_modelo.json' para ser
#   usado pela Etapa 11.
# ============================================================

import logging
import pandas as pd
import os
import json
import joblib
from pathlib import Path

# ==========================================================
# FUNÇÃO PRINCIPAL – REFINAMENTO DO CONHECIMENTO
# ==========================================================
def refinar_conhecimento(modelo_ols, respostas=None):
    """
    Refina o conhecimento com base no MELHOR modelo da Etapa 7.
    - Se OLS: usa p-valores.
    - Se RF/GB: usa feature_importances_.
    """

    etapa = "ETAPA 9 - Refinamento do Conhecimento"
    logging.info(f"({etapa}) - Avaliando significância/importância...")

    try:
        # ======================================================
        # 1. Carregar os metadados do melhor modelo
        # ======================================================
        caminho_json = Path("resultados/tabelas/melhor_modelo.json")
        if not caminho_json.exists():
            raise FileNotFoundError("Arquivo 'melhor_modelo.json' não encontrado. Execute a Etapa 7.")
        
        with open(caminho_json, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        melhor_modelo_nome = meta.get("melhor_modelo")
        features = meta.get("features", [])
        caminho_modelo_salvo = meta.get("caminho_modelo_salvo")

        logging.info(f"({etapa}) - Modelo vencedor identificado: {melhor_modelo_nome}")

        # ======================================================
        # 2. Extração dos resultados (lógica baseada no modelo)
        # ======================================================
        
        resultados_df = pd.DataFrame()
        variaveis_relevantes = []

        if melhor_modelo_nome in ["RandomForestRegressor", "GradientBoostingRegressor"]:
            # --- LÓGICA NOVA: Usar Feature Importance ---
            if not caminho_modelo_salvo or not Path(caminho_modelo_salvo).exists():
                raise FileNotFoundError(f"Arquivo do modelo '{caminho_modelo_salvo}' não encontrado.")

            modelo_nao_linear = joblib.load(caminho_modelo_salvo)
            importancias = modelo_nao_linear.feature_importances_

            resultados_df = pd.DataFrame({
                "Variável": features,
                "Importancia": importancias
            }).sort_values(by="Importancia", ascending=False)
            
            # Define "relevante" como variável com importância > 0.01 (1%)
            variaveis_relevantes = resultados_df[resultados_df["Importancia"] > 0.01]["Variável"].tolist()
            logging.info(f"({etapa}) - Análise de Feature Importance (do {melhor_modelo_nome}) concluída.")

        else:
            # --- LÓGICA ANTIGA: Usar P-Valor do OLS ---
            if modelo_ols is None:
                raise ValueError("Modelo OLS é 'None', mas foi selecionado como o melhor.")
                
            resultados_df = pd.DataFrame({
                "Variável": modelo_ols.params.index,
                "Coeficiente": modelo_ols.params.values,
                "P-valor": modelo_ols.pvalues.values,
                "Erro Padrão": modelo_ols.bse.values
            })
            
            resultados_df = resultados_df[~resultados_df["Variável"].str.contains('Intercept|const', case=False)]
            variaveis_relevantes = resultados_df.loc[resultados_df["P-valor"] <= 0.05, "Variável"].tolist()
            logging.info(f"({etapa}) - Análise de P-Valor (do OLS) concluída.")


        # ======================================================
        # 3. Criação da pasta de resultados
        # ======================================================
        os.makedirs("resultados/tabelas", exist_ok=True)

        # ======================================================
        # 4. Salvamento dos resultados em CSV
        # ======================================================
        if "Importancia" in resultados_df.columns:
            caminho_csv = "resultados/tabelas/variaveis_importancia_rf.csv"
        else:
            caminho_csv = "resultados/tabelas/variaveis_significativas_ols.csv"
            
        resultados_df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")
        logging.info(f"({etapa}) - Relatório de variáveis salvo em '{caminho_csv}'.")

        # ======================================================
        # 5. Relatório interpretativo
        # ======================================================
        if not variaveis_relevantes:
             logging.warning(f"({etapa}) - Nenhuma variável relevante/significativa encontrada (critério > 0.01 ou p < 0.05).")
        else:
            logging.info(
                f"({etapa}) - Variáveis relevantes identificadas: {variaveis_relevantes}"
            )

        # ======================================================
        # 6. (Opcional) Análises complementares (MODIFICADO)
        # ======================================================
        if respostas is not None:
            try:
                alvo = meta.get("alvo", "indice_autoeficacia_norm")
                if alvo in respostas.columns:
                    media_alvo = respostas[alvo].mean(skipna=True)
                    n_validos = int(respostas[alvo].notna().sum())
                    
                    logging.info(
                        f"({etapa}) - Média geral do índice alvo ({alvo}): {media_alvo:.3f} (N={n_validos})"
                    )

                    # ########################################################
                    # # ### INÍCIO DA MODIFICAÇÃO (Adicionar ao JSON) ###
                    # ########################################################
                    # Adiciona a média ao JSON para a Etapa 11
                    meta['alvo_media'] = media_alvo
                    meta['alvo_n_validos'] = n_validos
                    
                    with open(caminho_json, 'w', encoding='utf-8') as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)
                        
                    logging.info(f"({etapa}) - Média do alvo adicionada ao '{caminho_json.name}'.")
                    # ########################################################
                    # # ### FIM DA MODIFICAÇÃO ###
                    # ########################################################

            except Exception as e:
                logging.warning(f"({etapa}) - Análise complementar ignorada: {e}")

        # ======================================================
        # 7. Retorno final
        # ======================================================
        logging.info(
            f"({etapa}) - Refinamento concluído com sucesso."
        )
        return variaveis_relevantes

    except Exception as e:
        logging.error(
            f"[ERRO FATAL] Falha na Etapa 9 - Refinamento: {e}", exc_info=True
        )
        raise