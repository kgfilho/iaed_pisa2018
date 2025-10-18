# ============================================================
# ETAPA 09 - REFINAMENTO DO CONHECIMENTO
# ------------------------------------------------------------
# Objetivo:
#   - Analisar o modelo OLS estimado na Etapa 07;
#   - Identificar variáveis estatisticamente significativas (p < 0.05);
#   - Reavaliar a hipótese inicial à luz dos resultados empíricos;
#   - Gerar síntese interpretativa para subsidiar políticas públicas.
#
# Entradas:
#   - modelo_ols: objeto statsmodels.OLS ajustado (Etapa 07)
#
# Saídas:
#   - DataFrame com variáveis significativas
#   - Relatório salvo em resultados/tabelas/variaveis_significativas.csv
# ============================================================

import logging
import pandas as pd
import os

# ==========================================================
# FUNÇÃO PRINCIPAL – REFINAMENTO DO CONHECIMENTO
# ==========================================================
def refinar_conhecimento(modelo_ols, respostas=None):
    """
    Refina o modelo ajustado na etapa anterior, avaliando:
    - Variáveis significativas (p-valor ≤ 0.05)
    - Coeficientes relevantes
    - Ajuste da hipótese com base nos achados empíricos
    - (Opcional) Geração de estatísticas complementares a partir do DataFrame original

    Parâmetros
    ----------
    modelo_ols : statsmodels.regression.linear_model.RegressionResultsWrapper
        Modelo OLS ajustado na Etapa 7.
    respostas : pandas.DataFrame, opcional
        Conjunto de dados transformados (utilizado para análises adicionais).
    
    Retorna
    -------
    variaveis_significativas : list
        Lista com os nomes das variáveis estatisticamente significativas.
    """

    logging.info("ETAPA 9 - Refinamento do Conhecimento) - Avaliando significância e ajustando hipótese...")

    try:
        # ======================================================
        # 1. Extração dos resultados do modelo OLS
        # ======================================================
        resultados = pd.DataFrame({
            "Variável": modelo_ols.params.index,
            "Coeficiente": modelo_ols.params.values,
            "P-valor": modelo_ols.pvalues.values,
            "Erro Padrão": modelo_ols.bse.values
        })

        # ======================================================
        # 2. Identificação das variáveis significativas
        # ======================================================
        variaveis_significativas = resultados.loc[resultados["P-valor"] <= 0.05, "Variável"].tolist()

        # ======================================================
        # 3. Criação da pasta de resultados (se necessário)
        # ======================================================
        os.makedirs("resultados/tabelas", exist_ok=True)

        # ======================================================
        # 4. Salvamento dos resultados em CSV
        # ======================================================
        caminho_csv = "resultados/tabelas/variaveis_significativas.csv"
        resultados.to_csv(caminho_csv, index=False, encoding="utf-8-sig")

        logging.info(f"ETAPA 9 - Refinamento do Conhecimento) - Relatório salvo em '{caminho_csv}'.")

        # ======================================================
        # 5. Relatório interpretativo
        # ======================================================
        logging.info(
            f"ETAPA 9 - Refinamento do Conhecimento) - Variáveis significativas identificadas: {variaveis_significativas}"
        )

        # ======================================================
        # 6. (Opcional) Análises complementares com o DataFrame
        # ======================================================
        if respostas is not None:
            try:
                # Exemplo: cálculo da média do índice de bem-estar
                if "indice_bem_estar" in respostas.columns:
                    media_bem_estar = respostas["indice_bem_estar"].mean()
                    logging.info(
                        f"ETAPA 9 - Refinamento do Conhecimento) - Média geral do índice de bem-estar docente: {media_bem_estar:.3f}"
                    )

                # Outras análises complementares podem ser incluídas aqui
                # como correlações adicionais, distribuição por cluster, etc.

            except Exception as e:
                logging.warning(f"ETAPA 9 - Análise complementar ignorada: {e}")

        # ======================================================
        # 7. Retorno final
        # ======================================================
        logging.info(
            f"ETAPA 9 - Refinamento do Conhecimento) - Relatório interpretativo gerado com sucesso."
        )
        return variaveis_significativas

    except Exception as e:
        logging.error(
            f"[ERRO FATAL] Falha na Etapa 9 - Refinamento do Conhecimento: {e}", exc_info=True
        )
        raise

