# ============================================================
# ETAPA 07 - DESCOBERTA DE PADRÕES E MODELAGEM
# ------------------------------------------------------------
# Objetivo:
#   - Ajustar modelos estatísticos de regressão (OLS) para
#     identificar relações significativas entre variáveis
#     explicativas e o índice de bem-estar docente.
#   - Estimar a contribuição de fatores formativos, pedagógicos
#     e contextuais sobre o bem-estar dos professores de Matemática.
#
# Saída esperada:
#   - Modelo OLS ajustado
#   - Tabela de coeficientes e métricas estatísticas exportada
# ============================================================

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from utils_log import log_mensagem


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def _mapear_valor(v):
    """
    Conversão robusta de categorias textuais para valores numéricos.
    Mantém consistência com a Etapa 05.
    """
    if pd.isna(v):
        return np.nan

    s = str(v).strip().lower()

    # Likert (inglês)
    likert = {
        "strongly disagree": 1,
        "disagree": 2,
        "neutral": 3,
        "agree": 4,
        "strongly agree": 5,
    }
    if s in likert:
        return float(likert[s])

    # Binários (inglês/português) e booleanos comuns
    positivos = {"checked", "yes", "sim", "true", "1", "y", "s"}
    negativos = {"not checked", "no", "não", "nao", "false", "0", "n"}

    if s in positivos:
        return 1.0
    if s in negativos:
        return 0.0

    # Tentativa de conversão numérica direta
    try:
        return float(s)
    except Exception:
        return np.nan


def _converter_dataframe_numerico(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o mapeador robusto apenas nas colunas object e
    força numérico nas demais; retorna DataFrame float.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "O":
            df[col] = df[col].map(_mapear_valor)
        # Coerção final para numérico
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.astype(float)


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================
def ajustar_modelo(df: pd.DataFrame):
    etapa = "ETAPA 7 - Descoberta de Padrões e Modelagem"
    log_mensagem(etapa, "Ajustando modelo de regressão OLS...", "inicio")

    # ============================================================
    # 1) Seleção e cópia das variáveis relevantes
    # ------------------------------------------------------------
    variaveis = [c for c in df.columns if "TC0" in c or "indice_bem_estar" in c]
    df_modelo = df[variaveis].copy()

    # ============================================================
    # 2) Conversão robusta para formato numérico
    # ------------------------------------------------------------
    # Elimina o uso de .replace() para evitar FutureWarning.
    # Garante dtype float em todas as colunas.
    # ============================================================
    df_modelo = _converter_dataframe_numerico(df_modelo)

    # ============================================================
    # 3) Definição da variável dependente e preditoras
    # ------------------------------------------------------------
    if "indice_bem_estar_norm" not in df_modelo.columns:
        raise ValueError("[ERRO] Coluna 'indice_bem_estar_norm' não encontrada para modelagem.")

    y = df_modelo["indice_bem_estar_norm"].astype(float)
    X = df_modelo.drop(columns=["indice_bem_estar", "indice_bem_estar_norm"], errors="ignore")

    # Remove colunas totalmente nulas
    X = X.dropna(axis=1, how="all")

    # Remove colunas sem variância (constantes)
    if not X.empty:
        variancias = X.var(numeric_only=True)
        cols_variancia_zero = variancias[variancias == 0.0].index.tolist()
        if cols_variancia_zero:
            X = X.drop(columns=cols_variancia_zero, errors="ignore")

    # Checagem após limpeza
    if X.empty:
        raise ValueError("[ERRO] Não há variáveis explicativas numéricas válidas após a limpeza.")

    # ============================================================
    # 4) Alinhamento de índices e remoção de NaNs linha a linha
    # ------------------------------------------------------------
    dados = pd.concat([y.rename("y"), X], axis=1)
    dados = dados.dropna(axis=0, how="any")

    if dados.empty:
        raise ValueError("[ERRO] Não há observações completas para ajustar o modelo OLS.")

    y_valid = dados["y"].astype(float)
    X_valid = dados.drop(columns=["y"]).astype(float)
    X_valid = sm.add_constant(X_valid, has_constant="add")

    # ============================================================
    # 5) Ajuste do modelo OLS
    # ------------------------------------------------------------
    modelo = sm.OLS(y_valid, X_valid, missing="drop").fit()

    # ============================================================
    # 6) Exportação dos resultados
    # ------------------------------------------------------------
    os.makedirs("resultados/tabelas", exist_ok=True)
    path_csv = "resultados/tabelas/modelo_ols_resultados.csv"

    confint = modelo.conf_int()
    resumo_df = pd.DataFrame({
        "Variável": modelo.params.index,
        "Coeficiente": modelo.params.values,
        "P-Valor": modelo.pvalues.values,
        "IC_Inf": confint[0].values,
        "IC_Sup": confint[1].values
    })

    resumo_df.to_csv(path_csv, index=False, encoding="utf-8-sig")

    log_mensagem(
        etapa,
        f"Modelagem concluída. R² = {modelo.rsquared:.4f} | R² ajustado = {modelo.rsquared_adj:.4f} | "
        f"Observações = {int(modelo.nobs)}",
        "fim"
    )
    log_mensagem(etapa, f"Tabela de resultados OLS salva em '{path_csv}'.", "fim")

    return modelo
