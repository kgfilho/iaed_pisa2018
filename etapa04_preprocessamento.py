# ============================================================
# ETAPA 04 - PRÉ-PROCESSAMENTO (LIMPEZA E TRATAMENTO)
# ------------------------------------------------------------
# Objetivo:
#   - Realizar a limpeza dos dados brutos provenientes da
#     base PISA 2018.
#   - Tratar valores ausentes, tipos incorretos e inconsistências.
#   - Garantir que apenas registros válidos e completos sigam
#     para a próxima etapa (Transformação).
#
# Saída esperada:
#   - DataFrame limpo e padronizado.
# ============================================================

import pandas as pd
import numpy as np
from utils_log import log_mensagem


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================
def preprocessar_dados(df: pd.DataFrame, cenario: dict) -> pd.DataFrame:
    """
    Limpa e prepara os dados para as etapas seguintes do pipeline.
    """

    etapa = "ETAPA 4 - Pré-processamento (Limpeza e Tratamento)"
    log_mensagem(etapa, "Iniciando limpeza de dados...", "inicio")

    # ============================================================
    # 1) Cópia de segurança e metadados do cenário
    # ------------------------------------------------------------
    # Trabalha sobre uma cópia para preservar o original.
    # ============================================================
    df = df.copy()
    pais = cenario.get("pais", "Desconhecido")
    tema = cenario.get("tema", "Não definido")

    # ============================================================
    # 2) Remoção de colunas completamente vazias
    # ------------------------------------------------------------
    # Elimina variáveis que não contêm nenhuma informação útil.
    # ============================================================
    df.dropna(axis=1, how="all", inplace=True)

    # ============================================================
    # 3) Tratamento de valores ausentes
    # ------------------------------------------------------------
    # Substitui valores ausentes por preenchimento anterior (forward fill).
    # Substitui valores restantes por zero, garantindo consistência.
    # ============================================================
    df = df.ffill().fillna(0)

    # ============================================================
    # 4) Padronização de tipos numéricos
    # ------------------------------------------------------------
    # Converte colunas para tipo numérico sempre que possível,
    # sem usar o parâmetro deprecated "errors='ignore'".
    # ============================================================
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            # Mantém a coluna original se não for conversível
            continue

    # ============================================================
    # 5) Normalização de texto e categorias
    # ------------------------------------------------------------
    # Remove espaços e padroniza capitalização de variáveis textuais.
    # ============================================================
    df_obj = df.select_dtypes(include=["object"]).columns
    for col in df_obj:
        df[col] = df[col].astype(str).str.strip().str.title()

    # ============================================================
    # 6) Remoção de duplicatas
    # ------------------------------------------------------------
    # Elimina linhas repetidas para evitar distorções estatísticas.
    # ============================================================
    df.drop_duplicates(inplace=True)

    # ============================================================
    # 7) Filtro de registros inválidos (exemplo: sem país definido)
    # ------------------------------------------------------------
    # Garante que apenas registros válidos permaneçam na base.
    # ============================================================
    if "CNT" in df.columns:
        df = df[df["CNT"].notna()]

    # ============================================================
    # 8) Registro e resumo estatístico
    # ------------------------------------------------------------
    # Calcula quantidade final de registros válidos e loga no console.
    # ============================================================
    linhas_restantes = len(df)
    log_mensagem(etapa, f"Dados limpos: {linhas_restantes} linhas restantes.", "fim")

    return df
