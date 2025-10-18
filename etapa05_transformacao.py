# ============================================================
# ETAPA 05 - TRANSFORMAÇÃO DE DADOS
# ------------------------------------------------------------
# Objetivo:
#   - Gerar índices e métricas derivadas das variáveis brutas.
#   - Criar indicadores compostos de bem-estar docente a partir
#     das respostas dos questionários PISA 2018.
#   - Padronizar escalas e preparar os dados para a mineração.
#
# Saída esperada:
#   - DataFrame com novas colunas calculadas (índices e métricas).
# ============================================================

import pandas as pd
import numpy as np
from utils_log import log_mensagem


# ============================================================
# FUNÇÃO AUXILIAR: MAPEADOR ROBUSTO DE CATEGORIAS → NUMÉRICO
# ------------------------------------------------------------
# Regras:
#   - Likert: Strongly Disagree=1 ... Strongly Agree=5
#   - Binário: yes/checked/true/sim/1 => 1 ; no/not checked/false/não/0 => 0
#   - Valores numéricos são preservados (quando possível)
#   - Demais valores → NaN (para não quebrar a média)
# ============================================================
def _mapear_valor(v):
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

    # Binário amplo
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


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================
def transformar_dados(df: pd.DataFrame, cenario: dict) -> pd.DataFrame:
    """
    Transforma os dados brutos do PISA em variáveis derivadas,
    criando índices e indicadores para análises posteriores.
    """

    etapa = "ETAPA 5 - Transformação de Dados"
    log_mensagem(etapa, "Gerando índices derivados e métricas compostas...", "inicio")

    # ============================================================
    # 1) Cópia de segurança e metadados
    # ------------------------------------------------------------
    df = df.copy()
    pais = cenario.get("pais", "Desconhecido")
    tema = cenario.get("tema", "Não definido")

    # ============================================================
    # 2) Seleção das colunas relevantes
    # ------------------------------------------------------------
    colunas_bem_estar = [
        "TC014Q01HA: Did you complete a teacher education or training programme?",
        "TC015Q01NA: How did you receive your initial teaching qualifications?",
        "TC018Q01NA: Included in teacher education, training or other qualification: Reading, writing and literature",
        "TC018Q01NB: Teach it in the <national modal grade for 15-year-olds> in the current school year: Reading, writing and literature",
        "TC018Q02NA: Included in teacher education, training or other qualification: Mathematics",
        "TC018Q02NB: Teach it in the <national modal grade for 15-year-olds> in the current school year: Mathematics",
        "TC018Q03NA: Included in teacher education, training or other qualification: Science",
        "TC018Q03NB: Teach it in the <national modal grade for 15-year-olds> in the current school year: Science",
        "TC018Q04NA: Included in teacher education, training or other qualification: Technology",
        "TC018Q04NB: Teach it in the <national modal grade for 15-year-olds> in the current school year: Technology",
        "TC199Q05HA: In your teaching, to what extent can you do: Motivate students who show low interest in school work"
    ]

    colunas_existentes = [c for c in colunas_bem_estar if c in df.columns]
    if not colunas_existentes:
        raise ValueError("[ERRO] Nenhuma das colunas esperadas foi encontrada na base de dados.")

    # ============================================================
    # 3) Conversão robusta das respostas para valores numéricos
    # ------------------------------------------------------------
    # Aplica o mapeador elemento a elemento e garante dtype float.
    # ============================================================
    sub = df[colunas_existentes].map(_mapear_valor).astype(float)
    
    # ============================================================
    # 4) Geração do índice composto de bem-estar
    # ------------------------------------------------------------
    # Média linha a linha ignorando NaNs.
    # ============================================================
    df["indice_bem_estar"] = sub.mean(axis=1, skipna=True)

    # ============================================================
    # 5) Padronização do índice (escala 0–1)
    # ------------------------------------------------------------
    minimo = df["indice_bem_estar"].min()
    maximo = df["indice_bem_estar"].max()

    if pd.notna(minimo) and pd.notna(maximo) and maximo != minimo:
        df["indice_bem_estar_norm"] = (df["indice_bem_estar"] - minimo) / (maximo - minimo)
    else:
        df["indice_bem_estar_norm"] = df["indice_bem_estar"]

    # ============================================================
    # 6) Criação de faixas interpretativas
    # ------------------------------------------------------------
    df["faixa_bem_estar"] = pd.cut(
        df["indice_bem_estar_norm"],
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["Baixo", "Médio", "Alto"]
    )

    # ============================================================
    # 7) Registro e saída
    # ------------------------------------------------------------
    n_validos = df["indice_bem_estar"].notna().sum()
    colunas_usadas_preview = colunas_existentes[:5] + (["..."] if len(colunas_existentes) > 5 else [])

    log_mensagem(
        etapa,
        f"Transformações concluídas. Índice gerado para {n_validos} docentes. "
        f"Colunas utilizadas: {colunas_usadas_preview}",
        "fim"
    )

    return df
