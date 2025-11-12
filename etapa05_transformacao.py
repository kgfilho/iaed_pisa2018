# ============================================================
# ETAPA 05 - TRANSFORMAÇÃO DE DADOS (VERSÃO MODIFICADA)
# ------------------------------------------------------------
# - (MODIFICADO) Adiciona o Passo 6 para salvar um JSON
#   contendo quais colunas foram usadas para criar cada índice
#   (para ser usado pela Etapa 11).
# ============================================================

import pandas as pd
import numpy as np
from utils_log import log_mensagem
import json
from pathlib import Path


# ============================================================
# FUNÇÃO AUXILIAR: MAPEADOR ROBUSTO DE CATEGORIAS → NUMÉRICO
# ============================================================
def _mapear_valor(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()

    # Escalas Likert (1-4, 1-5, etc.)
    likert = {
        # Escala 1-4 (Discordo/Concordo)
        "strongly disagree": 1,
        "disagree": 2,
        "agree": 3,
        "strongly agree": 4,
        
        # Escala 1-4 (Extensão / Frequência)
        "not at all": 1,
        "very little": 2,
        "to some extent": 3,
        "to a large extent": 4,
        "never or hardly ever": 1,
        "several times a year": 2,
        "several times a month": 3,
        "several times a week": 4,
    }
    if s in likert:
        return float(likert[s])

    # Binário amplo
    positivos = {"checked", "yes", "sim", "true", "1", "y", "s", "completed"}
    negativos = {"not checked", "no", "não", "nao", "false", "0", "n", "not completed"}

    if s in positivos:
        return 1.0
    if s in negativos:
        return 0.0

    # Tentativa de conversão numérica direta
    try:
        return float(s)
    except Exception:
        return np.nan

# ##########################################################################
# ### INÍCIO DAS LISTAS DE CÓDIGOS (PREFIXOS) ###
# ##########################################################################

# 1. ALVO (Y): CÓDIGOS DE AUTOEFICÁCIA (Bloco TC199)
colunas_autoeficacia_codigos = [
    "TC199Q01HA", "TC199Q02HA", "TC199Q03HA", "TC199Q04HA", "TC199Q05HA",
    "TC199Q06HA", "TC199Q07HA", "TC199Q08HA", "TC199Q09HA", "TC199Q10HA",
    "TC199Q11HA", "TC199Q12HA"
]

# 2. PREDITOR (X): CÓDIGOS DE FORMAÇÃO CONTINUADA (Bloco TC045)
colunas_formacao_codigos = [
    "TC045Q01NB", "TC045Q02NB", "TC045Q03NB", "TC045Q04NB", "TC045Q05NB",
    "TC045Q06NB", "TC045Q07NB", "TC045Q08NB", "TC045Q09NB", "TC045Q10NB",
    "TC045Q11NB", "TC045Q12NB", "TC045Q13NB", "TC045Q14NB", "TC045Q15NB",
    "TC045Q16HB", "TC045Q17NB"
] # Nota: Seu arquivo parece ter até 18HB, mas 17NB é o padrão PISA

# 3. PREDITOR (X): CÓDIGOS DE OBSTÁCULOS/CARGA (Bloco TC028)
colunas_carga_codigos = [
    "TC028Q01NA", "TC028Q02NA", "TC028Q03NA", "TC028Q04NA",
    "TC028Q05NA", "TC028Q06NA", "TC028Q07NA", "TC028Q08NA"
]

# 4. PREDITOR (X): CÓDIGOS DE CLIMA DISCIPLINAR (Bloco TC170)
colunas_clima_codigos = [
    "TC170Q01HA", "TC170Q02HA", "TC170Q03HA", "TC170Q04HA", "TC170Q05HA"
]

# 5. PREDITOR (X): CÓDIGOS DE COOPERAÇÃO DOCENTE (Blocos TC046 e TC031)
colunas_cooperacao_codigos = [
    "TC046Q04NA", "TC046Q05NA", "TC046Q06NA", "TC046Q07NA",
    "TC031Q04NA", "TC031Q11NA", "TC031Q13NA", "TC031Q14NA",
    "TC031Q15NA", "TC031Q18NA", "TC031Q20NA"
]

# 6. PREDITOR (X): CÓDIGOS DE SATISFAÇÃO NO TRABALHO (Bloco TC198)
colunas_satisfacao_codigos = [
    "TC198Q01HA", "TC198Q02HA", "TC198Q03HA", "TC198Q04HA", "TC198Q05HA",
    "TC198Q06HA", "TC198Q07HA", "TC198Q08HA", "TC198Q09HA", "TC198Q10HA"
]
colunas_satisfacao_negativas = ["TC198Q03HA", "TC198Q04HA", "TC198Q06HA"]

# ##########################################################################
# ### FIM DAS LISTAS DE CÓDIGOS (PREFIXOS) ###
# ##########################################################################


# ============================================================
# FUNÇÃO AUXILIAR: Encontra o nome real da coluna pelo prefixo
# ============================================================
def _encontrar_nomes_reais(colunas_df: list, codigos_prefixo: list) -> list:
    nomes_encontrados = []
    colunas_df_lower = {col.lower(): col for col in colunas_df}
    
    for codigo in codigos_prefixo:
        codigo_lower = codigo.lower()
        matches = [col for col_lower, col in colunas_df_lower.items() if col_lower.startswith(codigo_lower)]
        
        if matches:
            nomes_encontrados.append(matches[0])
            
    return nomes_encontrados


# ============================================================
# FUNÇÃO PRINCIPAL (MODIFICADA)
# ============================================================
def transformar_dados(df: pd.DataFrame, cenario: dict) -> pd.DataFrame:
    etapa = "ETAPA 5 - Transformação de Dados"
    log_mensagem(etapa, "Gerando índices (Engenharia de Features)...", "inicio")

    # 1) Cópia de segurança e lista de colunas
    df = df.copy()
    colunas_reais_df = list(df.columns)

    # 2) Encontrar os nomes REAIS das colunas
    nomes_autoeficacia = _encontrar_nomes_reais(colunas_reais_df, colunas_autoeficacia_codigos)
    nomes_formacao = _encontrar_nomes_reais(colunas_reais_df, colunas_formacao_codigos)
    nomes_carga = _encontrar_nomes_reais(colunas_reais_df, colunas_carga_codigos)
    nomes_clima = _encontrar_nomes_reais(colunas_reais_df, colunas_clima_codigos)
    nomes_cooperacao = _encontrar_nomes_reais(colunas_reais_df, colunas_cooperacao_codigos)
    nomes_satisfacao = _encontrar_nomes_reais(colunas_reais_df, colunas_satisfacao_codigos)
    nomes_satisfacao_neg = _encontrar_nomes_reais(colunas_reais_df, colunas_satisfacao_negativas)

    colunas_para_mapear = list(set(
        nomes_autoeficacia + nomes_formacao + nomes_carga + 
        nomes_clima + nomes_cooperacao + nomes_satisfacao
    ))

    if not colunas_para_mapear:
        log_mensagem(etapa, f"Nenhuma das colunas PISA (TC199, TC045, etc.) foi encontrada.", "erro")
        raise ValueError("[ERRO] Nenhuma das colunas PISA esperadas foi encontrada. Verifique os dados.")
    else:
        log_mensagem(etapa, f"Encontradas {len(colunas_para_mapear)} colunas PISA para engenharia de features.", "info")

    # 3) Conversão robusta das respostas para valores numéricos
    sub = df[colunas_para_mapear].map(_mapear_valor).astype(float)

    # 4) Geração dos Índices Compostos
    
    # 4.1. ALVO (Y): Autoeficácia
    if nomes_autoeficacia:
        df["indice_autoeficacia"] = sub.reindex(columns=nomes_autoeficacia).mean(axis=1, skipna=True)
    else:
        log_mensagem(etapa, "Nenhuma coluna de Autoeficácia (TC199) encontrada.", "erro")

    # 4.2. PREDITOR (X): Formação
    if nomes_formacao:
        df["formacao_continuada_soma"] = sub.reindex(columns=nomes_formacao).sum(axis=1, skipna=True)
    else:
        log_mensagem(etapa, "Nenhuma coluna de Formação (TC045) encontrada.", "aviso")

    # 4.3. PREDITOR (X): Obstáculos/Carga
    if nomes_carga:
        obstaculos_media = sub.reindex(columns=nomes_carga).mean(axis=1, skipna=True)
        df["carga_trabalho_media"] = 5 - obstaculos_media # Escala Invertida
        log_mensagem(etapa, "Índice 'carga_trabalho_media' (TC028) criado.", "info")
    else:
        log_mensagem(etapa, "Nenhuma coluna de Carga/Obstáculos (TC028) encontrada.", "aviso")

    # 4.4. PREDITOR (X): Clima Disciplinar
    if nomes_clima:
        clima_negativo_media = sub.reindex(columns=nomes_clima).mean(axis=1, skipna=True)
        df["clima_media"] = 5 - clima_negativo_media # Escala Invertida
        log_mensagem(etapa, "Índice 'clima_media' (TC170) criado.", "info")
    else:
        log_mensagem(etapa, "Nenhuma coluna de Clima Disciplinar (TC170) encontrada.", "aviso")

    # 4.5. PREDITOR (X): Cooperação
    if nomes_cooperacao:
        df["cooperacao_media"] = sub.reindex(columns=nomes_cooperacao).mean(axis=1, skipna=True)
        log_mensagem(etapa, "Índice 'cooperacao_media' (TC046, TC031) criado.", "info")
    else:
        log_mensagem(etapa, "Nenhuma coluna de Cooperação (TC046, TC031) encontrada.", "aviso")

    # 4.6. PREDITOR (X): Satisfação
    if nomes_satisfacao:
        sub_satisfacao = sub.reindex(columns=nomes_satisfacao).copy()
        for col_neg in nomes_satisfacao_neg:
            if col_neg in sub_satisfacao:
                sub_satisfacao[col_neg] = 5 - sub_satisfacao[col_neg]
        df["satisfacao_media"] = sub_satisfacao.mean(axis=1, skipna=True)
        log_mensagem(etapa, "Índice 'satisfacao_media' (TC198) criado.", "info")
    else:
        log_mensagem(etapa, "Nenhuma coluna de Satisfação (TC198) encontrada.", "aviso")

    # 5) Padronização do ALVO
    alvo = "indice_autoeficacia"
    if alvo in df.columns:
        minimo = df[alvo].min()
        maximo = df[alvo].max()

        if pd.notna(minimo) and pd.notna(maximo) and maximo != minimo:
            df[f"{alvo}_norm"] = (df[alvo] - minimo) / (maximo - minimo)
        else:
            df[f"{alvo}_norm"] = df[alvo]
        
        n_validos = df[alvo].notna().sum()
        log_mensagem(etapa, f"Transformações concluídas. Índice alvo '{alvo}' gerado para {n_validos} docentes.", "fim")
    else:
        log_mensagem(etapa, f"Índice alvo '{alvo}' não pôde ser calculado.", "erro")
        raise ValueError(f"Índice alvo '{alvo}' não pôde ser gerado.")

    # ####################################################################
    # # ### INÍCIO DA MODIFICAÇÃO (Salvar Composição) ###
    # ####################################################################
    
    # 6) Salvar a composição dos índices para a Etapa 11 (LLM)
    try:
        composicao = {
            "indice_autoeficacia": nomes_autoeficacia,
            "formacao_continuada_soma": nomes_formacao,
            "carga_trabalho_media": nomes_carga,
            "clima_media": nomes_clima,
            "cooperacao_media": nomes_cooperacao,
            "satisfacao_media": nomes_satisfacao
        }
        
        # Remove listas vazias (caso um índice não tenha sido encontrado)
        composicao = {k: v for k, v in composicao.items() if v}

        caminho_tabelas = Path("resultados/tabelas")
        caminho_tabelas.mkdir(parents=True, exist_ok=True)
        caminho_composicao = caminho_tabelas / "composicao_indices.json"
        
        with open(caminho_composicao, 'w', encoding='utf-8') as f:
            json.dump(composicao, f, indent=2, ensure_ascii=False)
            
        log_mensagem(etapa, f"Composição dos índices salva em '{caminho_composicao}'", "info")
            
    except Exception as e:
        log_mensagem(etapa, f"Falha ao salvar composição dos índices: {e}", "aviso")
    
    # ####################################################################
    # # ### FIM DA MODIFICAÇÃO ###
    # ####################################################################

    return df