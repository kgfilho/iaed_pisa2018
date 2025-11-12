# ============================================================
# ETAPA 06 - MINERAÇÃO DE DADOS (VERSÃO CORRIGIDA)
# ------------------------------------------------------------
# Objetivo:
#   - Aplicar PCA e K-Means.
#   - (CORRIGIDO) Lidar com valores NaN gerados na Etapa 5
#     (ex: índices de docentes que não responderam)
#     para evitar falhas no PCA.
# ============================================================

import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils_log import log_mensagem


# ============================================================
# FUNÇÃO PRINCIPAL (MODIFICADA)
# ============================================================
def minerar_dados(df: pd.DataFrame):
    """
    Executa a análise de componentes principais (PCA)
    e o agrupamento K-Means para identificar padrões
    nos dados docentes do PISA 2018.

    (Versão corrigida para tratar NaNs antes do PCA)
    """

    etapa = "ETAPA 6 - Mineração de Dados"
    log_mensagem(etapa, "Executando PCA e agrupamento (K-Means)...", "inicio")

    df_original = df # Preserva o DataFrame original
    
    # ============================================================
    # 1) Seleção de colunas numéricas válidas
    # ------------------------------------------------------------
    # Apenas variáveis numéricas são adequadas para o PCA e o K-Means.
    # ============================================================
    df_numerico = df.select_dtypes(include=[np.number]).copy()

    if df_numerico.empty:
        raise ValueError("[ERRO] Nenhuma variável numérica disponível para mineração de dados.")

    # ####################################################################
    # ### INÍCIO DA CORREÇÃO (Tratamento de NaNs) ###
    # ####################################################################

    # ============================================================
    # 2) Identificar linhas com dados numéricos completos
    # ------------------------------------------------------------
    # O PCA não aceita NaNs. Vamos rodar a mineração apenas
    # nos registros que estão 100% completos.
    # ============================================================
    
    # Guarda o índice das linhas que NÃO têm NaNs
    idx_completos = df_numerico.dropna().index
    df_numerico_completo = df_numerico.loc[idx_completos]
    
    n_total = len(df_numerico)
    n_completos = len(df_numerico_completo)
    n_removidos = n_total - n_completos
    
    if n_completos == 0:
        raise ValueError("[ERRO] Nenhum registro completo (sem NaNs) encontrado para o PCA. Verifique a Etapa 5.")
        
    log_mensagem(etapa, f"Mineração usará {n_completos} de {n_total} registros (removendo {n_removidos} com NaNs).", "info")


    # ============================================================
    # 3) Padronização dos dados (APENAS nos dados completos)
    # ------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numerico_completo)

    # ============================================================
    # 4) Aplicação do PCA (Análise de Componentes Principais)
    # ------------------------------------------------------------
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled) # Agora X_scaled não tem NaNs
    variancia = np.sum(pca.explained_variance_ratio_) * 100

    # ============================================================
    # 5) Agrupamento com K-Means
    # ------------------------------------------------------------
    modelo_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = modelo_kmeans.fit_predict(X_pca)

    # ============================================================
    # 6) Geração das novas colunas (alocação segura)
    # ------------------------------------------------------------
    # Criamos as colunas com NaN no DataFrame original...
    # ... e preenchemos apenas os índices que foram processados.
    # ============================================================
    
    # Criar colunas vazias (com NaN)
    df_original["pca1"] = np.nan
    df_original["pca2"] = np.nan
    df_original["cluster"] = np.nan

    # Preencher apenas as linhas que tinham dados completos
    df_original.loc[idx_completos, "pca1"] = X_pca[:, 0]
    df_original.loc[idx_completos, "pca2"] = X_pca[:, 1]
    df_original.loc[idx_completos, "cluster"] = clusters

    # ####################################################################
    # ### FIM DA CORREÇÃO ###
    # ####################################################################

    # ============================================================
    # 7) Registro e saída
    # ------------------------------------------------------------
    log_mensagem(
        etapa,
        f"Mineração concluída. {n_completos} docentes agrupados em 3 clusters. "
        f"Variância explicada pelos 2 primeiros componentes: {variancia:.2f}%",
        "fim"
    )

    # ============================================================
    # 8) Retorno final
    # ------------------------------------------------------------
    return df_original, modelo_kmeans