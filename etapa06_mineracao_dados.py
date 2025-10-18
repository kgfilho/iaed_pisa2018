# ============================================================
# ETAPA 06 - MINERAÇÃO DE DADOS
# ------------------------------------------------------------
# Objetivo:
#   - Aplicar técnicas de redução de dimensionalidade (PCA)
#     e agrupamento (K-Means) sobre os dados transformados.
#   - Gerar agrupamentos de docentes por padrões de respostas
#     e calcular a variância explicada.
#   - Registrar logs em cada fase do processo.
#
# Saída esperada:
#   - DataFrame enriquecido com colunas PCA1, PCA2 e Cluster.
#   - Modelo KMeans ajustado.
# ============================================================

import logging
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils_log import log_mensagem


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================
def minerar_dados(df: pd.DataFrame):
    """
    Executa a análise de componentes principais (PCA)
    e o agrupamento K-Means para identificar padrões
    nos dados docentes do PISA 2018.
    """

    etapa = "ETAPA 6 - Mineração de Dados"
    log_mensagem(etapa, "Executando PCA e agrupamento (K-Means)...", "inicio")

    # ============================================================
    # 1) Seleção de colunas numéricas válidas
    # ------------------------------------------------------------
    # Apenas variáveis numéricas são adequadas para o PCA e o K-Means.
    # ============================================================
    df_numerico = df.select_dtypes(include=[np.number]).copy()

    if df_numerico.empty:
        raise ValueError("[ERRO] Nenhuma variável numérica disponível para mineração de dados.")

    # ============================================================
    # 2) Padronização dos dados
    # ------------------------------------------------------------
    # Os dados são normalizados (média = 0, desvio = 1) para evitar
    # distorções provocadas por escalas diferentes.
    # ============================================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numerico)

    # ============================================================
    # 3) Aplicação do PCA (Análise de Componentes Principais)
    # ------------------------------------------------------------
    # Reduz a dimensionalidade mantendo a variância essencial.
    # ============================================================
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    variancia = np.sum(pca.explained_variance_ratio_) * 100

    # ============================================================
    # 4) Agrupamento com K-Means
    # ------------------------------------------------------------
    # Define automaticamente 3 clusters (configurável).
    # ============================================================
    modelo_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = modelo_kmeans.fit_predict(X_pca)

    # ============================================================
    # 5) Geração das novas colunas (sem fragmentação)
    # ------------------------------------------------------------
    # Usa concatenação única para evitar PerformanceWarning.
    # ============================================================
    novas_colunas = pd.DataFrame({
        "pca1": X_pca[:, 0],
        "pca2": X_pca[:, 1],
        "cluster": clusters
    }, index=df.index)

    df_final = pd.concat([df.reset_index(drop=True), novas_colunas.reset_index(drop=True)], axis=1)

    # ============================================================
    # 6) Registro e saída
    # ------------------------------------------------------------
    # Exibe informações no log sobre a variância explicada e
    # o número de docentes agrupados.
    # ============================================================
    log_mensagem(
        etapa,
        f"Mineração concluída. {len(df_final)} docentes agrupados em 3 clusters. "
        f"Variância explicada pelos 2 primeiros componentes: {variancia:.2f}%",
        "fim"
    )

    # ============================================================
    # 7) Retorno final
    # ------------------------------------------------------------
    # Retorna somente dois elementos, conforme padrão do pipeline.
    # ============================================================
    return df_final, modelo_kmeans
