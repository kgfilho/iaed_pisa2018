# ============================================================
# ETAPA 08 - INTERPRETAÇÃO E VISUALIZAÇÃO
# ------------------------------------------------------------
# Objetivo:
#   - Gerar representações gráficas e visuais dos resultados
#     do modelo OLS e das correlações entre variáveis.
#   - Favorecer a compreensão de padrões e agrupamentos
#     relacionados ao bem-estar docente.
#
# Saída esperada:
#   - Gráficos salvos em 'resultados/figuras/'
# ============================================================
# ============================================================
# ETAPA 08 - INTERPRETAÇÃO E VISUALIZAÇÃO
# ------------------------------------------------------------
# Objetivo:
#   - Gerar gráficos e tabelas estatísticas para interpretação
#     dos resultados obtidos nas etapas anteriores.
#   - Produzir relatório visual resumido em formato Markdown.
# ============================================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
from utils_log import log_mensagem

sns.set(context="notebook", style="whitegrid")


def _garantir_pastas():
    os.makedirs("resultados/figuras", exist_ok=True)
    os.makedirs("resultados/tabelas", exist_ok=True)
    os.makedirs("resultados/relatorios", exist_ok=True)


def _salvar_fig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def gerar_graficos(df: pd.DataFrame, modelo_ols=None, modelo_kmeans=None):
    etapa = "ETAPA 8 - Interpretação e Visualização"
    log_mensagem(etapa, "Iniciando geração de gráficos e tabelas...", "inicio")
    _garantir_pastas()

    # ========================================================
    # A - Indicadores descritivos do índice de bem-estar
    # ========================================================
    try:
        if "indice_bem_estar_norm" in df.columns:
            serie = pd.to_numeric(df["indice_bem_estar_norm"], errors="coerce").dropna()

            plt.figure(figsize=(8, 6))
            sns.histplot(serie, bins=20, kde=False)
            plt.title("Distribuição do Índice de Bem-Estar Docente (Normalizado)")
            plt.xlabel("Índice de Bem-Estar (0–1)")
            plt.ylabel("Frequência")
            _salvar_fig("resultados/figuras/histograma_bem_estar.png")
            log_mensagem(etapa, "Histograma do índice de bem-estar salvo.", "fim")

            plt.figure(figsize=(8, 6))
            sns.kdeplot(serie, fill=True)
            plt.title("Densidade do Índice de Bem-Estar Docente (Normalizado)")
            plt.xlabel("Índice de Bem-Estar (0–1)")
            plt.ylabel("Densidade")
            _salvar_fig("resultados/figuras/densidade_bem_estar.png")
            log_mensagem(etapa, "Gráfico de densidade do índice de bem-estar salvo.", "fim")

            if "faixa_bem_estar" in df.columns:
                contagens = df["faixa_bem_estar"].value_counts(dropna=False)
                plt.figure(figsize=(7, 5))
                contagens.reindex(["Baixo", "Médio", "Alto"]).plot(kind="bar")
                plt.title("Distribuição por Faixa de Bem-Estar")
                plt.xlabel("Faixa")
                plt.ylabel("Número de Docentes")
                _salvar_fig("resultados/figuras/barras_faixa_bem_estar.png")
                contagens.to_csv("resultados/tabelas/faixa_bem_estar_contagens.csv", encoding="utf-8-sig")
                log_mensagem(etapa, "Barras por faixa e tabela de contagens salvas.", "fim")
    except Exception as e:
        log_mensagem(etapa, f"[AVISO] Falha nos gráficos descritivos: {e}", "fim")

    # ========================================================
    # B - Projeção PCA e distribuição de clusters
    # ========================================================
    try:
        if {"pca1", "pca2"}.issubset(df.columns):
            if "cluster" in df.columns:
                tmp = df[["pca1", "pca2", "cluster"]].dropna()
                if not tmp.empty:
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(data=tmp, x="pca1", y="pca2", hue="cluster", s=40)
                    plt.title("Projeção PCA dos Docentes por Cluster")
                    plt.xlabel("PCA 1")
                    plt.ylabel("PCA 2")
                    _salvar_fig("resultados/figuras/pca_clusters.png")
                    log_mensagem(etapa, "Dispersão PCA por cluster salva.", "fim")

                    dist = tmp["cluster"].value_counts().sort_index()
                    plt.figure(figsize=(7, 5))
                    dist.plot(kind="bar")
                    plt.title("Distribuição de Docentes por Cluster")
                    plt.xlabel("Cluster")
                    plt.ylabel("Número de Docentes")
                    _salvar_fig("resultados/figuras/clusters_distribuicao.png")
                    dist.to_csv("resultados/tabelas/clusters_distribuicao.csv", encoding="utf-8-sig")
                    log_mensagem(etapa, "Distribuição de clusters salva.", "fim")

                    if modelo_kmeans is not None and hasattr(modelo_kmeans, "cluster_centers_"):
                        centros = np.asarray(modelo_kmeans.cluster_centers_)
                        if centros.shape[1] >= 2:
                            df_centros = pd.DataFrame(centros[:, :2], columns=["PCA1", "PCA2"])
                            plt.figure(figsize=(6, 3.8))
                            sns.heatmap(df_centros, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
                            plt.title("Centróides dos Clusters (Espaço PCA)")
                            _salvar_fig("resultados/figuras/centroides_clusters_heatmap.png")
                            df_centros.to_csv("resultados/tabelas/centroides_clusters.csv", encoding="utf-8-sig", index_label="cluster")
                            log_mensagem(etapa, "Heatmap de centróides salvo.", "fim")
    except Exception as e:
        log_mensagem(etapa, f"[AVISO] Falha nos gráficos de PCA/Clusters: {e}", "fim")

    # ========================================================
    # C - Mapa de calor de correlações numéricas
    # ========================================================
    try:
        num_df = df.select_dtypes(include="number")
        if not num_df.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(num_df.corr(), cmap="coolwarm", center=0)
            plt.title("Mapa de Calor das Correlações Numéricas")
            _salvar_fig("resultados/figuras/mapa_calor_correlacoes.png")
            num_df.corr().to_csv("resultados/tabelas/correlacoes_principais.csv", encoding="utf-8-sig")
            log_mensagem(etapa, "Mapa de calor de correlações e tabela salvos.", "fim")
    except Exception as e:
        log_mensagem(etapa, f"[AVISO] Falha no mapa de calor: {e}", "fim")

    # ========================================================
    # D - Dispersão entre variável explicativa e bem-estar
    # ========================================================
    try:
        var_explicativa = [c for c in df.columns if c.startswith("TC045Q01NA")]
        if var_explicativa:
            var_x = var_explicativa[0]
            tmp = df[[var_x, "indice_bem_estar_norm"]].copy()
            tmp[var_x] = pd.to_numeric(tmp[var_x], errors="coerce")
            tmp = tmp.dropna(subset=[var_x, "indice_bem_estar_norm"])
            if not tmp.empty:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=var_x, y="indice_bem_estar_norm", data=tmp, alpha=0.6, s=40)
                plt.title("Relação entre Formação Docente e Bem-Estar")
                plt.xlabel(var_x)
                plt.ylabel("Índice de Bem-Estar Normalizado")
                _salvar_fig("resultados/figuras/grafico_dispersao.png")
                log_mensagem(etapa, "Gráfico de dispersão salvo.", "fim")
    except Exception as e:
        log_mensagem(etapa, f"[AVISO] Falha no gráfico de dispersão: {e}", "fim")

    # ========================================================
    # E - Boxplot do bem-estar por cluster
    # ========================================================
    try:
        if "cluster" in df.columns and "indice_bem_estar_norm" in df.columns:
            tmpb = df[["cluster", "indice_bem_estar_norm"]].dropna()
            if not tmpb.empty:
                plt.figure(figsize=(8, 6))
                sns.boxplot(data=tmpb, x="cluster", y="indice_bem_estar_norm", hue="cluster", legend=False)
                plt.title("Distribuição do Bem-Estar por Cluster")
                plt.xlabel("Cluster")
                plt.ylabel("Índice de Bem-Estar Normalizado")
                _salvar_fig("resultados/figuras/boxplot_cluster.png")
                log_mensagem(etapa, "Boxplot por cluster salvo.", "fim")
    except Exception as e:
        log_mensagem(etapa, f"[AVISO] Falha no boxplot: {e}", "fim")

    # ========================================================
    # F - Importância e diagnósticos do modelo OLS
    # ========================================================
    try:
        if modelo_ols is not None:
            coef = modelo_ols.params.drop("const", errors="ignore")
            importancia = pd.DataFrame({
                "Variável": coef.index,
                "Coeficiente": coef.values,
                "Importância_Abs": coef.abs().values
            }).sort_values("Importância_Abs", ascending=False)
            top_vars = importancia.head(20)

            plt.figure(figsize=(10, 8))
            sns.barplot(data=top_vars, y="Variável", x="Coeficiente")
            plt.title("Top 20 Variáveis Mais Influentes no Bem-Estar Docente (OLS)")
            plt.xlabel("Coeficiente")
            plt.ylabel("Variável")
            plt.axvline(0, color="gray", linestyle="--", linewidth=1)
            _salvar_fig("resultados/figuras/importancia_variaveis_ols.png")
            importancia.to_csv("resultados/tabelas/importancia_variaveis_ols.csv", index=False, encoding="utf-8-sig")
            log_mensagem(etapa, "Importância das variáveis do OLS salva.", "fim")

            fitted = modelo_ols.fittedvalues
            resid = modelo_ols.resid

            plt.figure(figsize=(7.5, 6))
            sns.scatterplot(x=fitted, y=resid, s=18, alpha=0.7)
            plt.axhline(0, color="gray", linestyle="--", linewidth=1)
            plt.title("Resíduos vs. Valores Ajustados (OLS)")
            plt.xlabel("Ajustado")
            plt.ylabel("Resíduo")
            _salvar_fig("resultados/figuras/residuos_vs_ajustado.png")
            log_mensagem(etapa, "Gráfico Resíduos vs Ajustado salvo.", "fim")

            plt.figure(figsize=(6.5, 6.5))
            sm.qqplot(resid, line="45", fit=True)
            plt.title("Q-Q Plot dos Resíduos (OLS)")
            _salvar_fig("resultados/figuras/qqplot_residuos.png")
            log_mensagem(etapa, "Q-Q Plot dos resíduos salvo.", "fim")

            if "indice_bem_estar_norm" in df.columns:
                y_obs = pd.to_numeric(df["indice_bem_estar_norm"], errors="coerce")
                comp = pd.DataFrame({"observado": y_obs, "ajustado": fitted}).dropna()
                if not comp.empty:
                    plt.figure(figsize=(7.5, 6))
                    sns.scatterplot(x=comp["observado"], y=comp["ajustado"], s=18, alpha=0.7)
                    lims = [min(comp.min()), max(comp.max())]
                    plt.plot(lims, lims, linestyle="--", color="gray", linewidth=1)
                    plt.title("Predito vs Observado (OLS)")
                    plt.xlabel("Observado")
                    plt.ylabel("Predito")
                    _salvar_fig("resultados/figuras/predito_vs_observado.png")
                    comp.to_csv("resultados/tabelas/predito_observado_ols.csv", index=False, encoding="utf-8-sig")
                    log_mensagem(etapa, "Predito vs Observado salvo.", "fim")

            pd.DataFrame({"ajustado": fitted, "residuo": resid}).to_csv(
                "resultados/tabelas/residuos_modelo_ols.csv", index=False, encoding="utf-8-sig"
            )
    except Exception as e:
        log_mensagem(etapa, f"[AVISO] Falha nos diagnósticos do OLS: {e}", "fim")

    # ========================================================
    # G - Estatísticas descritivas gerais
    # ========================================================
    try:
        desc = df.select_dtypes(include="number").describe().T
        desc.to_csv("resultados/tabelas/estatisticas_descritivas.csv", encoding="utf-8-sig")
        log_mensagem(etapa, "Tabela de estatísticas descritivas salva.", "fim")
    except Exception as e:
        log_mensagem(etapa, f"[AVISO] Falha ao salvar estatísticas descritivas: {e}", "fim")

    # ========================================================
    # H - Relatório visual em Markdown
    # ========================================================
    try:
        caminho_relatorio = "resultados/relatorios/resumo_visual.md"
        conteudo = []
        conteudo.append("# 📊 Relatório Visual – Bem-Estar Docente no Chile\n")
        conteudo.append(f"**Data de geração:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        conteudo.append("Este relatório apresenta, de forma resumida, os principais resultados visuais obtidos pelo pipeline de análise do bem-estar docente.\n")
        conteudo.append("---\n")
        conteudo.append("## 🔹 Indicadores Descritivos\n")
        conteudo.append("- [Histograma do índice de bem-estar](../figuras/histograma_bem_estar.png)\n")
        conteudo.append("- [Densidade do índice de bem-estar](../figuras/densidade_bem_estar.png)\n")
        conteudo.append("- [Distribuição por faixa de bem-estar](../figuras/barras_faixa_bem_estar.png)\n")
        conteudo.append("- [Mapa de calor das correlações](../figuras/mapa_calor_correlacoes.png)\n")
        conteudo.append("## 🔹 Mineração de Dados (PCA e Clusters)\n")
        conteudo.append("- [Projeção PCA por cluster](../figuras/pca_clusters.png)\n")
        conteudo.append("- [Distribuição de docentes por cluster](../figuras/clusters_distribuicao.png)\n")
        conteudo.append("- [Centróides dos clusters](../figuras/centroides_clusters_heatmap.png)\n")
        conteudo.append("- [Boxplot por cluster](../figuras/boxplot_cluster.png)\n")
        conteudo.append("## 🔹 Modelagem e Diagnóstico (OLS)\n")
        conteudo.append("- [Importância das variáveis OLS](../figuras/importancia_variaveis_ols.png)\n")
        conteudo.append("- [Resíduos vs Ajustado](../figuras/residuos_vs_ajustado.png)\n")
        conteudo.append("- [Q-Q Plot dos resíduos](../figuras/qqplot_residuos.png)\n")
        conteudo.append("- [Predito vs Observado](../figuras/predito_vs_observado.png)\n")
        conteudo.append("## 🔹 Tabelas Complementares\n")
        conteudo.append("- [Estatísticas descritivas](../tabelas/estatisticas_descritivas.csv)\n")
        conteudo.append("- [Variáveis significativas](../tabelas/variaveis_significativas.csv)\n")
        conteudo.append("- [Modelo OLS completo](../tabelas/modelo_ols_resultados.csv)\n")
        conteudo.append("- [Importância das variáveis OLS](../tabelas/importancia_variaveis_ols.csv)\n")
        conteudo.append("---\n")
        conteudo.append("### Observação Geral\n")
        conteudo.append("As figuras e tabelas reunidas neste relatório permitem observar padrões de bem-estar docente e fatores explicativos relevantes identificados nas etapas de mineração e modelagem.\n")

        with open(caminho_relatorio, "w", encoding="utf-8") as f:
            f.write("\n".join(conteudo))

        log_mensagem(etapa, f"Relatório visual resumido salvo em {caminho_relatorio}", "fim")
    except Exception as e:
        log_mensagem(etapa, f"[AVISO] Falha ao gerar resumo visual: {e}", "fim")

    log_mensagem(etapa, "Visualizações e tabelas geradas com sucesso.", "fim")
