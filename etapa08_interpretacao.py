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

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils_log import log_mensagem


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================
def gerar_graficos(df: pd.DataFrame, modelo_ols=None):
    etapa = "ETAPA 8 - Interpretação e Visualização"
    log_mensagem(etapa, "Gerando gráficos e análises visuais...", "inicio")

    # ============================================================
    # 1) Preparação de diretórios
    # ------------------------------------------------------------
    os.makedirs("resultados/figuras", exist_ok=True)

    # ============================================================
    # 2) Gráfico de dispersão (variável explicativa x índice)
    # ------------------------------------------------------------
    var_explicativa = [c for c in df.columns if c.startswith("TC045Q01NA")]
    if var_explicativa:
        var_x = var_explicativa[0]
        tmp = df[[var_x, "indice_bem_estar_norm"]].copy()
        tmp[var_x] = pd.to_numeric(tmp[var_x], errors="coerce")
        tmp = tmp.dropna(subset=[var_x, "indice_bem_estar_norm"])

        if not tmp.empty:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(
                x=var_x,
                y="indice_bem_estar_norm",
                data=tmp,
                alpha=0.6,
                s=40
            )
            plt.title("Relação entre formação docente e bem-estar")
            plt.xlabel(var_x[:60] + "...")
            plt.ylabel("Índice de Bem-Estar Normalizado")
            plt.subplots_adjust(top=0.9, bottom=0.12)
            path_disp = "resultados/figuras/grafico_dispersao.png"
            plt.savefig(path_disp)
            plt.close()
            log_mensagem(etapa, f"Gráfico de dispersão salvo (variável: {var_x}).", "fim")

    # ============================================================
    # 3) Mapa de calor das correlações numéricas
    # ------------------------------------------------------------
    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(num_df.corr(), cmap="coolwarm", center=0)
        plt.title("Mapa de Calor das Correlações Numéricas")
        plt.subplots_adjust(top=0.9, bottom=0.12)
        path_heat = "resultados/figuras/mapa_calor_correlacoes.png"
        plt.savefig(path_heat)
        plt.close()
        log_mensagem(etapa, "Mapa de calor de correlações salvo.", "fim")

    # ============================================================
    # 4) Boxplot por cluster
    # ------------------------------------------------------------
    if "cluster" in df.columns and "indice_bem_estar_norm" in df.columns:
        tmpb = df[["cluster", "indice_bem_estar_norm"]].dropna().copy()
        if not tmpb.empty:
            plt.figure(figsize=(8, 6))
            sns.boxplot(
                data=tmpb,
                x="cluster",
                y="indice_bem_estar_norm",
                hue="cluster",
                palette="coolwarm",
                legend=False
            )
            plt.title("Distribuição do Bem-Estar por Cluster")
            plt.xlabel("Cluster (Agrupamento de Docentes)")
            plt.ylabel("Índice de Bem-Estar Normalizado")
            plt.subplots_adjust(top=0.9, bottom=0.12)
            path_box = "resultados/figuras/boxplot_cluster.png"
            plt.savefig(path_box)
            plt.close()
            log_mensagem(etapa, "Boxplot por cluster salvo.", "fim")

    # ============================================================
    # 5) Gráfico adicional: Importância das variáveis do modelo OLS
    # ------------------------------------------------------------
    # Objetivo:
    #   - Visualizar as variáveis mais influentes no índice de bem-estar
    #   - Ordenadas pelo valor absoluto do coeficiente estimado
    # ============================================================
    if modelo_ols is not None:
        try:
            coef = modelo_ols.params.drop("const", errors="ignore")
            importancia = pd.DataFrame({
                "Variável": coef.index,
                "Coeficiente": coef.values,
                "Importância_Abs": coef.abs().values
            }).sort_values("Importância_Abs", ascending=False)

            top_vars = importancia.head(15)

            plt.figure(figsize=(10, 7))
            sns.barplot(
                data=top_vars,
                y="Variável",
                x="Coeficiente",
                palette="vlag"
            )
            plt.title("Top 15 Variáveis Mais Influentes no Bem-Estar Docente")
            plt.xlabel("Coeficiente (Peso no Modelo OLS)")
            plt.ylabel("Variável")
            plt.axvline(0, color="gray", linestyle="--", linewidth=1)
            plt.subplots_adjust(top=0.9, bottom=0.12)
            path_imp = "resultados/figuras/importancia_variaveis_ols.png"
            plt.savefig(path_imp, bbox_inches="tight")
            plt.close()
            log_mensagem(etapa, "Gráfico de importância das variáveis salvo.", "fim")

        except Exception as e:
            log_mensagem(etapa, f"[AVISO] Falha ao gerar gráfico de importância: {e}", "fim")

    # ============================================================
    # 6) Finalização da etapa
    # ------------------------------------------------------------
    log_mensagem(etapa, "Visualizações geradas e salvas com sucesso.", "fim")
