# ============================================================
# ETAPA 08 – INTERPRETACAO E VISUALIZACAO (VERSÃO CORRIGIDA)
# ------------------------------------------------------------
# - (CORRIGIDO) Atualiza a função _alvo para encontrar
#   o novo índice 'indice_autoeficacia_norm' criado na Etapa 5.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_log import log_mensagem


# ============================== utilidades ==============================

def _garantir_pastas():
    Path("resultados/figuras").mkdir(parents=True, exist_ok=True)
    Path("resultados/tabelas").mkdir(parents=True, exist_ok=True)
    Path("resultados/relatorios").mkdir(parents=True, exist_ok=True)

def _num(s: pd.Series) -> pd.Series:
    """Converte Series para numerico com NaN em valores invalidos."""
    return pd.to_numeric(s, errors="coerce")

# ####################################################################
# ### INÍCIO DA CORREÇÃO ###
# ####################################################################
def _alvo(df: pd.DataFrame) -> str:
    """
    Localiza a variável alvo (Y) para gerar os gráficos.
    Priorizamos o novo índice de autoeficácia.
    """
    # Lista de prioridade, começando com o novo alvo (autoeficácia)
    alvos_preferidos = [
        "indice_autoeficacia_norm", # Nosso novo alvo, normalizado
        "indice_autoeficacia",      # Nosso novo alvo, bruto
        "indice_bem_estar_norm",    # Antigo (fallback)
        "ibd",                      # Antigo (fallback)
        "indice_bem_estar"          # Antigo (fallback)
    ]
    for c in alvos_preferidos:
        if c in df.columns:
            return c
    
    # Mensagem de erro atualizada
    raise ValueError("Nao foi possivel localizar a coluna do indice (ex: 'indice_autoeficacia_norm').")
# ####################################################################
# ### FIM DA CORREÇÃO ###
# ####################################################################


def _salvar_tabela(df: pd.DataFrame, caminho: str):
    df.to_csv(caminho, index=False, encoding="utf-8-sig")


# ============================== visualizacoes ==============================

def gerar_visualizacoes(respostas: pd.DataFrame):
    etapa = "ETAPA 8 - Interpretação e Visualização"
    log_mensagem(etapa, "Iniciando geração de gráficos e tabelas...", "inicio")
    _garantir_pastas()

    # Agora a função _alvo() encontrará "indice_autoeficacia_norm"
    alvo = _alvo(respostas)
    df = respostas.copy()
    
    log_mensagem(etapa, f"Gerando gráficos para a variável alvo: '{alvo}'", "info")


    # ———————————— Histograma e Densidade do índice ————————————
    try:
        bem = _num(df[alvo]).dropna()
        
        # Mudar os títulos dos gráficos para o novo alvo
        titulo_grafico = f"Distribuição do '{alvo}'"
        xlabel_grafico = f"Índice ({alvo})"

        plt.figure(figsize=(8, 4))
        plt.hist(bem.values, bins=30)
        plt.title(titulo_grafico)
        plt.xlabel(xlabel_grafico); plt.ylabel("Frequência")
        plt.tight_layout(); plt.savefig(f"resultados/figuras/histograma_{alvo}.png"); plt.close()
        log_mensagem(etapa, f"Histograma '{alvo}' salvo.", "info")

        plt.figure(figsize=(8, 4))
        bem.plot(kind="kde")
        plt.title(f"Densidade do '{alvo}'")
        plt.xlabel(xlabel_grafico)
        plt.tight_layout(); plt.savefig(f"resultados/figuras/densidade_{alvo}.png"); plt.close()
        log_mensagem(etapa, f"Gráfico de densidade '{alvo}' salvo.", "info")
    except Exception as e:
        log_mensagem(etapa, f"Falha ao gerar histograma/densidade: {e}", "aviso")
        pass

    # ———————————— Barras por faixa e contagens ————————————
    # (Esta seção será pulada, pois removemos 'faixa_bem_estar' na Etapa 5)
    try:
        if "faixa_bem_estar" in df.columns:
            cont = df["faixa_bem_estar"].value_counts(dropna=False).sort_index()
            cont_df = cont.rename_axis("faixa").reset_index(name="quantidade")
            _salvar_tabela(cont_df, "resultados/tabelas/contagem_faixas_bem_estar.csv")

            x = np.arange(len(cont.index))
            plt.figure(figsize=(7, 4))
            plt.bar(x, cont.values)
            plt.xticks(x, cont.index.astype(str))
            plt.title("Distribuição por faixa")
            plt.xlabel("Faixa"); plt.ylabel("Quantidade")
            plt.tight_layout(); plt.savefig("resultados/figuras/barras_faixa.png"); plt.close()
            log_mensagem(etapa, "Barras por faixa e tabela de contagens salvas.", "info")
    except Exception:
        pass

    # ———————————— PCA por cluster (se existir) ————————————
    # (Sem alterações)
    try:
        if {"PCA1", "PCA2"}.issubset(df.columns):
            x = _num(df["PCA1"]).values
            y = _num(df["PCA2"]).values
            c = df["cluster"] if "cluster" in df.columns else None

            plt.figure(figsize=(6, 6))
            if c is not None:
                clusters = sorted(pd.Series(c).dropna().unique())
                for k in clusters:
                    mask = (c == k)
                    plt.scatter(x[mask], y[mask], s=18, alpha=0.7, label=f"Cluster {k}")
                plt.legend()
            else:
                plt.scatter(x, y, s=18, alpha=0.7)
            plt.xlabel("PCA1"); plt.ylabel("PCA2"); plt.title("PCA por cluster")
            plt.tight_layout(); plt.savefig("resultados/figuras/pca_clusters.png"); plt.close()
            log_mensagem(etapa, "Dispersão PCA por cluster salva.", "info")
    except Exception:
        pass

    # ———————————— Distribuição de clusters (barras) ————————————
    # (Sem alterações)
    try:
        if "cluster" in df.columns:
            contc = pd.Series(df["cluster"]).value_counts(dropna=False).sort_index()
            contc_df = contc.rename_axis("cluster").reset_index(name="quantidade")
            _salvar_tabela(contc_df, "resultados/tabelas/distribuicao_clusters.csv")

            x = np.arange(len(contc.index))
            plt.figure(figsize=(7, 4))
            plt.bar(x, contc.values)
            plt.xticks(x, contc.index.astype(str))
            plt.title("Distribuição de clusters")
            plt.xlabel("Cluster"); plt.ylabel("Quantidade")
            plt.tight_layout(); plt.savefig("resultados/figuras/clusters_distribuicao.png"); plt.close()
            log_mensagem(etapa, "Distribuição de clusters salva.", "info")

    except Exception:
        pass

    # ———————————— Mapa de calor de correlações ————————————
    # (Sem alterações)
    try:
        num_df = df.select_dtypes(include=["number"]).copy()
        num_df = num_df.apply(pd.to_numeric, errors="coerce")
        corr = num_df.corr()

        plt.figure(figsize=(20, 16))
        im = plt.imshow(corr.values, aspect="auto", cmap="viridis")
        plt.colorbar(im, fraction=0.046, pad=0.04)

        plt.xticks(
            range(len(corr.columns)),
            corr.columns,
            rotation=45,
            ha='right',
            fontsize=8
        )
        plt.yticks(
            range(len(corr.index)),
            corr.index,
            fontsize=8
        )

        plt.title("Mapa de calor de correlações entre variáveis", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig("resultados/figuras/mapa_calor_correlacoes.png", dpi=300, bbox_inches="tight")
        plt.close()

        _salvar_tabela(
            corr.reset_index().rename(columns={"index": "variavel"}),
            "resultados/tabelas/correlacoes.csv"
        )

        log_mensagem(etapa, "Mapa de calor de correlações e tabela salvos.", "info")

    except Exception as e:
        log_mensagem(etapa, f"Falha ao gerar mapa de calor: {e}", "erro")
        
    # ———————————— Boxplot por cluster ————————————
    try:
        if "cluster" in df.columns:
            bem = _num(df[alvo])
            grupos = [ _num(bem[df["cluster"] == k]).dropna().values
                       for k in sorted(pd.Series(df["cluster"]).dropna().unique()) ]
            plt.figure(figsize=(8, 5))
            plt.boxplot(grupos, showfliers=False)
            plt.xticks(
                ticks=range(1, len(grupos) + 1),
                labels=[f"Cluster {k}" for k in sorted(pd.Series(df["cluster"]).dropna().unique())]
            )
            plt.title(f"Índice '{alvo}' por cluster")
            plt.ylabel(f"Índice ({alvo})")
            plt.tight_layout(); plt.savefig("resultados/figuras/boxplot_cluster.png"); plt.close()
            log_mensagem(etapa, "Boxplot por cluster salvo.", "info")
    except Exception:
        pass

    # ———————————— Estatísticas descritivas do índice ————————————
    try:
        stats = _num(df[alvo]).describe().to_frame(name=alvo)
        _salvar_tabela(stats.reset_index().rename(columns={"index": "estatistica"}),
                       f"resultados/tabelas/estatisticas_{alvo}.csv")
        log_mensagem(etapa, "Tabela de estatísticas descritivas salva.", "info")
    except Exception:
        pass

    # ———————————— Relatório visual resumido ————————————
    # (Atualizado para refletir os novos nomes de arquivo)
    try:
        resumo = [
            "# Resumo visual do estudo",
            "",
            f"## Alvo da Análise: {alvo}",
            "",
            "## Gráficos gerados",
            f"- `figuras/histograma_{alvo}.png`",
            f"- `figuras/densidade_{alvo}.png`",
            "- `figuras/barras_faixa.png` (se houver `faixa_bem_estar`)",
            "- `figuras/pca_clusters.png` (se houver `PCA1` e `PCA2`)",
            "- `figuras/clusters_distribuicao.png` (se houver `cluster`)",
            "- `figuras/mapa_calor_correlacoes.png`",
            "- `figuras/boxplot_cluster.png` (se houver `cluster`)",
            "",
            "## Tabelas geradas",
            "- `tabelas/contagem_faixas_bem_estar.csv`",
            "- `tabelas/distribuicao_clusters.csv`",
            "- `tabelas/correlacoes.csv`",
            f"- `tabelas/estatisticas_{alvo}.csv`",
        ]
        Path("resultados/relatorios/resumo_visual.md").write_text("\n".join(resumo), encoding="utf-8")
        log_mensagem(etapa, "Relatório visual resumido salvo em resultados/relatorios/resumo_visual.md", "info")
    except Exception:
        pass

    log_mensagem(etapa, "Visualizações e tabelas geradas com sucesso.", "fim")
    return True


# --- Compatibilidade com o main.py antigo ---
def gerar_graficos(respostas: pd.DataFrame):
    """Alias para manter compatibilidade com o main.py."""
    return gerar_visualizacoes(respostas)