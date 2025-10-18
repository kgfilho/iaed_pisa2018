# ============================================================
# ETAPA 3 – COLETA DE DADOS (PISA 2018)
# ------------------------------------------------------------
# Objetivo:
#   - Localizar e carregar automaticamente os arquivos de respostas e questionário.
#   - Aceita formatos CSV ou XLSX.
#   - Detecta delimitador correto e emite logs detalhados.
#
# Entradas:
#   - Caminhos dos arquivos de dados do PISA 2018.
# Saídas:
#   - DataFrames: (respostas, questionario)
# ============================================================

import os
import pandas as pd
from utils_log import log_mensagem


def coletar_dados(cenario: dict):
    etapa = "ETAPA 3 - Coleta de Dados (PISA 2018)"
    log_mensagem(etapa, "Iniciando leitura das planilhas...", "inicio")

    # ============================================================
    # 1) Busca automática pelos arquivos na pasta 'dados'
    # ------------------------------------------------------------
    pasta_dados = "dados"
    if not os.path.exists(pasta_dados):
        raise FileNotFoundError(f"[ERRO CRÍTICO] Pasta '{pasta_dados}' não encontrada. "
                                "Crie a pasta e coloque nela os arquivos de respostas e questionário.")

    arquivos = os.listdir(pasta_dados)
    arquivos_lower = [f.lower() for f in arquivos]

    # Detecta automaticamente os arquivos
    arquivo_respostas = next(
        (os.path.join(pasta_dados, f) for f in arquivos if "resposta" in f.lower()), None
    )
    arquivo_questionario = next(
        (os.path.join(pasta_dados, f) for f in arquivos if "question" in f.lower()), None
    )

    if not arquivo_respostas or not arquivo_questionario:
        raise FileNotFoundError(
            f"[ERRO CRÍTICO] Não foi possível localizar os arquivos esperados na pasta '{pasta_dados}'.\n"
            f"Arquivos encontrados: {arquivos}\n"
            f"Certifique-se de incluir algo como:\n"
            f"  - 'TCH_CH_Respostas.xlsx'\n"
            f"  - 'TCH_CHL_Questionario.xlsx'"
        )

    # ============================================================
    # 2) Carregamento robusto com detecção de formato
    # ------------------------------------------------------------
    def carregar_arquivo(caminho):
        extensao = os.path.splitext(caminho)[1].lower()
        if extensao == ".csv":
            try:
                try:
                    return pd.read_csv(caminho, sep=",", encoding="utf-8", engine="python")
                except Exception:
                    return pd.read_csv(caminho, sep="\t", encoding="utf-8", engine="python")
            except Exception as e:
                raise RuntimeError(f"[ERRO FATAL] Falha ao ler '{caminho}'. Detalhes: {e}")
        elif extensao in [".xlsx", ".xls"]:
            try:
                return pd.read_excel(caminho, sheet_name=0)
            except Exception as e:
                raise RuntimeError(f"[ERRO FATAL] Falha ao ler '{caminho}'. Detalhes: {e}")
        else:
            raise ValueError(f"[ERRO] Formato de arquivo não suportado: {caminho}")

    respostas = carregar_arquivo(arquivo_respostas)
    questionario = carregar_arquivo(arquivo_questionario)

    # ============================================================
    # 3) Padronização e limpeza inicial
    # ------------------------------------------------------------
    respostas.columns = [str(col).strip() for col in respostas.columns]
    questionario.columns = [str(col).strip() for col in questionario.columns]

    respostas = respostas.loc[:, ~respostas.columns.duplicated()]
    questionario = questionario.loc[:, ~questionario.columns.duplicated()]

    # ============================================================
    # 4) Verificação de integridade dos dados
    # ------------------------------------------------------------
    if respostas.empty or questionario.empty:
        raise ValueError("[ERRO CRÍTICO] Um dos arquivos está vazio após a leitura.")

    if respostas.shape[1] < 100:
        log_mensagem(etapa, "[ALERTA] A planilha de respostas contém poucas colunas. Verifique o arquivo fonte.", "aviso")

    # ============================================================
    # 5) Registro final
    # ------------------------------------------------------------
    log_mensagem(
        etapa,
        f"Leitura concluída: respostas={respostas.shape}, questionário={questionario.shape}",
        "fim"
    )

    return respostas, questionario
