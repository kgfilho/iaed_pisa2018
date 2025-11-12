# ============================================================
# ETAPA 3 – COLETA DE DADOS (PISA 2018)
# ------------------------------------------------------------
# Objetivo:
#   - Localizar e carregar automaticamente os arquivos de respostas e questionário.
#   - (CORRIGIDO) Ser mais específico na detecção do arquivo de
#     respostas para evitar carregar 'fields.csv' ou 'lbl.csv'.
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

    # ####################################################################
    # ### INÍCIO DA CORREÇÃO ###
    # ####################################################################
    # Modificamos a lógica para ser mais específica.
    
    # 1.1) Detecta o arquivo de Respostas
    #      Procura um arquivo que tenha "resposta" E "data" no nome,
    #      mas que NÃO tenha "lbl" ou "fields".
    arquivo_respostas = next(
        (os.path.join(pasta_dados, f) for f in arquivos 
         if "resposta" in f.lower() 
         and "data" in f.lower() 
         and "lbl" not in f.lower() 
         and "fields" not in f.lower()
        ), 
        None
    )
    
    # Se falhar, tenta uma lógica mais simples (pode ser arriscado)
    if not arquivo_respostas:
        arquivo_respostas = next(
            (os.path.join(pasta_dados, f) for f in arquivos if "resposta" in f.lower()), None
        )

    # 1.2) Detecta o arquivo de Questionário
    arquivo_questionario = next(
        (os.path.join(pasta_dados, f) for f in arquivos if "question" in f.lower()), None
    )

    if not arquivo_respostas or not arquivo_questionario:
        log_mensagem(etapa, f"Arquivos encontrados na pasta 'dados': {arquivos}", "erro")
        raise FileNotFoundError(
            f"[ERRO CRÍTICO] Não foi possível localizar os arquivos esperados na pasta '{pasta_dados}'.\n"
            f"Certifique-se de que os arquivos de *dados* (não apenas 'fields') e *questionário* estão presentes."
        )
    
    log_mensagem(etapa, f"Arquivo de respostas selecionado: {arquivo_respostas}", "info")
    log_mensagem(etapa, f"Arquivo de questionário selecionado: {arquivo_questionario}", "info")

    # ####################################################################
    # ### FIM DA CORREÇÃO ###
    # ####################################################################

    # ============================================================
    # 2) Carregamento robusto com detecção de formato
    # ------------------------------------------------------------
    def carregar_arquivo(caminho):
        extensao = os.path.splitext(caminho)[1].lower()
        if extensao == ".csv":
            try:
                # Tenta UTF-8 e vírgula
                return pd.read_csv(caminho, sep=",", encoding="utf-8", engine="python", low_memory=False)
            except Exception:
                try:
                    # Tenta Ponto e Vírgula
                    return pd.read_csv(caminho, sep=";", encoding="utf-8", engine="python", low_memory=False)
                except Exception as e:
                    raise RuntimeError(f"[ERRO FATAL] Falha ao ler '{caminho}'. Verifique o separador (',' ou ';'). Detalhes: {e}")
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

    # Verifica se as colunas PISA (códigos) parecem estar presentes
    colunas_exemplo = [col for col in respostas.columns if str(col).startswith("TC")]
    if not colunas_exemplo:
        log_mensagem(etapa, f"Colunas carregadas: {list(respostas.columns[:10])}...", "aviso")
        log_mensagem(etapa, "[ALERTA] Nenhuma coluna iniciada com 'TC' foi encontrada. O arquivo de respostas pode estar incorreto.", "aviso")
    else:
        log_mensagem(etapa, f"Encontradas {len(colunas_exemplo)} colunas PISA (ex: {colunas_exemplo[0]}).", "info")


    # ============================================================
    # 5) Registro final
    # ------------------------------------------------------------
    log_mensagem(
        etapa,
        f"Leitura concluída: respostas={respostas.shape}, questionário={questionario.shape}",
        "fim"
    )

    return respostas, questionario