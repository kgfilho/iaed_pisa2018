# ==========================================================
# ETAPA PRINCIPAL – PIPELINE GERAL DE ANÁLISE (PISA 2018)
# ==========================================================
# (MODIFICADO) Adiciona 'dotenv' para carregar as chaves de API
# do arquivo .env antes que a Etapa 11 seja importada.
# ==========================================================

import os
from dotenv import load_dotenv # Carrega o .env
load_dotenv() # Executa o carregamento

import argparse
import logging
from datetime import datetime

# ===== Importação das Etapas =====
from etapa01_escolha_cenario import escolher_cenario
from etapa02_hipotese import formular_hipotese
from etapa03_coleta_dados import coletar_dados
from etapa04_preprocessamento import preprocessar_dados
from etapa05_transformacao import transformar_dados
from etapa06_mineracao_dados import minerar_dados
from etapa07_descoberta_modelos import ajustar_modelo
from etapa08_interpretacao import gerar_graficos
from etapa09_refinamento import refinar_conhecimento
from etapa10_recomendacoes import gerar_recomendacoes

# Etapa 11 é opcional: só importamos se existir/configurada corretamente
try:
    from etapa11_relatorio_llm import gerar_relatorio_automatico
    ETAPA11_DISPONIVEL = True
except Exception:
    ETAPA11_DISPONIVEL = False

# ===== Configuração de Logs =====
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] (%(message)s)",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ==========================================================
# PARSERS E OPÇÕES DE EXECUÇÃO
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline KDD – PISA 2018 (Bem-Estar Docente)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Pula a Etapa 11 (Geração de relatório via LLM)."
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        choices=["groq", "google", "auto"],
        help="Seleciona o provedor do LLM ('groq' | 'google' | 'auto')."
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Sobrescreve o modelo do LLM para esta execução (ex.: 'llama-3.3-70b' ou 'gemini-2.5-flash')."
    )
    return parser.parse_args()

# ==========================================================
# FUNÇÃO PRINCIPAL – EXECUÇÃO DO PIPELINE COMPLETO
# ==========================================================
def main():
    args = parse_args()
    inicio = datetime.now()
    logging.info("PIPELINE GERAL - Iniciando execução completa...)")

    try:
        # ==================================================
        # ETAPA 1 – ESCOLHA DO CENÁRIO
        # ==================================================
        cenario = escolher_cenario()
        logging.info(f"ETAPA 1 - Escolha do Cenário - Cenário definido: {cenario}")
        logging.info(f"PIPELINE GERAL - Cenário selecionado: {cenario})")

        # ==================================================
        # ETAPA 2 – FORMULAÇÃO DA HIPÓTESE
        # ==================================================
        hipotese = formular_hipotese(cenario)
        logging.info("ETAPA 2 - Formulação da Hipótese - Hipótese e variáveis definidas.")
        logging.info(f"PIPELINE GERAL - Hipótese definida: {hipotese['descricao']})")

        # ==================================================
        # ETAPA 3 – COLETA DE DADOS (PISA 2018)
        # ==================================================
        respostas, questionario = coletar_dados(cenario) # <--- Variável é 'questionario'
        
        # ########################################################
        # # ### INÍCIO DA CORREÇÃO ###
        # ########################################################
        # Corrigido: 'questionário' (com acento) -> 'questionario' (sem acento)
        logging.info(f"PIPELINE GERAL - Coleta concluída: respostas={respostas.shape}, questionario={questionario.shape})")
        # ########################################################
        # # ### FIM DA CORREÇÃO ###
        # ########################################################


        # ==================================================
        # ETAPA 4 – PRÉ-PROCESSAMENTO (LIMPEZA E TRATAMENTO)
        # ==================================================
        respostas = preprocessar_dados(respostas, cenario)
        logging.info(f"PIPELINE GERAL - Pré-processamento concluído: {len(respostas)} registros válidos.)")

        # ==================================================
        # ETAPA 5 – TRANSFORMAÇÃO DE DADOS
        # ==================================================
        respostas = transformar_dados(respostas, cenario)
        logging.info("PIPELINE GERAL - Transformação concluída: dados prontos para mineração.)")

        # ==================================================
        # ETAPA 6 – MINERAÇÃO DE DADOS (PCA + K-MEANS)
        # ==================================================
        respostas, modelo_kmeans = minerar_dados(respostas)
        logging.info(f"PIPELINE GERAL - Mineração concluída: {len(respostas)} docentes agrupados.)")

        # ==================================================
        # ETAPA 7 – DESCOBERTA DE PADRÕES E MODELAGEM (OLS)
        # ==================================================
        modelo_ols = ajustar_modelo(respostas)
        logging.info("PIPELINE GERAL - Modelo OLS ajustado com sucesso.)")

        # ==================================================
        # ETAPA 8 – INTERPRETAÇÃO E VISUALIZAÇÃO
        # ==================================================
        gerar_graficos(respostas)
        logging.info("PIPELINE GERAL - Visualizações geradas e salvas.)")

        # ==================================================
        # ETAPA 9 – REFINAMENTO DO CONHECIMENTO
        # ==================================================
        variaveis_significativas = refinar_conhecimento(modelo_ols, respostas)
        logging.info(f"PIPELINE GERAL - Refinamento concluído: {len(variaveis_significativas)} variáveis significativas.)")

        # ==================================================
        # ETAPA 10 – GERAÇÃO DE RECOMENDAÇÕES (POLÍTICAS PÚBLICAS)
        # ==================================================
        gerar_recomendacoes(cenario, modelo_ols, respostas)
        logging.info("PIPELINE GERAL - Recomendações de políticas públicas geradas com sucesso.)")

        # ==================================================
        # ETAPA 11 – RELATÓRIO AUTOMÁTICO COM LLM (OPCIONAL)
        # ==================================================
        if args.no_llm:
            logging.info("PIPELINE GERAL - Etapa 11 (LLM) ignorada por parâmetro --no-llm.)")
        else:
            if not ETAPA11_DISPONIVEL:
                logging.warning("PIPELINE GERAL - Etapa 11 não disponível (módulo ausente ou não configurado).")
            else:
                # Se o usuário passou --llm-model, sobrescreve o modelo apenas para esta execução
                if args.llm_model:
                    os.environ["LLM_MODEL"] = args.llm_model

                try:
                    relatorio_texto = gerar_relatorio_automatico(
                        provider=args.llm_provider,
                        model=args.llm_model
                    )
                    if isinstance(relatorio_texto, str) and len(relatorio_texto.strip()) > 0:
                        logging.info("PIPELINE GERAL - Relatório interpretativo gerado via LLM (Groq ou Google).")
                    else:
                        logging.warning("PIPELINE GERAL - Etapa 11 executada, mas o texto retornado está vazio.")
                except Exception as e_llm:
                    logging.warning(f"PIPELINE GERAL - Etapa 11 não executada: {str(e_llm)}")

        # ==================================================
        # FINALIZAÇÃO DO PIPELINE
        # ==================================================
        fim = datetime.now()
        duracao = (fim - inicio).total_seconds()
        logging.info(f"PIPELINE GERAL - Execução finalizada com sucesso em {duracao:.2f} segundos.)")

    except Exception as e:
        logging.error(f"[ERRO FATAL] Falha na execução do pipeline: {str(e)}", exc_info=True)


# ==========================================================
# EXECUÇÃO DIRETA
# ==========================================================
if __name__ == "__main__":
    main()