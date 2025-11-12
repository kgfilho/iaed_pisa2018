# ============================================================
# ETAPA 11 – RELATÓRIO AUTOMÁTICO COM LLM (VERSÃO FINAL)
# ------------------------------------------------------------
# - (MODIFICADO) _selecionar_provedor_modelo: Corrige os nomes
#   padrão dos modelos para 'gemini-2.5-flash' (Google) e
#   'llama-3.3-70b-versatile' (Groq), conforme especificado.
# - (MODIFICADO) _executar_geracao: Também atualiza o nome
#   padrão do modelo Groq.
# ============================================================

import os
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Provedores de LLM
try:
    from groq import Groq
    GROQ_DISPONIVEL = True
except ImportError:
    GROQ_DISPONIVEL = False

try:
    import google.generativeai as genai
    GOOGLE_DISPONIVEL = True
except ImportError:
    GOOGLE_DISPONIVEL = False

# ============================================================
# FUNÇÕES AUXILIARES: COLETA E FORMATAÇÃO DE ARTEFATOS
# (Esta seção inteira não foi alterada)
# ============================================================

def _coletar_artefatos() -> dict:
    artefatos = {}
    base_path = Path("resultados")

    # 1. Metadados do Modelo
    try:
        with open(base_path / "tabelas/melhor_modelo.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
            artefatos['melhor_modelo_meta'] = meta
            artefatos['alvo'] = meta.get('alvo')
            artefatos['alvo_media'] = meta.get('alvo_media')
            artefatos['alvo_n_validos'] = meta.get('alvo_n_validos')
    except Exception as e:
        logging.warning(f"Artefato 'melhor_modelo.json' não encontrado: {e}")
        artefatos['melhor_modelo_meta'] = {"erro": str(e)}

    # 2. Comparação de Todos os Modelos
    try:
        artefatos['comparacao_modelos_csv'] = pd.read_csv(
            base_path / "tabelas/comparacao_modelos.csv"
        ).to_string(index=False)
    except Exception as e:
        logging.warning(f"Artefato 'comparacao_modelos.csv' não encontrado: {e}")

    # 3. Variáveis Relevantes
    try:
        p_rf = base_path / "tabelas/variaveis_importancia_rf.csv"
        p_ols = base_path / "tabelas/variaveis_significativas_ols.csv"
        if p_rf.exists():
            artefatos['variaveis_relevantes_csv'] = pd.read_csv(p_rf).to_string(index=False)
        elif p_ols.exists():
            artefatos['variaveis_relevantes_csv'] = pd.read_csv(p_ols).to_string(index=False)
        else:
            raise FileNotFoundError("Nenhum CSV de variáveis relevantes/importantes encontrado.")
    except Exception as e:
        logging.warning(f"Artefato de variáveis relevantes não encontrado: {e}")

    # 4. Recomendações brutas
    try:
        artefatos['recomendacoes_txt'] = (
            base_path / "textos/recomendacoes_politicas_publicas.txt"
        ).read_text(encoding="utf-8")
    except Exception as e:
        logging.warning(f"Artefato 'recomendacoes_politicas_publicas.txt' não encontrado: {e}")

    # 5. Matriz de Correlação
    try:
        df_corr = pd.read_csv(base_path / "tabelas/correlacoes.csv")
        if len(df_corr) > 15:
            principais = [artefatos['alvo']] + artefatos['melhor_modelo_meta'].get('features', [])
            principais = [c for c in principais if c in df_corr.columns][:15]
            df_corr = df_corr.set_index('variavel').loc[principais, principais].reset_index()
        artefatos['correlacoes_csv'] = df_corr.to_string(index=False)
    except Exception as e:
        logging.warning(f"Artefato 'correlacoes.csv' não encontrado ou falha ao filtrar: {e}")

    # 6. Composição dos Índices
    try:
        caminho_composicao = base_path / "tabelas/composicao_indices.json"
        with open(caminho_composicao, "r", encoding="utf-8") as f:
            artefatos['composicao_indices'] = json.load(f)
    except Exception as e:
        logging.warning(f"Artefato 'composicao_indices.json' não encontrado: {e}")

    return artefatos

def _gerar_prompt_sistema() -> str:
    """Define a persona e a missão do LLM."""
    return """
Você é um Analista de Dados Educacionais Sênior, especializado em interpretar
modelos estatísticos (como OLS e Random Forest) e dados da pesquisa PISA.

Sua missão é gerar um relatório executivo completo em português do Brasil,
baseado nos artefatos quantitativos fornecidos (JSONs, CSVs).

O relatório deve:
1.  Ser estruturado em Markdown (usando #, ##, ###, *).
2.  Ser técnico, mas acessível a gestores de políticas públicas.
3.  Explicar o objetivo do estudo, a metodologia (modelo vencedor) e os
    resultados (R² e variáveis mais importantes).
4.  Interpretar o que as variáveis mais importantes significam na prática.
5.  Refinar as recomendações preliminares, dando-lhes mais contexto.
6.  Não inventar informações. Baseie-se estritamente nos dados fornecidos.

ESTRUTURA OBRIGATÓRIA DO RELATÓRIO:
# Relatório Executivo: Análise de Bem-Estar Docente no Chile (PISA 2018)

## 1. Resumo Executivo (Insights Principais)
(Seja breve e direto. Quais foram as 2-3 descobertas mais importantes?)

## 2. Contexto e Objetivo do Estudo
(Qual era o alvo da análise e o que buscávamos entender?)

## 3. Metodologia: Seleção do Modelo Preditivo
(Qual modelo venceu (OLS, RF)? Por quê (R²)? Explique o R² encontrado.)
(Mencione a média da variável alvo se ela for extrema (ex: 0.98), pois
isso explica a dificuldade em obter um R² alto.)

## 4. Principais Fatores Preditivos (Feature Importance)
(Quais foram as variáveis mais importantes que o modelo encontrou?
Liste-as em ordem e explique o que elas significam.)

## 5. Análise Detalhada dos Índices Utilizados
(Explique quais perguntas do PISA foram usadas para construir os
índices-chave, como 'clima_media' ou 'carga_trabalho_media'.)

## 6. Recomendações para Políticas Públicas
(Refine as recomendações preliminares com base nos seus insights.)

## 7. Limitações e Próximos Passos
(Quais foram as limitações (ex: R² moderado, dados faltantes)?)
"""

def _gerar_prompt_usuario(artefatos: dict) -> str:
    """Formata todos os artefatos em um único prompt de usuário."""
    
    prompt_parts = [
        "Por favor, gere o relatório executivo com base nos seguintes artefatos:",
        "\n--- INÍCIO DOS ARTEFATOS ---\n"
    ]

    # 1. Contexto (do JSON)
    meta = artefatos.get('melhor_modelo_meta', {})
    contexto_original = meta.get('contexto', {})
    if contexto_original:
        prompt_parts.append("== CONTEXTO DO ESTUDO ==\n" + json.dumps(contexto_original, indent=2, ensure_ascii=False))

    # 2. Desempenho do Modelo Vencedor (do JSON)
    prompt_parts.append("\n== DESEMPENHO DO MODELO VENCEDOR ==\n" +
                        f"Modelo Vencedor: {meta.get('melhor_modelo')}\n" +
                        f"Variável Alvo (Y): {meta.get('alvo')}\n" +
                        f"Preditores (X) Utilizados: {meta.get('features')}\n"
    )
    if artefatos.get('alvo_media'):
        prompt_parts.append(
            f"Média da Variável Alvo ({meta.get('alvo')}): {artefatos['alvo_media']:.4f}\n" +
            f"(N Válido: {artefatos.get('alvo_n_validos')})\n"
        )

    # 3. Comparação de Todos os Modelos (CSV)
    if artefatos.get('comparacao_modelos_csv'):
        prompt_parts.append("\n== DESEMPENHO DE TODOS OS MODELOS (R² E MÉTRICAS) ==\n" +
                            artefatos['comparacao_modelos_csv'])

    # 4. Variáveis Relevantes (CSV da Etapa 9)
    if artefatos.get('variaveis_relevantes_csv'):
        prompt_parts.append("\n== PRINCIPAIS VARIÁVEIS (POR IMPORTÂNCIA OU P-VALOR) ==\n" +
                            artefatos['variaveis_relevantes_csv'])
    
    # 5. Composição dos Índices (JSON da Etapa 5)
    if artefatos.get('composicao_indices'):
        prompt_parts.append("\n== COMPOSIÇÃO DOS ÍNDICES (A 'RECEITA') ==\n" +
                            "(Mapeamento de 'Índice Gerado' -> ['Colunas PISA Originais'])\n" +
                            json.dumps(artefatos['composicao_indices'], indent=2, ensure_ascii=False))

    # 6. Recomendações Preliminares (TXT da Etapa 10)
    if artefatos.get('recomendacoes_txt'):
        prompt_parts.append("\n== RECOMENDAÇÕES PRELIMINARES (DA ETAPA 10) ==\n" +
                            artefatos['recomendacoes_txt'])

    # 7. Correlações (CSV)
    if artefatos.get('correlacoes_csv'):
        prompt_parts.append("\n== MATRIZ DE CORRELAÇÃO (AMOSTRA) ==\n" +
                            artefatos['correlacoes_csv'])

    prompt_parts.append("\n--- FIM DOS ARTEFATOS ---")
    
    return "\n".join(prompt_parts)


# ============================================================
# FUNÇÕES DE EXECUÇÃO: LLM
# ============================================================

def _selecionar_provedor_modelo(provider: str | None, model: str | None) -> tuple:
    """
    Seleciona o cliente da API (Groq ou Google) e o nome do modelo.
    (Nomes padrão corrigidos para 'gemini-2.5-flash' e 'llama-3.3-70b-versatile')
    """
    
    groq_api_key = os.environ.get("GROQ_API_KEY")
    google_api_key = os.environ.get("GOOGLE_API_KEY")

    # Configuração do Google
    if google_api_key and (provider == "google" or (provider in [None, "auto"] and not groq_api_key)):
        try:
            genai.configure(api_key=google_api_key)
            
            # ########################################################
            # # ### INÍCIO DA MODIFICAÇÃO (Nome do Modelo) ###
            # ########################################################
            model_name = model if model else "gemini-2.5-flash" # <-- CORRIGIDO
            # ########################################################
            # # ### FIM DA MODIFICAÇÃO ###
            # ########################################################
            
            cliente = genai.GenerativeModel(model_name)
            logging.info(f"Usando Provedor: Google (Modelo: {model_name})")
            return cliente, "google"
        except Exception as e:
            logging.error(f"Falha ao configurar Google Gemini: {e}")
            if not groq_api_key:
                 raise ConnectionError("Falha no Google e GROQ_API_KEY não encontrada.")

    # Configuração do Groq
    if groq_api_key:
        try:
            cliente = Groq(api_key=groq_api_key)
            
            # ########################################################
            # # ### INÍCIO DA MODIFICAÇÃO (Nome do Modelo) ###
            # ########################################################
            model_name = model if model else "llama-3.3-70b-versatile" # <-- CORRIGIDO
            # ########################################################
            # # ### FIM DA MODIFICAÇÃO ###
            # ########################################################

            logging.info(f"Usando Provedor: Groq (Modelo: {model_name})")
            return cliente, "groq"
        except Exception as e:
            logging.error(f"Falha ao configurar Groq: {e}")
            raise ConnectionError("Falha ao inicializar a API do Groq.")

    raise ConnectionError("Nenhuma chave de API (GROQ_API_KEY ou GOOGLE_API_KEY) foi encontrada nas variáveis de ambiente.")


def _executar_geracao(cliente, tipo_provedor: str, system_prompt: str, user_prompt: str, model_name: str | None) -> str:
    """
    Executa a chamada à API (Groq ou Google) e retorna a resposta em texto.
    (Usa a sintaxe antiga do Google para compatibilidade)
    """
    try:
        if tipo_provedor == "groq":
            # ########################################################
            # # ### INÍCIO DA MODIFICAÇÃO (Nome do Modelo) ###
            # ########################################################
            # Garante que o modelo padrão aqui seja o mesmo definido acima
            model_name = model_name if model_name else "llama-3.3-70b-versatile" # <-- CORRIGIDO
            # ########################################################
            # # ### FIM DA MODIFICAÇÃO ###
            # ########################################################
            
            chat_completion = cliente.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model_name,
                temperature=0.3,
                max_tokens=4096, # Llama 3.3 tem 8k, mas 4k é seguro para a resposta
            )
            return chat_completion.choices[0].message.content
        
        elif tipo_provedor == "google":
            # Usando a sintaxe de compatibilidade (generate_content)
            # que funcionou na correção anterior.
            logging.info("Usando sintaxe de compatibilidade do Google (generate_content).")
            prompt_combinado = [
                system_prompt,
                "---", # Separador
                user_prompt
            ]
            response = cliente.generate_content(prompt_combinado)
            return response.text
        
        else:
            raise ValueError(f"Tipo de provedor desconhecido: {tipo_provedor}")

    except Exception as e:
        logging.error(f"Erro durante a chamada da API do LLM ({tipo_provedor}): {e}", exc_info=True)
        if hasattr(e, 'response'):
            try:
                erro_api = e.response.json()
                logging.error(f"Detalhes do erro da API: {erro_api}")
                return f"Erro ao gerar relatório: {erro_api}"
            except Exception:
                return f"Erro ao gerar relatório: {str(e)}"
        return f"Erro ao gerar relatório: {str(e)}"


# ============================================================
# FUNÇÃO PRINCIPAL: PONTO DE ENTRADA
# (Esta seção inteira não foi alterada)
# ============================================================

def gerar_relatorio_automatico(provider: str | None = None, model: str | None = None):
    etapa = "ETAPA 11 - Relatório Automático (LLM)"
    logging.info(f"({etapa}) - Iniciando geração de relatório...")
    
    base_path = Path("resultados/textos_llm")
    base_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Coleta de dados
        logging.info(f"({etapa}) - Coletando artefatos das Etapas 1-10...")
        artefatos = _coletar_artefatos()
        if not artefatos.get('melhor_modelo_meta'):
            raise FileNotFoundError("Artefatos essenciais (melhor_modelo.json) não encontrados.")

        # 2. Seleção do Provedor
        logging.info(f"({etapa}) - Selecionando provedor de LLM...")
        cliente, tipo_provedor = _selecionar_provedor_modelo(provider, model)
        
        # 3. Geração dos Prompts
        logging.info(f"({etapa}) - Gerando prompts (sistema e usuário)...")
        system_prompt = _gerar_prompt_sistema()
        user_prompt = _gerar_prompt_usuario(artefatos)
        
        (base_path / f"prompt_usuario_{datetime.now():%Y%m%d_%H%M%S}.txt").write_text(
            user_prompt, encoding="utf-8"
        )

        # 4. Execução
        logging.info(f"({etapa}) - Executando chamada à API {tipo_provedor.upper()}... (Isso pode levar um momento)")
        relatorio_texto = _executar_geracao(
            cliente,
            tipo_provedor,
            system_prompt,
            user_prompt,
            model_name=model
        )

        # 5. Salvamento
        if not relatorio_texto or len(relatorio_texto) < 100:
             logging.warning(f"({etapa}) - Resposta do LLM foi curta ou vazia: {relatorio_texto}")
             raise ValueError("A resposta do LLM foi muito curta ou vazia.")
             
        caminho_relatorio = base_path / "relatorio_final_llm.md"
        caminho_relatorio.write_text(relatorio_texto, encoding="utf-8")
        
        logging.info(f"({etapa}) - Relatório final salvo com sucesso em '{caminho_relatorio}'")
        return relatorio_texto

    except Exception as e:
        logging.error(f"[ERRO FATAL] Falha na Etapa 11: {e}", exc_info=True)
        (base_path / "relatorio_ERRO.txt").write_text(f"Falha ao gerar o relatório: {e}", encoding="utf-8")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] (%(levelname)s) %(message)s")
    print("Executando Etapa 11 em modo de teste...")
    if not Path("resultados/tabelas/melhor_modelo.json").exists():
        print("\n[ERRO DE TESTE]")
        print("Os artefatos das Etapas 1-10 não existem.")
        print("Execute 'python main.py --no-llm' primeiro para gerar os resultados.")
    else:
        gerar_relatorio_automatico()