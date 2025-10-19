# ============================================================
# ETAPA 11 - RELATÓRIO AUTOMATIZADO COM LLM (GROQ | GOOGLE)
# ------------------------------------------------------------
# Objetivo:
#   - Ler os resultados consolidados (CSV's e contexto) das etapas 5 a 10;
#   - Montar um prompt com visão executiva e técnica;
#   - Chamar um LLM (Groq OU Google) e salvar o relatório em texto/markdown.
#
# Entradas:
#   - Arquivos gerados nas etapas anteriores (ex.: resultados/tabelas/*.csv)
#   - Variáveis de ambiente no .env:
#       GROQ_API_KEY        -> para provider = "groq"
#       GOOGLE_API_KEY      -> para provider = "google"
#       LLM_MODEL (opcional)-> ex.: "llama-3.3-70b" ou "gemini-2.5-flash"
#
# Saídas:
#   - resultados/relatorio_llm.md  (relatório consolidado)
# ============================================================

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# Carrega .env (caso exista)
load_dotenv()

# ===== Imports opcionais conforme provider =====
# (Importamos dentro das funções para não criar dependência dura quando o provider não é usado.)

# ============================================================
# Utilidades internas
# ============================================================

def _ler_conteudos_para_prompt() -> str:
    """Carrega os principais artefatos em texto para embutir no prompt."""
    partes = []

    # (1) Variáveis significativas (Etapa 9)
    var_sig = Path("resultados/tabelas/variaveis_significativas.csv")
    if var_sig.exists():
        try:
            df = pd.read_csv(var_sig)
            partes.append("### Variáveis significativas (topo do CSV)\n")
            partes.append(df.head(30).to_string(index=False))
        except Exception as e:
            partes.append(f"(Falha ao ler {var_sig}: {e})")
    else:
        partes.append("(Arquivo de variáveis significativas não encontrado.)")

    # (2) Resultados OLS (Etapa 7)
    ols = Path("resultados/tabelas/modelo_ols_resultados.csv")
    if ols.exists():
        try:
            df = pd.read_csv(ols)
            partes.append("\n\n### Resultados OLS (amostra das 20 primeiras linhas)\n")
            partes.append(df.head(20).to_string(index=False))
        except Exception as e:
            partes.append(f"(Falha ao ler {ols}: {e})")
    else:
        partes.append("\n(Arquivo de resultados OLS não encontrado.)")

    # (3) Comparação de modelos (Etapa 7 – seleção automática)
    comp = Path("resultados/tabelas/comparacao_modelos.csv")
    meta = Path("resultados/tabelas/melhor_modelo.json")
    if comp.exists() and meta.exists():
        try:
            dfc = pd.read_csv(comp)
            meta_json = json.loads(meta.read_text(encoding="utf-8"))
            partes.append("\n\n### Comparação de Modelos (ranking)\n")
            # ordena para facilitar a leitura: R2_CV desc, RMSE asc, R2_aj desc, AIC asc
            dfc["__r2cv"] = dfc["R2_CV"].fillna(-1e9)
            dfc["__rmse"] = dfc["RMSE_CV"].fillna(1e9)
            dfc["__r2aj"] = dfc["R2_ajustado"].fillna(-1e9)
            dfc["__aic"]  = dfc["AIC"].fillna(1e9)
            dfc = dfc.sort_values(by=["__r2cv","__rmse","__r2aj","__aic"], ascending=[False,True,False,True])
            partes.append(dfc[["modelo","R2_CV","RMSE_CV","R2_ajustado","AIC","BIC","notas"]].head(6).to_string(index=False))

            partes.append("\n\n### Modelo selecionado automaticamente\n")
            partes.append(f"Melhor modelo: {meta_json.get('melhor_modelo')}")
            partes.append(f"Critério: {meta_json.get('criterio')}")
            partes.append(f"Alvo: {meta_json.get('alvo')}")
            partes.append(f"Total de variáveis: {len(meta_json.get('features', []))}")
            partes.append("\n(Justifique no texto: o modelo foi escolhido por apresentar melhor desempenho segundo o critério acima.)")
        except Exception as e:
            partes.append(f"(Falha ao ler comparação/metadata da Etapa 7: {e})")
    else:
        partes.append("\n(Arquivos de comparação de modelos não encontrados.)")

    return "\n".join(partes)


def _montar_prompt_base(cenario: dict | None = None) -> str:
    """Cria um prompt consistente para geração do relatório."""
    contexto = _ler_conteudos_para_prompt()
    cabecalho = (
        "Você é um analista de políticas educacionais. "
        "Gere um relatório executivo e técnico, claro e objetivo, "
        "sobre o bem-estar docente de Matemática no Chile, a partir dos resultados abaixo. "
        "Organize em: Introdução, Dados e Método, Principais Evidências, Implicações, "
        "Recomendações de Política Pública (de curto, médio e longo prazo), Limitações e Próximos Passos.\n"
    )
    if cenario:
        cabecalho += f"\nCenário: {cenario}\n"
    cabecalho += "\n=== Evidências e Resultados ===\n"
    return cabecalho + contexto


def _salvar_relatorio(texto: str, caminho: Path = Path("resultados/relatorio_llm.md")) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(texto, encoding="utf-8")
    return caminho

# ============================================================
# Provedor: GOOGLE (Gemini) – Caminho idêntico ao seu teste
# ============================================================

def _gerar_google(model_name: str | None, prompt: str) -> str:
    """
    Implementação com google-generativeai idêntica ao seu teste mínimo:
      - lê GOOGLE_API_KEY
      - genai.configure(api_key=...)
      - model = genai.GenerativeModel("gemini-2.5-flash")
      - response = model.generate_content(prompt)
      - return response.text
    """
    import google.generativeai as genai
    from dotenv import load_dotenv
    import os

    # 1) Carrega .env (já feito no topo, mas é idempotente)
    load_dotenv()

    # 2) Obtém a chave
    api_key = os.getenv("GOOGLE_API_KEY")

    # 3) Verificação crítica
    if not api_key:
        raise RuntimeError(
            "A variável de ambiente GOOGLE_API_KEY não foi encontrada. "
            "Verifique seu arquivo .env."
        )

    # 4) Configura o SDK
    genai.configure(api_key=api_key)

    # 5) Modelo
    # Se não foi passado por argumento/ambiente, usamos o mesmo do seu teste:
    model_name = model_name or os.getenv("LLM_MODEL") or "gemini-2.5-flash"

    # 6) Chamada
    model = genai.GenerativeModel(model_name)
    try:
        response = model.generate_content(prompt)
        # Em chamadas bem-sucedidas, o conteúdo vem em response.text
        texto = getattr(response, "text", None) or ""
        return texto.strip()
    except Exception as e:
        # Propaga a exceção com mensagem clara
        raise RuntimeError(f"Falha na chamada ao Google Generative AI: {e}")

# ============================================================
# Provedor: GROQ (Llama)
# ============================================================

def _gerar_groq(model_name: str | None, prompt: str) -> str:
    """
    Implementação com groq: usa GROQ_API_KEY e chat.completions.create.
    """
    from groq import Groq
    from dotenv import load_dotenv
    import os

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "A variável de ambiente GROQ_API_KEY não foi encontrada. "
            "Verifique seu arquivo .env."
        )

    client = Groq(api_key=api_key)
    model_name = model_name or os.getenv("LLM_MODEL") or "llama-3.3-70b-versatile"

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Você é um analista de políticas educacionais."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1800,
        )
        texto = resp.choices[0].message.content if resp and resp.choices else ""
        return (texto or "").strip()
    except Exception as e:
        raise RuntimeError(f"Falha na chamada ao Groq: {e}")

# ============================================================
# API PÚBLICA DA ETAPA 11
# ============================================================

def gerar_relatorio_automatico(provider: str | None = None, model: str | None = None, cenario: dict | None = None) -> str:
    """
    provider: "google" | "groq" | "auto" | None
      - None ou "auto": tenta GROQ, se falhar tenta GOOGLE (ou o inverso, se preferir).
    model: nome do modelo específico (opcional)
    """
    logging.info("ETAPA 11 - Relatório LLM - Preparando prompt e selecionando provedor...")

    prompt = _montar_prompt_base(cenario)

    # Respeita parâmetro, senão variável de ambiente, senão 'auto'
    provider = (provider or os.getenv("LLM_PROVIDER") or "auto").strip().lower()

    if provider == "google":
        logging.info("ETAPA 11 - Relatório LLM - Provedor selecionado: GOOGLE (Gemini).")
        texto = _gerar_google(model, prompt)

    elif provider == "groq":
        logging.info("ETAPA 11 - Relatório LLM - Provedor selecionado: GROQ (Llama).")
        texto = _gerar_groq(model, prompt)

    else:
        # Estratégia AUTO: tenta primeiro GROQ, se der erro, tenta GOOGLE
        logging.info("ETAPA 11 - Relatório LLM - Provedor 'auto': tentando GROQ, depois GOOGLE (fallback).")
        try:
            texto = _gerar_groq(model, prompt)
        except Exception as e_groq:
            logging.warning(f"ETAPA 11 - Groq falhou: {e_groq}. Tentando Google...")
            texto = _gerar_google(model, prompt)

    if not texto:
        logging.warning("ETAPA 11 - Relatório LLM - Texto vazio retornado. Verifique logs e chaves de API.")
    else:
        caminho = _salvar_relatorio(texto)
        logging.info(f"ETAPA 11 - Relatório LLM - Relatório salvo em '{caminho}'.")
    return texto
