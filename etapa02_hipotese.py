# ============================================================
# ETAPA 02 - FORMULAÇÃO DA HIPÓTESE
# ------------------------------------------------------------
# Objetivo:
#   - Formular a hipótese de pesquisa e estabelecer as variáveis
#     dependente e explicativas (independentes).
#   - Conectar a etapa de definição de cenário (Etapa 01) com
#     a coleta e análise de dados (Etapas 03–07).
#   - Registrar a execução no log.
#
# Saída esperada:
#   {
#       "descricao": "O bem-estar docente influencia a motivação...",
#       "variavel_dependente": "...",
#       "variaveis_explicativas": [...],
#       "contexto": {...}
#   }
# ============================================================

from utils_log import log_mensagem


def formular_hipotese(cenario: dict):
    etapa = "ETAPA 2 - Formulação da Hipótese"
    log_mensagem(etapa, "Formulando hipótese e variáveis...", "inicio")

    # ============================================================
    # 1) Contexto recebido da Etapa 1
    # ------------------------------------------------------------
    # O dicionário 'cenario' deve conter:
    # {"pais": ..., "disciplina": ..., "publico": ..., "tema": ...}
    # ============================================================
    pais = cenario.get("pais", "Desconhecido")
    disciplina = cenario.get("disciplina", "Desconhecida")
    publico = cenario.get("publico", "Desconhecido")
    tema = cenario.get("tema", "Não definido")

    # ============================================================
    # 2) Definição da hipótese central
    # ------------------------------------------------------------
    # A hipótese pode ser adaptada conforme o país e o tema.
    # Exemplo: “O bem-estar docente influencia a motivação
    # dos alunos e o desempenho em Matemática.”
    # ============================================================
    hipotese_texto = (
        f"O nível de {tema.lower()} dos professores de {disciplina.lower()} "
        f"no {pais} está associado positivamente à sua formação profissional "
        f"e ao desenvolvimento contínuo de competências pedagógicas."
    )

    # ============================================================
    # 3) Variáveis envolvidas
    # ------------------------------------------------------------
    # Dependente: Indicador de bem-estar docente (exemplo: motivação)
    # Explicativas: Formação inicial, formação continuada e práticas pedagógicas.
    # ============================================================
    variavel_dependente = (
        "TC199Q05HA: In your teaching, to what extent can you do: "
        "Motivate students who show low interest in school work"
    )

    variaveis_explicativas = [
        "TC014Q01HA: Did you complete a teacher education or training programme?",
        "TC018Q02NA: Included in teacher education, training or other qualification: Mathematics",
        "TC045Q01NB: Included in professional development during last 12 months: "
        "Knowledge and understanding of my subject field(s)",
        "TC045Q13NB: Included in professional development during last 12 months: "
        "Internal evaluation or self-evaluation of schools",
        "TC045Q16HB: Included in professional development during last 12 months: "
        "Second language teaching"
    ]

    # ============================================================
    # 4) Montagem do objeto estruturado
    # ------------------------------------------------------------
    # Inclui 'descricao' para compatibilidade com o main.py
    # ============================================================
    definicao = {
        "descricao": hipotese_texto,
        "hipotese": hipotese_texto,
        "variavel_dependente": variavel_dependente,
        "variaveis_explicativas": variaveis_explicativas,
        "contexto": {
            "pais": pais,
            "disciplina": disciplina,
            "publico": publico,
            "tema": tema
        }
    }

    # ============================================================
    # 5) Registro final da etapa
    # ============================================================
    log_mensagem(etapa, "Hipótese e variáveis definidas.", "fim")

    return definicao
