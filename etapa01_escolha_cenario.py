# ============================================================
# ETAPA 01 - ESCOLHA DO CENÁRIO
# ------------------------------------------------------------
# Objetivo:
#   - Definir o contexto de análise (país, público, disciplina e tema)
#   - Retornar um dicionário estruturado com as informações do cenário
#   - Registrar no log o início e o fim da etapa
#
# Saída esperada:
#   {
#       "pais": "Chile",
#       "disciplina": "Matemática",
#       "publico": "Docentes",
#       "tema": "Bem-estar docente"
#   }
#
# ============================================================

from utils_log import log_mensagem


def escolher_cenario():
    etapa = "ETAPA 1 - Escolha do Cenário"
    log_mensagem(etapa, "Definindo país, público e disciplina...", "inicio")

    # ============================================================
    # 1) Definições do cenário analítico
    # ------------------------------------------------------------
    # Estas configurações podem futuramente ser parametrizadas por
    # arquivo JSON, interface Gradio/Streamlit ou input do usuário.
    # ============================================================
    pais = "Chile"
    disciplina = "Matemática"
    publico = "Docentes"
    tema = "Bem-estar docente"

    cenario = {
        "pais": pais,
        "disciplina": disciplina,
        "publico": publico,
        "tema": tema
    }

    # ============================================================
    # 2) Registro da definição no log e retorno
    # ============================================================
    log_mensagem(etapa, f"Cenário definido: {cenario}", "fim")
    return cenario
