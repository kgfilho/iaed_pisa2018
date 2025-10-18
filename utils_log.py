# utils_log.py
from datetime import datetime
import os

def log_mensagem(etapa, mensagem, tipo="info"):
    """
    Exibe e registra mensagens padronizadas de execução do pipeline.
    tipo: 'info', 'inicio', 'fim', 'erro'
    """
    cores = {
        "inicio": "\033[95m",   # roxo
        "fim": "\033[92m",      # verde
        "info": "\033[94m",     # azul
        "erro": "\033[91m",     # vermelho
    }
    cor = cores.get(tipo, "\033[0m")
    reset = "\033[0m"
    hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    texto = f"[{hora}] ({etapa}) - {mensagem}"
    print(cor + texto + reset)

    # grava em arquivo
    os.makedirs("logs", exist_ok=True)
    with open("logs/execucao.log", "a", encoding="utf-8") as f:
        f.write(texto + "\n")
