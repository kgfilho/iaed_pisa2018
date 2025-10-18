import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Tenta carregar as variáveis de ambiente do arquivo .env
load_dotenv()

# 2. Obtém a chave
api_key = os.getenv("GOOGLE_API_KEY")

# 3. VERIFICAÇÃO CRÍTICA: Confirma se a chave foi carregada
if not api_key:
    # Se a chave não for encontrada, imprime um erro claro e para.
    print("ERRO: A variável de ambiente GOOGLE_API_KEY não foi encontrada.")
    print("Verifique se o arquivo .env existe e contém a chave.")
else:
    # 4. Configura o SDK
    genai.configure(api_key=api_key)

    # 5. Executa a chamada
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content("Explique o que é o PISA 2018.")
        print(response.text)
    except Exception as e:
        # Captura e imprime quaisquer erros que ocorram durante a chamada da API (ex: Auth Error)
        print(f"\nOcorreu um erro durante a chamada da API:")
        print(e)