# 🧩 PISA 2018 – Bem-Estar Docente no Chile  
### *Pipeline de Descoberta de Conhecimento com Inteligência Artificial e LLMs*

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)
![License](https://img.shields.io/badge/Licença-Acadêmica-green)
![Dataset](https://img.shields.io/badge/Dataset-PISA%202018-orange)

---

## 📘 Descrição Geral

Este projeto implementa um **pipeline completo de análise de dados educacionais** com base no **PISA 2018**, concentrando-se no tema do **bem-estar docente** no **Chile**, especialmente entre **professores de Matemática**.

O sistema segue a metodologia **KDD (Knowledge Discovery in Databases)** e combina estatística, aprendizado de máquina e **modelos de linguagem (LLMs)** para transformar dados quantitativos em **recomendações qualitativas de políticas públicas**.

---

## 🧠 Arquitetura do Pipeline

O pipeline executa **11 etapas**, automatizadas no arquivo `main.py`, que estruturam todo o processo de descoberta de conhecimento:

| Etapa | Nome | Descrição resumida |
|-------|------|--------------------|
| 1 | **Escolha do Cenário** | Define país, público e disciplina de análise. |
| 2 | **Formulação da Hipótese** | Gera a hipótese central e variáveis envolvidas. |
| 3 | **Coleta de Dados** | Lê os microdados do PISA 2018 e o questionário docente. |
| 4 | **Pré-processamento** | Realiza limpeza, normalização e filtragem dos dados. |
| 5 | **Transformação** | Cria índices derivados e métricas compostas. |
| 6 | **Mineração de Dados** | Executa *PCA* e *K-Means* para agrupamento e segmentação. |
| 7 | **Descoberta de Padrões** | Ajusta modelos estatísticos (ex.: Regressão OLS). |
| 8 | **Interpretação e Visualização** | Gera gráficos e relatórios exploratórios. |
| 9 | **Refinamento do Conhecimento** | Identifica variáveis estatisticamente significativas. |
| 10 | **Geração de Recomendações** | Traduz resultados em diretrizes de políticas públicas. |
| 11 | **Relatório Automatizado (LLM)** | Gera relatório interpretativo via GROQ (LLaMA) ou Google Gemini. |

---

## ⚙️ Tecnologias Principais

| Categoria | Ferramenta / Biblioteca |
|------------|------------------------|
| Análise de Dados | **Pandas**, **NumPy** |
| Modelagem Estatística | **Statsmodels**, **Scikit-learn** |
| Visualização | **Matplotlib**, **Seaborn** |
| Integração com LLM | **GROQ SDK**, **google-generativeai** |
| Configuração | **dotenv** para leitura segura de chaves |
| Persistência de Resultados | **CSV** e **Markdown** (em `/resultados/`) |

---

## 📂 Estrutura do Projeto

```
├── main.py
├── .env
├── etapa01_escolha_cenario.py
├── etapa02_hipotese.py
├── etapa03_coleta_dados.py
├── etapa04_preprocessamento.py
├── etapa05_transformacao.py
├── etapa06_mineracao_dados.py
├── etapa07_descoberta_modelos.py
├── etapa08_interpretacao.py
├── etapa09_refinamento.py
├── etapa10_recomendacoes.py
├── etapa11_relatorio_llm.py
├── resultados/
│   ├── tabelas/
│   │   ├── variaveis_significativas.csv
│   │   ├── modelo_ols_resultados.csv
│   │   ├── recomendacoes_politicas.txt
│   └── graficos/
│       ├── correlacao.png
│       ├── clusters.png
│       ├── importancia_variaveis.png
```

---

## 🔑 Configuração do Ambiente

### 1️⃣ Instalação dos pacotes
```bash
pip install -r requirements.txt
```

### 2️⃣ Criação do arquivo `.env`
O arquivo `.env` deve conter suas chaves de API:

```bash
# Para uso do Gemini
GOOGLE_API_KEY=sua_chave_aqui

# Para uso do GROQ
GROQ_API_KEY=sua_chave_aqui
```

---

## 🚀 Execução

### 🧩 Execução completa do pipeline
```bash
python main.py
```

### 🤖 Execução com LLM (Google Gemini)
```bash
python main.py --llm-provider google --llm-model gemini-2.5-flash
```

### 🦙 Execução com LLM (GROQ / LLaMA)
```bash
python main.py --llm-provider groq --llm-model llama-3.3-70b
```

### 💤 Executar sem o LLM (somente até Etapa 10)
```bash
python main.py --no-llm
```

---

## 📊 Resultados e Saídas

As principais saídas são geradas dentro do diretório `resultados/`:

| Tipo | Arquivo | Conteúdo |
|------|----------|----------|
| 📈 Análises estatísticas | `modelo_ols_resultados.csv` | Resultados da regressão OLS. |
| 📋 Variáveis significativas | `variaveis_significativas.csv` | Colunas com p-valor significativo. |
| 💡 Recomendações | `recomendacoes_politicas.txt` | Políticas públicas derivadas dos achados. |
| 🧾 Relatório LLM | `relatorio_llm.md` | Relatório automatizado gerado via IA. |

---

## 🧭 Exemplo de Saída (Resumo)

```text
1. A variável “TC018Q02NA – Mathematics” apresenta correlação positiva com o bem-estar docente.
   → Recomendação: investir em formação continuada no ensino de Matemática.

2. A variável “TC045Q01NA – Knowledge and understanding of my subject field(s)” tem forte peso.
   → Recomendação: ampliar programas de atualização pedagógica e científica.

3. Síntese geral:
   → Criar políticas de valorização docente, suporte emocional e indicadores nacionais de bem-estar.
```

---

## 🤝 Contribuição

Sinta-se à vontade para contribuir com melhorias na modelagem, novas análises ou sugestões de integração com outras bases educacionais.

**Formas de contribuição:**
- Fork do repositório;
- Pull request com melhoria documentada;
- Sugestões de novas hipóteses ou variáveis do PISA.

---

## 📚 Referências

- OECD (2019). *PISA 2018 Database*. Paris: OECD Publishing.  
- Hair Jr., J. F. et al. (2021). *Multivariate Data Analysis*. Pearson.  
- Witten, I. H., Frank, E., Hall, M. A. (2020). *Data Mining: Practical Machine Learning Tools and Techniques*. Elsevier.  
- Van Rossum, G. (2023). *The Python Language Reference Manual*. Python Software Foundation.

---

## 🧑‍🏫 Autor

**Prof. Kleber**  
📧 [email protected]  
🔬 Pesquisador em Tecnologia da Informação e Educação  
💡 Foco em análise de dados educacionais, sistemas de recomendação e políticas públicas baseadas em evidências.

---

## 🪄 Licença

Este projeto possui **finalidade acadêmica** e segue os princípios de uso livre para pesquisa e ensino.  
Cite a autoria original ao reproduzir total ou parcialmente os códigos, gráficos ou resultados.

---

✨ *“Os dados são o início da sabedoria — mas a interpretação é o que transforma números em ação.”*
