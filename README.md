# 🧩 PISA 2018 – Pipeline de Descoberta de Conhecimento com IA  
## **Análise do Bem-Estar e Autoeficácia Docente no Chile (Professores de Matemática)**

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Estável-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-PISA%202018-orange)
![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20Gemini-blueviolet)

---

## 📝 Descrição Geral

Este projeto implementa um **pipeline completo de Descoberta de Conhecimento (KDD)** aplicado aos microdados do **PISA 2018**, com foco no **bem-estar e autoeficácia docente** entre professores de Matemática no **Chile**.  
O sistema une:

- Estatística clássica (OLS)  
- Aprendizado de máquina (Random Forest, Gradient Boosting)  
- Mineração de dados (PCA, K-Means)  
- Engenharia de índices derivados  
- Visualização analítica  
- Geração automática de relatórios com **LLMs (Groq LLaMA / Google Gemini)**  

O objetivo final é transformar dados quantitativos em **evidências interpretáveis** e **recomendações de políticas públicas**.

---

## 🧠 Arquitetura Geral do Pipeline (11 Etapas)

O pipeline segue a metodologia **KDD – Knowledge Discovery in Databases**:

| Etapa | Nome | Finalidade |
|------|-------|------------|
| **1** | Escolha do Cenário | Define país, disciplina, público e tema. |
| **2** | Formulação da Hipótese | Estabelece a hipótese científica e variáveis. |
| **3** | Coleta de Dados | Lê os microdados e o questionário docente. |
| **4** | Pré-processamento | Limpeza, padronização e tratamento de ausentes. |
| **5** | Transformação | Criação de índices derivados normalizados. |
| **6** | Mineração de Dados | PCA + K-Means para segmentação docente. |
| **7** | Descoberta de Modelos | Ajusta OLS e compara com modelos de ML. |
| **8** | Interpretação e Visualização | Gera gráficos, tabelas e estatísticas. |
| **9** | Refinamento do Conhecimento | Seleciona variáveis relevantes (p-valor ou importância). |
| **10** | Recomendações | Traduz achados em diretrizes para políticas públicas. |
| **11** | Relatório via LLM | Gera relatório executivo utilizando IA generativa. |

---

## 📂 Estrutura do Projeto

```
├── main.py
├── .env
│
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
│
├── dados/
│   ├── TCH_CH_Respostas.xlsx
│   └── TCH_CHL_Questionario.xlsx
│
└── resultados/
    ├── tabelas/
    ├── figuras/
    ├── textos/
    └── textos_llm/
```

---

## 🧬 Dados Utilizados

### **1. TCH_CH_Respostas.xlsx**
Microdados das respostas dos professores:

- Clima escolar  
- Carga de trabalho  
- Autoeficácia  
- Satisfação  
- Estresse  
- Informações demográficas  

### **2. TCH_CHL_Questionario.xlsx**
Dicionário de variáveis contendo:

- Descrição dos itens  
- Interpretação pedagógica  
- Escalas Likert  
- Agrupamento temático  

---

## ⚙️ Tecnologias e Bibliotecas Utilizadas

| Categoria | Ferramentas |
|----------|-------------|
| Manipulação de Dados | Pandas, NumPy |
| Estatística e ML | Statsmodels, Scikit-learn |
| Mineração | PCA, K-Means |
| Visualização | Matplotlib |
| LLM | Groq (LLaMA 3.3), Google Gemini |
| Configuração | dotenv |
| Exportação | CSV, PNG, Markdown |

---

## 🚀 Como Executar

### 1️⃣ Instalar dependências
```bash
pip install -r requirements.txt
```

### 2️⃣ Configurar o `.env`
```bash
GOOGLE_API_KEY=sua_chave_google
GROQ_API_KEY=sua_chave_groq
```

### 3️⃣ Executar o pipeline completo
```bash
python main.py
```

### 4️⃣ Executar com LLM específico
#### Google Gemini:
```bash
python main.py --llm-provider google --llm-model gemini-2.5-flash
```

#### Groq (LLaMA 3.3 70B):
```bash
python main.py --llm-provider groq --llm-model llama-3.3-70b-versatile
```

### 5️⃣ Executar sem LLM
```bash
python main.py --no-llm
```

---

## 📊 Saídas Geradas

As saídas são armazenadas em `/resultados/`:

### **Tabelas**
- `modelo_ols_resultados.csv`
- `comparacao_modelos.csv`
- `variaveis_importancia_rf.csv`
- `variaveis_significativas_ols.csv`
- `correlacoes.csv`
- `composicao_indices.json`

### **Figuras**
- Mapa de calor de correlações  
- PCA por cluster  
- Boxplots  
- Histogramas  

### **Textos**
- `recomendacoes_politicas_publicas.txt`
- `relatorio_final_llm.md`

---

## 🧩 Recomendações de Políticas Públicas

O pipeline traduz achados estatísticos em recomendações, como:

1. Investir na formação continuada de professores de Matemática.  
2. Promover ações de apoio emocional ao docente.  
3. Criar indicadores nacionais de bem-estar docente.  
4. Reduzir fatores de estresse ligados à carga administrativa.  

---

## 🧑‍🏫 Autores

Christiane
Kleber Galvão
Mariah

---

## 📄 Licença

Projeto de uso estritamente acadêmico.  
Cite a autoria ao utilizar códigos ou resultados.

---

✨ *“Os dados são o início; a interpretação é o caminho; a política pública é o impacto.”*
