# Análise Estatística e Regressão Linear

Este projeto realiza uma análise exploratória e diagnóstica de um conjunto de dados, com foco em regressão linear, multicolinearidade e heterocedasticidade. Abaixo estão descrições dos arquivos incluídos:

---

## 📁 Arquivos

### `dataset.py`
Carrega o dataset principal (`dataset_9.csv`) utilizando `pandas`. Serve como ponto de partida para os demais scripts, exportando o DataFrame como `dataset`.

---

### `analise_estatistica.py`
Realiza análise estatística descritiva:
- Preenche valores ausentes com a mediana (variáveis numéricas) ou moda (variáveis categóricas).
- Gera visualizações interativas (barras para numéricas e pizza para categóricas) com uso de `plotly`.
- Permite alternar entre variáveis com botões dropdown.

---

### `modelo_regressao_linear.py`
Define a função `ajustar_regressao_linear()` que:
- Prepara os dados (remove a variável dependente, transforma variáveis categóricas em dummies).
- Ajusta um modelo de regressão linear com `statsmodels.OLS`.
- Exibe o sumário do modelo ao ser executado como script principal.

---

### `multicolinearidade.py`
Avalia a presença de multicolinearidade entre as variáveis independentes:
- Calcula a matriz de correlação entre variáveis.
- Identifica pares com correlação superior a 0.8.
- Calcula o Fator de Inflação da Variância (VIF) e Tolerância.
- Sinaliza variáveis problemáticas com VIF > 10 ou Tolerância < 0.1.

---

### `heterocedasticidade.py`
Verifica a presença de heterocedasticidade nos resíduos do modelo:
- Ajusta o modelo de regressão linear via `ajustar_regressao_linear`.
- Gera um gráfico de resíduos vs. valores ajustados.
- Executa o teste de Breusch-Pagan para detectar heterocedasticidade.
- Interpreta o p-valor para indicar presença ou ausência de problema.

---

## 🧩 Dependências
Certifique-se de instalar os seguintes pacotes antes de rodar os scripts:
```bash
pip install pandas plotly statsmodels