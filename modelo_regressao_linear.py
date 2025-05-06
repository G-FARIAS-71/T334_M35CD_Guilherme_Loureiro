from dataset import dataset
import statsmodels.api as sm
import pandas as pd
from pandas import DataFrame, Series
from statsmodels.regression.linear_model import RegressionResultsWrapper

def ajustar_regressao_linear(dados=dataset.copy().dropna()) -> RegressionResultsWrapper:
    if 'tempo_resposta' not in dados.columns:
        raise ValueError("A coluna 'tempo_resposta' não está presente no conjunto de dados.")

    # Trata variável dependente
    y: Series = dados['tempo_resposta'].astype(float)
    dados = dados.drop(columns=['tempo_resposta'])

    # Identifica colunas numéricas e categóricas
    var_num: list[str] = dados.select_dtypes(include=['float64', 'int64']).columns.tolist()
    var_cat: list[str] = dados.select_dtypes(include='object').columns.tolist()

    # Variáveis explicativas numéricas
    X_num: DataFrame = dados[var_num].astype(float) if var_num else pd.DataFrame(index=dados.index)

    # Variáveis categóricas transformadas em dummies
    X_cat: DataFrame = pd.get_dummies(dados[var_cat], drop_first=True, dtype=float) if var_cat else pd.DataFrame(index=dados.index)

    # Combina todas as variáveis explicativas
    X: DataFrame = pd.concat([X_num, X_cat], axis=1)

    # Adiciona constante ao modelo
    X = sm.add_constant(X)

    # Ajusta o modelo de regressão linear
    modelo: RegressionResultsWrapper = sm.OLS(y, X).fit()

    return modelo

if __name__ == "__main__":
    modelo: RegressionResultsWrapper = ajustar_regressao_linear()
    print(modelo.summary())