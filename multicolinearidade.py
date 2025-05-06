from dataset import dataset
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

if __name__ == "__main__":
    dataset_copia: pd.DataFrame = dataset.copy().dropna()
    
    # === 1. Remove a variável dependente ===
    y = dataset_copia['tempo_resposta']
    X = dataset_copia.drop(columns=['tempo_resposta'])

    # === 2. Separa variáveis numéricas e categóricas ===
    var_num = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    var_cat = X.select_dtypes(include='object').columns.tolist()

    # === 3. Converte categóricas em dummies ===
    X_num = X[var_num].astype(float) if var_num else pd.DataFrame(index=X.index)
    X_cat = pd.get_dummies(X[var_cat], drop_first=True, dtype=float) if var_cat else pd.DataFrame(index=X.index)

    # === 4. Junta numéricas e dummies ===
    X_final = pd.concat([X_num, X_cat], axis=1)

    # === 5. Adiciona constante ===
    X_const = sm.add_constant(X_final)

    # === 6. Matriz de Correlação e pares com alta correlação ===
    print("\n=== Matriz de Correlação ===")
    corr_matrix = X_final.corr()
    print(corr_matrix)

    print("\n=== Pares de Variáveis com Correlação > 0.8 ===")
    alta_correlacao = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.8:
                alta_correlacao.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))
    for var1, var2, corr in alta_correlacao:
        print(f"{var1} x {var2} = {corr:.2f}")

    # === 7. Calcula VIF e Tolerance ===
    print("\n=== VIF e Tolerance ===")
    vif_df = pd.DataFrame()
    vif_df['Variável'] = X_const.columns
    vif_df['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    vif_df['Tolerance'] = 1 / vif_df['VIF']
    print(vif_df)

    print("\n=== Variáveis com VIF > 10 ou Tolerance < 0.1 ===")
    problema_vif = vif_df[(vif_df['VIF'] > 10) | (vif_df['Tolerance'] < 0.1)]
    print(problema_vif)