import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from modelo_regressao_linear import ajustar_regressao_linear
from dataset import dataset

def stepwise_selection(dados: pd.DataFrame, target: str, threshold_in: float = 0.05, threshold_out: float = 0.10):
    """
    Realiza seleção de variáveis stepwise baseada em p-valor.
    """
    X = dados.drop(columns=[target])
    y = dados[target]

    # Dummies e tipos
    X_num = X.select_dtypes(include=['float64', 'int64']).astype(float)
    X_cat = pd.get_dummies(X.select_dtypes(include='object'), drop_first=True, dtype=float)
    X_all = pd.concat([X_num, X_cat], axis=1)

    included = []
    while True:
        changed = False
        excluded = list(set(X_all.columns) - set(included))
        
        # Forward step
        new_pvals = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X_all[included + [new_column]]))).fit()
            new_pvals[new_column] = model.pvalues[new_column]
        
        best_pval = new_pvals.min()
        if best_pval < threshold_in:
            best_feature = new_pvals.idxmin()
            included.append(best_feature)
            changed = True

        # Backward step
        if included:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X_all[included]))).fit()
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()
            if worst_pval > threshold_out:
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
        
        if not changed:
            break

    X_selected = sm.add_constant(X_all[included])
    final_model = sm.OLS(y, X_selected).fit()
    return final_model

if __name__ == "__main__":
    df = dataset.dropna()

    # === Modelo 1: Todas as variáveis ===
    modelo1 = ajustar_regressao_linear(df)

    # === Modelo 2: Stepwise ===
    modelo2 = stepwise_selection(df, target="tempo_resposta")

    # === Comparação ===
    print("\n=== MODELO 1 ===")
    print(modelo1.summary())
    
    print("\n=== MODELO 2 ===")
    print(modelo2.summary())

    print("\n=== COMPARAÇÃO DE MODELOS ===")
    print(f"R² Ajustado Modelo 1: {modelo1.rsquared_adj:.4f}")
    print(f"R² Ajustado Modelo 2: {modelo2.rsquared_adj:.4f}")

    f_test = anova_lm(modelo2, modelo1)
    print("\n=== TESTE F ENTRE OS MODELOS ===")
    print(f_test)