import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from modelo_regressao_linear import ajustar_regressao_linear

if __name__ == "__main__":
    # Ajusta o modelo de regressão linear com os dados já limpos
    modelo = ajustar_regressao_linear()

    # Obtém resíduos e valores ajustados
    residuos = modelo.resid
    valores_ajustados = modelo.fittedvalues

    # === Gráfico de Resíduos vs Valores Ajustados ===
    fig_residuos = px.scatter(
        x=valores_ajustados,
        y=residuos,
        labels={"x": "Valores Ajustados", "y": "Resíduos"},
        title="Resíduos vs Valores Ajustados"
    )
    fig_residuos.add_trace(go.Scatter(
        x=valores_ajustados,
        y=[0]*len(residuos),
        mode="lines",
        line=dict(color="red", dash="dash"),
        showlegend=False
    ))

    # === Teste de Breusch-Pagan ===
    # Parâmetros necessários para o teste
    bp_test = het_breuschpagan(residuos, modelo.model.exog)

    # Estatísticas do teste
    bp_stat = bp_test[0]
    bp_pvalor = bp_test[1]
    f_stat = bp_test[2]
    f_pvalor = bp_test[3]

    print("=== Teste de Breusch-Pagan ===")
    print(f"Estatística LM: {bp_stat:.4f}")
    print(f"p-valor LM: {bp_pvalor:.4f}")
    print(f"Estatística F: {f_stat:.4f}")
    print(f"p-valor F: {f_pvalor:.4f}")

    # Interpretação:
    if bp_pvalor < 0.05:
        print("→ Evidência de heterocedasticidade (p-valor < 0.05)")
    else:
        print("→ Sem evidência de heterocedasticidade (p-valor ≥ 0.05)")
        
    fig_residuos.show()