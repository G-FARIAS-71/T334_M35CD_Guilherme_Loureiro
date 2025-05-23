from dataset import dataset
from typing import Any
import pandas as pd
import plotly.graph_objects as go

if __name__ == "__main__":
    # Preenchimento de valores nulos
    dataset_copia = dataset.copy()
    for col in dataset_copia.select_dtypes(include=['float64', 'int64']).columns:
        mediana = dataset_copia[col].median()
        dataset_copia[col].fillna(mediana, inplace=True)

    for col in dataset_copia.select_dtypes(include='object').columns:
        if not dataset_copia[col].mode().empty:
            moda = dataset_copia[col].mode()[0]
            dataset_copia[col].fillna(moda, inplace=True)

    # Identifica tipos de variáveis
    var_num: list[str] = dataset_copia.select_dtypes(include=['float64', 'int64']).columns.tolist()
    var_cat: list[str] = dataset_copia.select_dtypes(include='object').columns.tolist()

    # Armazenar traços e botões
    fig: go.Figure = go.Figure()
    buttons: list[dict[str, Any]] = []
    traces: list[go.Bar | go.Pie] = []

    # === VARIÁVEIS NUMÉRICAS ===
    stats_var_num: dict[str, pd.Series] = {}
    for coluna in var_num:
        stats: pd.Series = dataset_copia[coluna].describe()
        stats_var_num[coluna] = stats

    df_stats_var_num: pd.DataFrame = pd.DataFrame(stats_var_num).transpose()

    for var in df_stats_var_num.index:
        valores: dict[str, float] = {
            'Média': df_stats_var_num.loc[var, 'mean'],
            'Desvio Padrão': df_stats_var_num.loc[var, 'std'],
            'Mínimo': df_stats_var_num.loc[var, 'min'],
            '25%': df_stats_var_num.loc[var, '25%'],
            '50%': df_stats_var_num.loc[var, '50%'],
            '75%': df_stats_var_num.loc[var, '75%'],
            'Máximo': df_stats_var_num.loc[var, 'max']
        }

        trace: go.Bar = go.Bar(
            x=list(valores.keys()),
            y=list(valores.values()),
            name=var,
            visible=False
        )
        fig.add_trace(trace)
        traces.append(trace)

    # === VARIÁVEIS CATEGÓRICAS ===
    for col in var_cat:
        valores: pd.Series = dataset_copia[col].value_counts(normalize=True)
        
        trace: go.Pie = go.Pie(
            labels=valores.index.tolist(),
            values=valores.values.tolist(),
            name=col,
            visible=False,
            hole=0.4
        )
        fig.add_trace(trace)
        traces.append(trace)

        # === BOTÕES DROPDOWN ===
    total_traces: int = len(var_num) + len(var_cat)
    for i, var in enumerate(var_num + var_cat):
        visibility: list[bool] = [False] * total_traces
        visibility[i] = True

        is_numerica: bool = i < len(var_num)
        layout_updates: dict[str, Any] = (
            {
                "title": {"text": f'Estatísticas Resumidas - {var}'},
                "xaxis": {"title": "Estatísticas", "visible": True},
                "yaxis": {"title": "Valor", "visible": True},
                "showlegend": False
            }
            if is_numerica else
            {
                "title": {"text": f'Proporção Relativa - {var}'},
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "showlegend": True
            }
        )

        button: dict[str, Any] = dict(
            label=var,
            method="update",
            args=[{"visible": visibility}, layout_updates]
        )
        buttons.append(button)

    # Layout final
    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=0.5,
            xanchor="center",
            y=1.2,
            yanchor="top"
        )],
        title=f'Estatísticas Resumidas - {var_num[0]}'
    )

    fig.data[0].visible = True
    fig.show()