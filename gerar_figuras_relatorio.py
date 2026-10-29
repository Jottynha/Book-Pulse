#!/usr/bin/env python3
"""Gera figuras vetoriais do relatório a partir dos resultados versionados.

As figuras são produzidas com Plotly, sem dependência de Matplotlib ou Seaborn.
Os PDFs são usados pelo LaTeX; os SVGs preservam uma versão editável para a
apresentação e para inspeção dos resultados.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).resolve().parent
RESULTADOS = BASE_DIR / "processed"
SAIDA = BASE_DIR / "figures"
PALETA = {
    "teal": "#0e7770",
    "coral": "#db765f",
    "gold": "#d7a645",
    "blue": "#6386a5",
    "ink": "#1d2730",
    "muted": "#71808a",
    "line": "#dce4e0",
}


def layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 17, "color": PALETA["ink"]}},
        font={"family": "Arial", "size": 11, "color": PALETA["ink"]},
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 58, "r": 24, "t": 58, "b": 52},
        legend={"orientation": "h", "y": -0.18, "x": 0},
        hovermode=False,
    )
    fig.update_xaxes(showline=True, linecolor=PALETA["line"], gridcolor="#eef2f0", zeroline=False)
    fig.update_yaxes(showline=False, gridcolor="#eef2f0", zeroline=False)
    return fig


def export(fig: go.Figure, name: str) -> None:
    SAIDA.mkdir(exist_ok=True)
    fig.write_image(SAIDA / f"{name}.pdf", width=920, height=520, scale=2)
    fig.write_image(SAIDA / f"{name}.svg", width=920, height=520, scale=1)


def cobertura() -> None:
    valores = [10000, 8033, 7182, 7031]
    rotulos = ["Amostra inicial", "Reviews confiáveis", "Reviews em inglês", "Livros confiáveis"]
    fig = go.Figure(go.Bar(
        x=valores[::-1],
        y=rotulos[::-1],
        orientation="h",
        text=[f"{valor:,}".replace(",", ".") for valor in valores[::-1]],
        textposition="outside",
        marker_color=[PALETA["blue"], PALETA["teal"], PALETA["gold"], PALETA["coral"]],
    ))
    fig.update_layout(xaxis_title="Registros", yaxis_title="")
    fig.update_xaxes(range=[0, 11200], tickformat=",")
    export(layout(fig, "Redução da amostra ao longo do pipeline"), "cobertura_pipeline")


def sentimento_por_nota() -> None:
    dados = pd.read_csv(RESULTADOS / "09_predicao_nota" / "negativas_por_nota.csv")
    fig = go.Figure()
    cores = {"Transformer": PALETA["teal"], "VADER": PALETA["gold"], "Léxico": PALETA["coral"]}
    for coluna, cor in cores.items():
        fig.add_trace(go.Scatter(
            x=dados["rating_valido"], y=dados[coluna], mode="lines+markers",
            name=coluna, line={"color": cor, "width": 3},
            marker={"size": 8, "color": cor},
        ))
    fig.update_layout(xaxis_title="Nota Goodreads", yaxis_title="Reviews negativas (%)")
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    export(layout(fig, "Polaridade negativa por nota atribuída"), "negativas_por_nota")


def metodos() -> None:
    dados = pd.read_csv(RESULTADOS / "09_predicao_nota" / "comparacao_metodos.csv")
    nomes = {
        "transformer": "Transformer",
        "tfidf_logistic_regression": "TF-IDF + logística",
        "vader": "VADER",
        "lexical": "Léxico",
        "maioria": "Maioria",
    }
    dados["method"] = dados["method"].map(nomes)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Acurácia balanceada", "F1 da classe negativa"))
    fig.add_trace(go.Bar(
        x=dados["method"], y=dados["balanced_accuracy_mean"],
        error_y={"type": "data", "array": dados["balanced_accuracy_std"]},
        marker_color=PALETA["teal"], name="Acurácia balanceada",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=dados["method"], y=dados["f1_negative_mean"],
        error_y={"type": "data", "array": dados["f1_negative_std"]},
        marker_color=PALETA["coral"], name="F1 negativo",
    ), row=1, col=2)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", gridcolor="#eef2f0")
    fig.update_xaxes(tickangle=-28)
    fig.update_layout(
        title={"text": "Comparação dos métodos sob validação cruzada agrupada", "font": {"size": 17}},
        showlegend=False,
    )
    export(layout(fig, "Comparação dos métodos sob validação cruzada agrupada"), "comparacao_metodos")


def generos() -> None:
    dados = pd.read_csv(RESULTADOS / "06_analise" / "resumo_por_genero.csv").head(10)
    dados = dados.sort_values("average_rating")
    fig = go.Figure(go.Bar(
        x=dados["average_rating"], y=dados["genre_1"].str.replace("_", " ", regex=False),
        orientation="h", text=dados["average_rating"].round(2), textposition="outside",
        marker_color=PALETA["blue"],
        customdata=dados[["reviews", "average_words"]],
        hovertemplate="%{y}<br>nota: %{x:.2f}<br>reviews: %{customdata[0]}<br>palavras: %{customdata[1]:.1f}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Nota média", yaxis_title="")
    fig.update_xaxes(range=[3.5, 4.2], dtick=.1)
    export(layout(fig, "Nota média dos dez gêneros com maior presença na amostra"), "generos_nota")


def main() -> None:
    cobertura()
    sentimento_por_nota()
    metodos()
    generos()
    print(f"Figuras geradas em {SAIDA}")


if __name__ == "__main__":
    main()
