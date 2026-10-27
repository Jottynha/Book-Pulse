#!/usr/bin/env python3
"""Predição da polaridade da nota a partir do texto, com rótulo fraco.

A nota Goodreads é convertida em rótulo binário (1-2 = negative, 4-5 =
positive; nota 3 fica fora). Esse rótulo mede "o texto prevê a nota?", e não
o sentimento validado por humanos. O script compara um classificador
supervisionado (TF-IDF + regressão logística) com os métodos já calculados no
passo 7 (lexical, VADER e transformer), sempre nas mesmas dobras de teste.

As dobras são agrupadas: reviews ligadas pelo mesmo usuário ou pelo mesmo
livro ficam no mesmo componente e nunca se dividem entre treino e teste. Uma
validação não agrupada é reportada à parte para medir o efeito de vazamento.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ENTRADA_PADRAO = BASE_DIR / "processed" / "07_sentimento" / "reviews_sentimento.parquet"
SAIDA_PADRAO = BASE_DIR / "processed" / "09_predicao_nota"
CLASSES = ["negative", "positive"]
METODOS_PRONTOS = {
    "lexical": "lexical_label",
    "vader": "vader_label",
    "transformer": "transformer_label",
}


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ENTRADA_PADRAO)
    parser.add_argument("--output", type=Path, default=SAIDA_PADRAO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--all-languages", action="store_true",
        help="Inclui reviews fora do inglês (por padrão ficam fora, pois os "
             "métodos comparados são para inglês).",
    )
    return parser.parse_args()


def rotulo_fraco(reviews: pd.DataFrame, todos_idiomas: bool = False) -> pd.DataFrame:
    dados = reviews[reviews["rating_valido"].notna() & reviews["rating_valido"].ne(3)].copy()
    if not todos_idiomas and "text_is_english" in dados:
        dados = dados[dados["text_is_english"]]
    dados["weak_label"] = np.where(dados["rating_valido"] >= 4, "positive", "negative")
    dados = dados[dados["text_for_inference"].str.strip().ne("")]
    return dados.reset_index(drop=True)


def componentes_usuario_livro(dados: pd.DataFrame) -> np.ndarray:
    """Reaproveita o agrupamento usuário-livro do passo 7."""
    caminho = BASE_DIR / "07_analise_sentimento.py"
    spec = importlib.util.spec_from_file_location("analise_sentimento", caminho)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Não foi possível carregar: {caminho}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo.componentes_usuario_livro(dados).to_numpy()


def metricas(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "f1_negative": float(f1_score(y_true, y_pred, pos_label="negative", average="binary", zero_division=0)),
    }


def novo_modelo(seed: int) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("classifier", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed
            )),
        ]
    )


def validar(
    dados: pd.DataFrame, divisor: Any, grupos: np.ndarray | None, seed: int
) -> list[dict[str, Any]]:
    linhas: list[dict[str, Any]] = []
    split = divisor.split(dados, dados["weak_label"], grupos)
    for dobra, (treino, teste) in enumerate(split, start=1):
        dados_treino, dados_teste = dados.iloc[treino], dados.iloc[teste]
        modelo = novo_modelo(seed).fit(dados_treino["text_for_inference"], dados_treino["weak_label"])
        y_true = dados_teste["weak_label"]
        previsoes = {
            "maioria": pd.Series(dados_treino["weak_label"].mode()[0], index=dados_teste.index),
            "tfidf_logistic_regression": pd.Series(
                modelo.predict(dados_teste["text_for_inference"]), index=dados_teste.index
            ),
        }
        for nome, coluna in METODOS_PRONTOS.items():
            if coluna in dados_teste:
                previsoes[nome] = dados_teste[coluna]
        for nome, y_pred in previsoes.items():
            # Métodos com classe neutra só são avaliados onde decidem a
            # polaridade; a fração descartada fica registrada em "coverage".
            decididas = y_pred.isin(CLASSES)
            linhas.append(
                {
                    "fold": dobra,
                    "method": nome,
                    "test_rows": len(dados_teste),
                    "coverage": float(decididas.mean()),
                    **metricas(y_true[decididas], y_pred[decididas]),
                }
            )
    return linhas


def termos_mais_informativos(dados: pd.DataFrame, seed: int, limite: int = 25) -> pd.DataFrame:
    modelo = novo_modelo(seed).fit(dados["text_for_inference"], dados["weak_label"])
    vocabulario = modelo.named_steps["tfidf"].get_feature_names_out()
    pesos = modelo.named_steps["classifier"].coef_[0]
    ordem = np.argsort(pesos)
    negativos = [(vocabulario[i], pesos[i], "negative") for i in ordem[:limite]]
    positivos = [(vocabulario[i], pesos[i], "positive") for i in ordem[::-1][:limite]]
    return pd.DataFrame(negativos + positivos, columns=["term", "coefficient", "direction"])


def figura_negativas_por_nota(reviews: pd.DataFrame, caminho: Path) -> pd.DataFrame:
    """Fração de reviews classificadas como negativas por nota, por método.

    Usa todas as notas (inclusive 3) das reviews em inglês. Linhas finas,
    marcadores distintos e rótulos diretos: a identidade não depende só da cor.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dados = reviews[reviews["rating_valido"].notna()]
    if "text_is_english" in dados:
        dados = dados[dados["text_is_english"]]
    series = {
        "Transformer": ("transformer_label", "#2a78d6", "o"),
        "VADER": ("vader_label", "#eb6834", "s"),
        "Léxico": ("lexical_label", "#1baf7a", "^"),
    }
    tabela = pd.DataFrame({
        nome: dados.groupby("rating_valido")[coluna].apply(lambda s: (s == "negative").mean() * 100)
        for nome, (coluna, _, _) in series.items()
    })
    fig, eixo = plt.subplots(figsize=(3.5, 2.5), dpi=300)
    for nome, (_, cor, marcador) in series.items():
        eixo.plot(
            tabela.index, tabela[nome], color=cor, linewidth=1.6,
            marker=marcador, markersize=4, label=nome,
        )
    # Legenda no canto vazio: na nota 5 as linhas se encontram e rótulos
    # diretos colidiriam.
    eixo.legend(frameon=False, fontsize=7, labelcolor="#3d3d3a", loc="upper right")
    eixo.set_xticks(tabela.index)
    eixo.set_xlabel("Nota Goodreads", fontsize=7, color="#3d3d3a")
    eixo.set_ylabel("Classificadas como negativas (%)", fontsize=7, color="#3d3d3a")
    eixo.tick_params(labelsize=7, colors="#5f5e5a", length=2)
    eixo.grid(axis="y", color="#e5e4df", linewidth=0.6)
    eixo.set_axisbelow(True)
    for lado in ("top", "right"):
        eixo.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eixo.spines[lado].set_color("#b5b4ad")
    fig.tight_layout()
    fig.savefig(caminho)
    plt.close(fig)
    return tabela.round(1)


def resumir(linhas: list[dict[str, Any]]) -> pd.DataFrame:
    tabela = pd.DataFrame(linhas)
    colunas = ["coverage", "accuracy", "balanced_accuracy", "macro_f1", "f1_negative"]
    resumo = tabela.groupby("method")[colunas].agg(["mean", "std"])
    resumo.columns = [f"{metrica}_{estatistica}" for metrica, estatistica in resumo.columns]
    return resumo.sort_values("balanced_accuracy_mean", ascending=False)


def main() -> None:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    args = argumentos()
    args.output.mkdir(parents=True, exist_ok=True)
    reviews = pd.read_parquet(args.input)
    figura_negativas_por_nota(reviews, args.output / "negativas_por_nota.png").to_csv(
        args.output / "negativas_por_nota.csv"
    )
    dados = rotulo_fraco(reviews, args.all_languages)
    grupos = componentes_usuario_livro(dados)

    agrupado = validar(
        dados,
        StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed),
        grupos,
        args.seed,
    )
    nao_agrupado = validar(
        dados,
        StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed),
        None,
        args.seed,
    )
    resumo = resumir(agrupado)
    resumo_vazamento = resumir(
        [linha for linha in nao_agrupado if linha["method"] == "tfidf_logistic_regression"]
    )
    pd.DataFrame(agrupado).to_csv(args.output / "dobras_agrupadas.csv", index=False)
    resumo.to_csv(args.output / "comparacao_metodos.csv")
    termos_mais_informativos(dados, args.seed).to_csv(
        args.output / "termos_tfidf.csv", index=False
    )
    (args.output / "metricas.json").write_text(
        json.dumps(
            {
                "label": "rótulo fraco: nota 1-2 = negative, 4-5 = positive, nota 3 excluída",
                "rows": len(dados),
                "languages": "all" if args.all_languages else "english_only",
                "class_counts": dados["weak_label"].value_counts().to_dict(),
                "groups": int(len(np.unique(grupos))),
                "largest_group_rows": int(np.bincount(grupos).max()),
                "folds": args.folds,
                "seed": args.seed,
                "grouped_cv": resumo.round(4).to_dict(orient="index"),
                "tfidf_ungrouped_cv": resumo_vazamento.round(4).to_dict(orient="index"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Predição de nota concluída: {args.output}")
    print(f"Reviews: {len(dados):,}; componentes usuário-livro: {len(np.unique(grupos)):,}")
    print(resumo[["coverage_mean", "balanced_accuracy_mean", "balanced_accuracy_std", "macro_f1_mean"]].round(3).to_string())
    print("TF-IDF sem agrupamento:")
    print(resumo_vazamento[["balanced_accuracy_mean", "macro_f1_mean"]].round(3).to_string())


if __name__ == "__main__":
    main()
