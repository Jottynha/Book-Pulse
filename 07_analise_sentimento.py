#!/usr/bin/env python3
"""Análise de sentimento com inferência exploratória e validação opcional.

Sem um arquivo de anotações humanas, este script não calcula métricas de
desempenho. A nota Goodreads permanece uma variável de comparação, nunca o
ground truth do sentimento.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ENTRADA_PADRAO = BASE_DIR / "processed" / "05_dados_prontos" / "base_analise.parquet"
SAIDA_PADRAO = BASE_DIR / "processed" / "07_sentimento"
MODELO_TRANSFORMER_PADRAO = "siebert/sentiment-roberta-large-english"
CLASSES = {"negative", "neutral", "positive"}


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ENTRADA_PADRAO)
    parser.add_argument("--output", type=Path, default=SAIDA_PADRAO)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotation-size", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--run-transformer", action="store_true")
    parser.add_argument("--transformer-model", default=MODELO_TRANSFORMER_PADRAO)
    return parser.parse_args()


def carregar_baseline() -> Any:
    caminho = BASE_DIR / "06_analisar_reviews.py"
    spec = importlib.util.spec_from_file_location("analise_anterior", caminho)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Não foi possível carregar o baseline: {caminho}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def preparar_base(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise FileNotFoundError(f"Base de entrada não encontrada: {caminho}")
    reviews = pd.read_parquet(caminho).copy()
    obrigatorias = {"review_id", "review_text", "user_id", "book_id"}
    faltantes = obrigatorias - set(reviews.columns)
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(faltantes)}")
    reviews["review_text"] = reviews["review_text"].fillna("").astype(str)
    reviews["text_for_inference"] = reviews["review_text"].map(
        lambda texto: texto.replace("\x00", " ").strip()
    )
    reviews["text_is_empty"] = reviews["text_for_inference"].eq("")
    reviews["text_sha256"] = reviews["text_for_inference"].map(
        lambda texto: hashlib.sha256(texto.encode("utf-8")).hexdigest()
    )
    reviews["exact_duplicate_text"] = reviews["text_sha256"].duplicated(keep=False)
    return reviews


def gerar_amostra_anotacao(
    reviews: pd.DataFrame, tamanho: int, seed: int, saida: Path
) -> None:
    elegiveis = reviews[~reviews["text_is_empty"]].copy()
    if "rating_valido" in elegiveis:
        elegiveis["rating_group"] = pd.cut(
            elegiveis["rating_valido"],
            bins=[0, 2, 3, 5],
            labels=["rating_1_2", "rating_3", "rating_4_5"],
            include_lowest=True,
        ).astype("string").fillna("rating_missing")
    else:
        elegiveis["rating_group"] = "rating_missing"
    elegiveis["length_group"] = pd.qcut(
        elegiveis["text_for_inference"].str.len().rank(method="first"),
        q=4,
        labels=["short", "medium_short", "medium_long", "long"],
    )
    grupos = ["rating_group", "length_group"]
    partes = []
    for _, grupo in elegiveis.groupby(grupos, observed=True):
        partes.append(
            grupo.sample(
                n=min(len(grupo), max(1, round(tamanho * len(grupo) / len(elegiveis)))),
                random_state=seed,
            )
        )
    amostra = pd.concat(partes).drop_duplicates("review_id")
    if len(amostra) < min(tamanho, len(elegiveis)):
        restantes = elegiveis[~elegiveis["review_id"].isin(amostra["review_id"])]
        amostra = pd.concat(
            [amostra, restantes.sample(
                n=min(tamanho - len(amostra), len(restantes)),
                random_state=seed,
            )]
        )
    # Variáveis de estratificação ficam fora da planilha para não induzir o
    # julgamento dos anotadores; sua composição é preservada em pesos_amostragem.
    identificadores = ["review_id", "text_for_inference"]
    anotacao = amostra[identificadores].copy()
    anotacao["label"] = ""
    anotacao["uncertain"] = ""
    anotacao["notes"] = ""
    anotacao.sample(frac=1, random_state=seed).to_csv(
        saida / "amostra_anotacao.csv", index=False, encoding="utf-8"
    )
    pesos = amostra.groupby(grupos, observed=True).size().rename("sample_count")
    pesos.to_csv(saida / "pesos_amostragem.csv")


def salvar_guia(saida: Path) -> None:
    guia = """# Guia de anotação de sentimento

## Unidade e classes

Anote a polaridade expressa na review inteira, e não a nota Goodreads:

- `positive`: avaliação global predominantemente favorável;
- `negative`: avaliação global predominantemente desfavorável;
- `neutral`: texto sem avaliação clara ou predominantemente descritivo;
- `mixed`: sentimentos positivos e negativos relevantes sem predominância clara;
- `unclear`: texto insuficiente, ilegível ou impossível de interpretar.

`mixed` e `unclear` não devem ser convertidos automaticamente para as três
classes de um classificador. Antes da avaliação, o protocolo deve documentar se
serão excluídos, reconciliados ou tratados como uma classe adicional.

## Regras

1. Considere negação e contexto: “not good” não é positivo.
2. Elogio com ressalva recebe a polaridade global; registre a dúvida.
3. Crítica pontual não torna toda a review negativa se a avaliação global for
   favorável.
4. Ironia e sarcasmo devem ser marcados como `unclear` quando não houver
   evidência suficiente.
5. Descrição de enredo sem opinião é `neutral`.
6. Reviews vazias ou sem conteúdo avaliativo devem ser marcadas conforme a
   regra escolhida pela equipe e documentadas.

Pelo menos dois anotadores devem classificar independentemente uma parte da
amostra. Divergências devem ser reconciliadas após o cálculo da concordância.
O arquivo entregue aos anotadores não contém a nota original para reduzir viés.
"""
    (saida / "guia_anotacao.md").write_text(guia, encoding="utf-8")


def inferir_vader(reviews: pd.DataFrame) -> pd.DataFrame:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise RuntimeError(
            "VADER indisponível. Instale vaderSentiment ou execute sem --vader."
        ) from exc
    analisador = SentimentIntensityAnalyzer()

    def analisar(texto: str) -> dict[str, Any]:
        scores = analisador.polarity_scores(texto)
        compound = scores["compound"]
        label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
        return {**scores, "vader_label": label}

    scores = reviews["text_for_inference"].map(analisar).apply(pd.Series)
    scores = scores.rename(
        columns={
            "neg": "vader_neg",
            "neu": "vader_neu",
            "pos": "vader_pos",
            "compound": "vader_compound",
        }
    )
    return pd.concat([reviews.reset_index(drop=True), scores], axis=1)


def fragmentos(texto: str, limite: int = 450, sobreposicao: int = 50) -> list[str]:
    palavras = texto.split()
    if not palavras:
        return []
    passo = max(1, limite - sobreposicao)
    return [" ".join(palavras[inicio:inicio + limite]) for inicio in range(0, len(palavras), passo)]


def inferir_transformer(
    reviews: pd.DataFrame, modelo_nome: str, batch_size: int, cache: Path
) -> pd.DataFrame:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Transformers indisponível. Instale transformers e torch para usar "
            "--run-transformer."
        ) from exc
    classificador = pipeline("sentiment-analysis", model=modelo_nome, truncation=True)
    resultados: list[dict[str, Any]] = []
    for texto in reviews["text_for_inference"]:
        partes = fragmentos(texto)
        if not partes:
            resultados.append({"transformer_label": "", "transformer_score": None, "transformer_chunks": 0})
            continue
        previsoes = classificador(partes, batch_size=batch_size)
        melhor = max(previsoes, key=lambda previsao: previsao["score"])
        resultados.append(
            {
                "transformer_label": str(melhor["label"]).lower(),
                "transformer_score": float(melhor["score"]),
                "transformer_chunks": len(partes),
            }
        )
    saida = pd.DataFrame(resultados)
    saida.to_parquet(cache, index=False)
    return pd.concat([reviews.reset_index(drop=True), saida], axis=1)


def avaliar_anotacoes(
    reviews: pd.DataFrame, caminho: Path, saida: Path, seed: int
) -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        cohen_kappa_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    anotacoes = pd.read_csv(caminho)
    obrigatorias = {"review_id", "label"}
    if not obrigatorias.issubset(anotacoes.columns):
        raise ValueError("Anotações precisam conter review_id e label.")
    dados = reviews.merge(anotacoes[["review_id", "label"]], on="review_id", how="inner")
    dados = dados[dados["label"].isin(CLASSES)].copy()
    if dados.empty:
        raise ValueError("Nenhum rótulo válido entre negative, neutral e positive.")
    metricas: dict[str, Any] = {"reference_rows": len(dados)}
    if {"annotator_a", "annotator_b"}.issubset(anotacoes.columns):
        independentes = anotacoes[
            anotacoes["annotator_a"].isin(CLASSES)
            & anotacoes["annotator_b"].isin(CLASSES)
        ]
        if not independentes.empty:
            metricas["annotator_agreement"] = {
                "rows": len(independentes),
                "observed_agreement": float(
                    (independentes["annotator_a"] == independentes["annotator_b"]).mean()
                ),
                "cohen_kappa": float(
                    cohen_kappa_score(independentes["annotator_a"], independentes["annotator_b"])
                ),
            }
    for coluna in ["vader_label", "lexical_label"]:
        if coluna not in dados:
            continue
        validos = dados[coluna].isin(CLASSES)
        y_true, y_pred = dados.loc[validos, "label"], dados.loc[validos, coluna]
        metricas[coluna] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "report": classification_report(
                y_true, y_pred, labels=sorted(CLASSES), output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=sorted(CLASSES)).tolist(),
        }
    if len(dados) >= 30 and dados["label"].nunique() >= 2:
        treino, teste = train_test_split(
            dados,
            test_size=0.25,
            random_state=seed,
            stratify=dados["label"],
        )
        modelo = Pipeline(
            [
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1, 2), min_df=2, sublinear_tf=True
                )),
                ("classifier", LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=seed
                )),
            ]
        )
        modelo.fit(treino["text_for_inference"], treino["label"])
        predicoes = modelo.predict(teste["text_for_inference"])
        metricas["tfidf_logistic_regression"] = {
            "split": {"train": len(treino), "test": len(teste), "random_state": seed},
            "accuracy": float(accuracy_score(teste["label"], predicoes)),
            "balanced_accuracy": float(
                balanced_accuracy_score(teste["label"], predicoes)
            ),
            "report": classification_report(
                teste["label"],
                predicoes,
                labels=sorted(CLASSES),
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                teste["label"], predicoes, labels=sorted(CLASSES)
            ).tolist(),
        }
    (saida / "metricas_anotadas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metricas


def gerar_graficos(reviews: pd.DataFrame, saida: Path) -> None:
    try:
        import plotly.express as px
    except ImportError as exc:
        raise RuntimeError("Plotly indisponível. Instale plotly para gerar HTML.") from exc
    if "vader_label" in reviews:
        fig = px.histogram(
            reviews, x="rating_valido", color="vader_label", barmode="group",
            title="Nota Goodreads e classe VADER",
            labels={"rating_valido": "Nota Goodreads", "vader_label": "Sentimento VADER"},
        )
        fig.write_html(saida / "nota_vs_vader.html", include_plotlyjs="cdn")
    if "vader_compound" in reviews:
        fig = px.histogram(
            reviews[~reviews["text_is_empty"]],
            x="vader_compound",
            title="Distribuição das pontuações de sentimento VADER",
        )
        fig.write_html(saida / "distribuicao_sentimento.html", include_plotlyjs="cdn")


def salvar_relatorio(reviews: pd.DataFrame, args: argparse.Namespace, saida: Path) -> None:
    relatorio = f"""# Passo 7 — Análise de sentimento

## Objetivo e escopo

Foram processadas **{len(reviews):,} reviews** da base preparada. Sentimento
textual, nota Goodreads e predição de nota são variáveis diferentes. A nota é
usada apenas para associação exploratória; não é ground truth de sentimento.

- Reviews vazias: {int(reviews['text_is_empty'].sum()):,}
- Textos com duplicata exata: {int(reviews['exact_duplicate_text'].sum()):,}
- Seed: `{args.seed}`
- Modelo transformer solicitado: `{args.transformer_model}`
- Transformer executado: `{'transformer_label' in reviews}`

## Métodos

O baseline lexical do passo 6 é preservado. O VADER, quando disponível, produz
`neg`, `neu`, `pos` e `compound`, usando os limiares convencionais `-0.05` e
`0.05`. O transformer é opcional, processa trechos sobrepostos e agrega pelo
trecho de maior confiança. Nenhum modelo é considerado validado sem anotações
humanas.

## Validação

Use `--annotations` com um CSV contendo `review_id` e `label` (`negative`,
`neutral` ou `positive`) para gerar métricas. Sem esse arquivo, não são
calculadas acurácia, precisão, recall ou F1. A anotação deve seguir
`guia_anotacao.md`, envolver pelo menos dois anotadores em parte da amostra e
registrar concordância e reconciliação. Com pelo menos 30 rótulos válidos e
duas classes, o script também treina TF-IDF (unigramas e bigramas) + regressão
logística em um split estratificado separado. Para calcular Cohen's kappa,
inclua no CSV as colunas independentes `annotator_a` e `annotator_b`.

## Limitações

VADER foi criado para textos informais e o transformer pode ter domínio,
idioma, licença e limites de entrada diferentes dos dados Goodreads. Reviews
do mesmo usuário ou livro não são independentes. Correlações não demonstram
causalidade. Rótulos derivados de notas, quando usados, são rótulos fracos.
"""
    (saida / "relatorio_metodologico.md").write_text(relatorio, encoding="utf-8")


def main() -> None:
    args = argumentos()
    args.output.mkdir(parents=True, exist_ok=True)
    reviews = preparar_base(args.input)
    baseline = carregar_baseline()
    reviews = baseline.extrair_variaveis_textuais(reviews)
    reviews["lexical_label"] = reviews["sentiment_label"]
    gerar_amostra_anotacao(reviews, args.annotation_size, args.seed, args.output)
    salvar_guia(args.output)
    try:
        reviews = inferir_vader(reviews)
    except RuntimeError as erro:
        print(f"Aviso: {erro}", file=sys.stderr)
    if args.run_transformer:
        cache = args.output / "transformer_cache.parquet"
        try:
            reviews = inferir_transformer(reviews, args.transformer_model, args.batch_size, cache)
        except RuntimeError as erro:
            print(f"Aviso: {erro}", file=sys.stderr)
    if args.annotations:
        avaliar_anotacoes(reviews, args.annotations, args.output, args.seed)
    gerar_graficos(reviews, args.output)
    reviews.to_parquet(args.output / "reviews_sentimento.parquet", index=False)
    (args.output / "execucao.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "arguments": vars(args) | {"input": str(args.input), "output": str(args.output), "annotations": str(args.annotations) if args.annotations else None},
                "rows": len(reviews),
                "columns": list(reviews.columns),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    salvar_relatorio(reviews, args, args.output)
    print(f"Análise de sentimento concluída: {args.output}")


if __name__ == "__main__":
    main()
