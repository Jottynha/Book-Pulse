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
import re
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
    parser.add_argument(
        "--annotations", nargs="+",
        help="CSV(s) ou URL da planilha (…/exec?format=csv). Aceita um CSV de "
             "referência, dois CSVs exportados por anotar.html ou um CSV com a "
             "coluna annotator contendo dois anotadores (Cohen's kappa).",
    )
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
    # Índice sequencial: as etapas seguintes concatenam colunas por posição.
    reviews = pd.read_parquet(caminho).reset_index(drop=True)
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
    reviews["text_language"] = reviews["text_for_inference"].map(detectar_idioma)
    reviews["text_is_english"] = reviews["text_language"].eq("en")
    return reviews


PALAVRAS_INGLES = {
    "the", "and", "is", "it", "to", "of", "this", "was", "i", "a", "in", "that",
    "book", "but", "for", "with", "not", "you", "my", "read", "loved", "love",
    "great", "good", "story", "series", "really", "so", "very",
}


def detectar_idioma(texto: str) -> str:
    """Idioma da review (código ISO 639-1) ou "und" quando indeterminado.

    langdetect é pouco confiável em textos muito curtos; nesses casos, uma
    review só de palavras ASCII com termos comuns do inglês ("Loved it!") é
    considerada inglês. O idioma do livro (`language_code`) não serve aqui:
    falta em parte da base e não diz em que língua a review foi escrita.
    """
    palavras = re.findall(r"[a-z']+", texto.lower())
    if not palavras:
        return "und"
    try:
        from langdetect import DetectorFactory, detect_langs
        from langdetect.lang_detect_exception import LangDetectException
    except ImportError as exc:
        raise RuntimeError("langdetect indisponível. Instale langdetect.") from exc
    DetectorFactory.seed = 0
    if len(palavras) >= 3:
        try:
            melhor = detect_langs(texto)[0]
            if melhor.prob >= 0.8:
                return melhor.lang
        except LangDetectException:
            pass
    if texto.isascii() and any(palavra in PALAVRAS_INGLES for palavra in palavras):
        return "en"
    return "und"


def componentes_usuario_livro(dados: pd.DataFrame) -> pd.Series:
    """Componente conexo de cada review no grafo bipartido usuário-livro.

    Reviews no mesmo componente compartilham usuário ou livro (direta ou
    indiretamente) e devem ficar do mesmo lado de um split.
    """
    pai: dict[str, str] = {}

    def raiz(no: str) -> str:
        pai.setdefault(no, no)
        while pai[no] != no:
            pai[no] = pai[pai[no]]
            no = pai[no]
        return no

    for usuario, livro in zip(dados["user_id"], dados["book_id"]):
        a, b = raiz(f"u:{usuario}"), raiz(f"b:{livro}")
        if a != b:
            pai[a] = b
    codigos: dict[str, int] = {}
    return pd.Series(
        [codigos.setdefault(raiz(f"u:{usuario}"), len(codigos)) for usuario in dados["user_id"]],
        index=dados.index,
    )


def gerar_amostra_anotacao(
    reviews: pd.DataFrame, tamanho: int, seed: int, saida: Path
) -> None:
    # Só reviews em inglês: é o idioma dos métodos avaliados (VADER, léxico e
    # transformer) e o par de idiomas da tradução de apoio na página.
    elegiveis = reviews[~reviews["text_is_empty"] & reviews["text_is_english"]].copy()
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
            "VADER indisponível. Instale vaderSentiment para gerar as colunas VADER."
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


def fragmentos(texto: str, limite: int = 350, sobreposicao: int = 50) -> list[str]:
    # 350 palavras ficam, em geral, abaixo do limite de 512 tokens do RoBERTa;
    # a truncagem do pipeline cobre apenas casos extremos.
    palavras = texto.split()
    if not palavras:
        return []
    passo = max(1, limite - sobreposicao)
    inicios = range(0, max(1, len(palavras) - sobreposicao), passo)
    return [" ".join(palavras[inicio:inicio + limite]) for inicio in inicios]


def inferir_transformer(
    reviews: pd.DataFrame, modelo_nome: str, batch_size: int, cache: Path
) -> pd.DataFrame:
    """Classifica cada review agregando todos os trechos.

    A probabilidade positiva da review é a média das probabilidades dos trechos,
    ponderada pelo número de palavras de cada trecho. Assim, uma review longa
    não é decidida por um único trecho isolado.
    """
    if cache.exists():
        anterior = pd.read_parquet(cache)
        if (
            anterior.attrs.get("model", modelo_nome) == modelo_nome
            and "review_id" in anterior
            and set(reviews["review_id"]).issubset(set(anterior["review_id"]))
        ):
            print(f"Transformer: reaproveitando cache {cache}")
            colunas = [coluna for coluna in anterior.columns if coluna != "review_id"]
            reviews = reviews.drop(columns=colunas, errors="ignore")
            return reviews.reset_index(drop=True).merge(anterior, on="review_id", how="left")
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Transformers indisponível. Instale transformers e torch para usar "
            "--run-transformer."
        ) from exc
    usar_gpu = torch.cuda.is_available()
    classificador = pipeline(
        "sentiment-analysis",
        model=modelo_nome,
        device=0 if usar_gpu else -1,
        dtype=torch.float16 if usar_gpu else torch.float32,
    )
    trechos: list[tuple[int, str]] = [
        (posicao, parte)
        for posicao, texto in enumerate(reviews["text_for_inference"])
        for parte in fragmentos(texto)
    ]
    # Ordenar por tamanho reduz padding dentro de cada lote.
    trechos.sort(key=lambda item: len(item[1]))
    probabilidades: dict[int, list[tuple[float, int]]] = {}
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    lotes = range(0, len(trechos), batch_size)
    for inicio in tqdm(lotes, desc="Transformer", unit="lote") if tqdm else lotes:
        lote = trechos[inicio:inicio + batch_size]
        previsoes = classificador(
            [parte for _, parte in lote],
            batch_size=batch_size,
            truncation=True,
            max_length=512,
            top_k=None,
        )
        for (posicao, parte), scores in zip(lote, previsoes):
            positiva = next(
                float(score["score"]) for score in scores
                if str(score["label"]).lower() == "positive"
            )
            probabilidades.setdefault(posicao, []).append((positiva, len(parte.split())))
    resultados: list[dict[str, Any]] = []
    for posicao in range(len(reviews)):
        partes = probabilidades.get(posicao, [])
        if not partes:
            resultados.append(
                {"transformer_label": "", "transformer_positive_prob": None,
                 "transformer_score": None, "transformer_chunks": 0}
            )
            continue
        peso_total = sum(peso for _, peso in partes)
        positiva = sum(prob * peso for prob, peso in partes) / peso_total
        resultados.append(
            {
                "transformer_label": "positive" if positiva >= 0.5 else "negative",
                "transformer_positive_prob": positiva,
                "transformer_score": max(positiva, 1 - positiva),
                "transformer_chunks": len(partes),
            }
        )
    saida = pd.DataFrame(resultados)
    saida.insert(0, "review_id", reviews["review_id"].to_numpy())
    saida.attrs["model"] = modelo_nome
    saida.to_parquet(cache, index=False)
    return pd.concat([reviews.reset_index(drop=True), saida.drop(columns="review_id")], axis=1)


def ultimo_rotulo(anotacoes: pd.DataFrame) -> pd.DataFrame:
    """Mantém a classificação mais recente de cada resenha."""
    anotacoes = anotacoes.dropna(subset=["label"])
    if "annotated_at" in anotacoes:
        anotacoes = anotacoes.sort_values("annotated_at", kind="stable")
    return anotacoes.drop_duplicates("review_id", keep="last")[["review_id", "label"]]


def carregar_anotacoes(caminhos: list[str]) -> pd.DataFrame:
    """Lê um CSV de referência ou combina dois anotadores independentes.

    Os dois anotadores podem vir em dois arquivos ou num único CSV (como o da
    planilha) com a coluna `annotator`. O rótulo de referência é aquele em que
    os anotadores concordam; divergências ficam sem `label` até serem
    reconciliadas.
    """
    tabelas = [pd.read_csv(caminho, dtype={"review_id": str, "annotator": str}) for caminho in caminhos]
    if len(tabelas) == 1 and "annotator" in tabelas[0]:
        anotadores = sorted(tabelas[0]["annotator"].dropna().unique())
        if len(anotadores) > 2:
            raise ValueError(f"Esperados até dois anotadores, encontrados: {anotadores}")
        tabelas = [tabelas[0][tabelas[0]["annotator"] == nome] for nome in anotadores]
        print(f"Anotadores: {', '.join(anotadores)}")
    if len(tabelas) == 1:
        return tabelas[0]
    if len(tabelas) != 2:
        raise ValueError("Use um CSV de referência ou exatamente dois anotadores.")
    a, b = (ultimo_rotulo(tabela) for tabela in tabelas)
    dados = a.merge(b, on="review_id", suffixes=("_a", "_b"))
    dados = dados.rename(columns={"label_a": "annotator_a", "label_b": "annotator_b"})
    dados["label"] = dados["annotator_a"].where(dados["annotator_a"] == dados["annotator_b"])
    return dados


def avaliar_anotacoes(
    reviews: pd.DataFrame, caminhos: list[str], saida: Path, seed: int
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
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import Pipeline

    anotacoes = carregar_anotacoes(caminhos)
    reviews = reviews.assign(review_id=reviews["review_id"].astype(str))
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
    for coluna in ["vader_label", "lexical_label", "transformer_label"]:
        if coluna not in dados:
            continue
        # O transformer é binário: só é comparável nas reviews rotuladas como
        # positive ou negative; as neutras ficam fora e são contadas à parte.
        classes = sorted(CLASSES) if coluna != "transformer_label" else ["negative", "positive"]
        validos = dados[coluna].isin(classes) & dados["label"].isin(classes)
        y_true, y_pred = dados.loc[validos, "label"], dados.loc[validos, coluna]
        if y_true.empty:
            continue
        metricas[coluna] = {
            "rows": int(validos.sum()),
            "excluded_reference_rows": int((~dados["label"].isin(classes)).sum()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "report": classification_report(
                y_true, y_pred, labels=classes, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        }
    if len(dados) >= 30 and dados["label"].nunique() >= 2:
        # Split agrupado: reviews que compartilham usuário ou livro ficam do
        # mesmo lado, evitando vazamento de estilo ou de vocabulário do livro.
        divisor = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
        indice_treino, indice_teste = next(
            divisor.split(dados, dados["label"], componentes_usuario_livro(dados))
        )
        treino, teste = dados.iloc[indice_treino], dados.iloc[indice_teste]
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
            "split": {"train": len(treino), "test": len(teste), "random_state": seed, "grouped_by": "user_book_component"},
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
- Reviews em inglês: {int(reviews["text_is_english"].sum()):,} (as demais ficam fora da amostra de anotação)
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
logística num split estratificado e agrupado por usuário e livro. Para calcular Cohen's kappa,
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
                "arguments": vars(args) | {"input": str(args.input), "output": str(args.output), "annotations": [str(caminho) for caminho in args.annotations] if args.annotations else None},
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
