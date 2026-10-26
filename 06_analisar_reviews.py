#!/usr/bin/env python3
"""Refina o texto das reviews e produz a primeira análise comportamental."""

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PASTA_ENTRADA = BASE_DIR / "processed" / "05_dados_prontos"
PASTA_SAIDA = BASE_DIR / "processed" / "06_analise"
ARQUIVO_BASE = PASTA_ENTRADA / "base_analise.parquet"

TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", flags=re.UNICODE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)

# Vocabulário pequeno e explícito para uma linha de base reproduzível.
# Ele não substitui um modelo de sentimento treinado para o domínio.
PALAVRAS_POSITIVAS = {
    "amazing", "beautiful", "best", "enjoy", "enjoyed", "excellent", "favorite",
    "fun", "funny", "good", "great", "interesting", "love", "loved", "perfect",
    "recommend", "recommended", "wonderful", "worth",
}
PALAVRAS_NEGATIVAS = {
    "awful", "bad", "boring", "confusing", "disappointing", "dislike", "hate",
    "hated", "messy", "poor", "predictable", "slow", "terrible", "waste",
    "worst", "weak",
}


def refinar_texto(valor: object) -> tuple[str, list[str]]:
    """Retorna texto normalizado e tokens, preservando o texto original na base."""
    if valor is None or pd.isna(valor):
        return "", []
    texto = html.unescape(str(valor))
    texto = unicodedata.normalize("NFKC", texto)
    texto = URL_RE.sub(" ", texto).casefold()
    texto = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", texto)
    tokens = TOKEN_RE.findall(texto)
    return " ".join(tokens), tokens


def extrair_variaveis_textuais(reviews: pd.DataFrame) -> pd.DataFrame:
    refinado = reviews["review_text"].map(refinar_texto)
    reviews = reviews.copy()
    reviews["review_text_refined"] = refinado.map(lambda item: item[0])
    reviews["review_tokens"] = refinado.map(lambda item: "|".join(item[1]))
    reviews["refined_word_count"] = refinado.map(lambda item: len(item[1]))
    reviews["refined_char_count"] = reviews["review_text_refined"].str.len()
    reviews["refined_unique_word_ratio"] = reviews["review_tokens"].map(
        lambda valor: (
            len(set(valor.split("|"))) / len(valor.split("|"))
            if valor
            else 0.0
        )
    )
    reviews["sentence_count"] = reviews["review_text"].map(
        lambda valor: len(re.findall(r"[.!?]+", str(valor))) if valor else 0
    )

    def contar_sentimento(tokens: str, palavras: set[str]) -> int:
        return sum(token in palavras for token in tokens.split("|") if token)

    reviews["positive_word_count"] = reviews["review_tokens"].map(
        lambda tokens: contar_sentimento(tokens, PALAVRAS_POSITIVAS)
    )
    reviews["negative_word_count"] = reviews["review_tokens"].map(
        lambda tokens: contar_sentimento(tokens, PALAVRAS_NEGATIVAS)
    )
    reviews["sentiment_score"] = (
        reviews["positive_word_count"] - reviews["negative_word_count"]
    )
    reviews["sentiment_label"] = reviews["sentiment_score"].map(
        lambda score: "positive" if score > 0 else "negative" if score < 0 else "neutral"
    )
    return reviews


def resumir_por_nota(reviews: pd.DataFrame) -> pd.DataFrame:
    validas = reviews[reviews["rating_valido"].notna()].copy()
    return (
        validas.groupby("rating_valido", as_index=False)
        .agg(
            reviews=("review_id", "count"),
            average_words=("refined_word_count", "mean"),
            median_words=("refined_word_count", "median"),
            average_sentiment=("sentiment_score", "mean"),
            positive_share=("sentiment_label", lambda valores: (valores == "positive").mean()),
            average_votes=("n_votes", "mean"),
            average_comments=("n_comments", "mean"),
        )
        .sort_values("rating_valido")
    )


def resumir_por_genero(reviews: pd.DataFrame) -> pd.DataFrame:
    validas = reviews[
        reviews["genre_1"].notna() & reviews["genre_1"].ne("")
    ].copy()
    return (
        validas.groupby("genre_1", as_index=False)
        .agg(
            reviews=("review_id", "count"),
            average_rating=("rating_valido", "mean"),
            average_words=("refined_word_count", "mean"),
            average_sentiment=("sentiment_score", "mean"),
            average_votes=("n_votes", "mean"),
        )
        .sort_values(["reviews", "average_rating"], ascending=[False, False])
    )


def resumir_por_usuario(reviews: pd.DataFrame) -> pd.DataFrame:
    return (
        reviews.groupby("user_id", as_index=False)
        .agg(
            reviews_in_sample=("review_id", "count"),
            average_rating=("rating_valido", "mean"),
            average_words=("refined_word_count", "mean"),
            average_votes=("n_votes", "mean"),
            average_comments=("n_comments", "mean"),
        )
        .sort_values("reviews_in_sample", ascending=False)
    )


def salvar_graficos(resumo_nota: pd.DataFrame, reviews: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(resumo_nota["rating_valido"], resumo_nota["reviews"], color="#4472C4")
    axes[0].set_title("Reviews por nota")
    axes[0].set_xlabel("Nota")
    axes[0].set_ylabel("Quantidade")
    axes[0].set_xticks(resumo_nota["rating_valido"])

    validas = reviews[reviews["rating_valido"].notna()]
    axes[1].scatter(
        validas["rating_valido"],
        validas["sentiment_score"],
        alpha=0.25,
        s=12,
        color="#ED7D31",
    )
    axes[1].set_title("Nota e sentimento lexical")
    axes[1].set_xlabel("Nota")
    axes[1].set_ylabel("Score (positivo - negativo)")
    fig.tight_layout()
    fig.savefig(PASTA_SAIDA / "visao_geral_reviews.png", dpi=150)
    plt.close(fig)


def salvar_relatorio(
    reviews: pd.DataFrame,
    resumo_nota: pd.DataFrame,
    resumo_genero: pd.DataFrame,
    resumo_usuario: pd.DataFrame,
) -> None:
    validas = reviews[reviews["rating_valido"].notna()]
    correlacoes = validas[
        [
            "rating_valido",
            "refined_word_count",
            "sentiment_score",
            "n_votes",
            "n_comments",
            "ratings_count",
        ]
    ].corr()["rating_valido"].drop("rating_valido")
    correlacoes.to_csv(PASTA_SAIDA / "correlacoes_com_rating.csv", header=["correlation"])

    top_generos = resumo_genero.head(10)
    linhas_generos = "\n".join(
        f"- `{row.genre_1}`: {int(row.reviews):,} reviews, "
        f"nota média {row.average_rating:.2f}"
        for row in top_generos.itertuples()
    ) or "- Nenhum gênero com dados suficientes."
    relatorio = f"""# Passo 6 — Refinamento textual e análise inicial

## Escopo

Esta etapa usa `base_analise.parquet`, que já contém somente matches confiáveis.
O texto original (`review_text`) é preservado. A coluna `review_text_refined`
remove HTML, URLs, caracteres de controle, diferenças de caixa e pontuação,
mantendo apenas tokens para facilitar as análises seguintes.

- Reviews processadas: {len(reviews):,}
- Reviews com texto refinado: {int(reviews['refined_word_count'].gt(0).sum()):,}
- Reviews sem texto: {int(reviews['refined_word_count'].eq(0).sum()):,}
- Média de palavras: {reviews['refined_word_count'].mean():.1f}
- Usuários na amostra: {resumo_usuario['user_id'].nunique():,}

## Primeiros indicadores comportamentais

Os arquivos `resumo_por_nota.csv`, `resumo_por_genero.csv`,
`resumo_por_usuario.csv` e `correlacoes_com_rating.csv` permitem investigar:

- se reviews com notas diferentes têm extensões e padrões de linguagem distintos;
- se gênero está associado a nota, tamanho ou votos de utilidade;
- se usuários mais ativos na amostra escrevem reviews diferentes;
- a associação entre nota, votos, comentários, tamanho e popularidade do livro.

### Gêneros mais frequentes

{linhas_generos}

## Sentimento lexical

`sentiment_score` é a contagem de palavras positivas menos negativas em um
vocabulário pequeno e explícito em inglês. `sentiment_label` classifica o score
como `positive`, `negative` ou `neutral`. Isso é uma linha de base auditável,
não uma medição definitiva: não trata bem negação, ironia, contexto ou
vocabulário específico de livros. As conclusões principais devem ser
confirmadas com uma abordagem de NLP mais robusta.

## Saídas

- `reviews_refinadas.parquet`: base com tokens e variáveis textuais;
- `resumo_por_nota.csv`: métricas agrupadas pela nota;
- `resumo_por_genero.csv`: métricas agrupadas pelo gênero principal;
- `resumo_por_usuario.csv`: atividade agregada por usuário;
- `correlacoes_com_rating.csv`: correlações exploratórias com a nota;
- `visao_geral_reviews.png`: distribuição de notas e relação inicial com sentimento.
"""
    (PASTA_SAIDA / "relatorio_analise.md").write_text(relatorio, encoding="utf-8")


def main() -> None:
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    reviews = extrair_variaveis_textuais(pd.read_parquet(ARQUIVO_BASE))
    resumo_nota = resumir_por_nota(reviews)
    resumo_genero = resumir_por_genero(reviews)
    resumo_usuario = resumir_por_usuario(reviews)

    reviews.to_parquet(PASTA_SAIDA / "reviews_refinadas.parquet", index=False)
    resumo_nota.to_csv(PASTA_SAIDA / "resumo_por_nota.csv", index=False)
    resumo_genero.to_csv(PASTA_SAIDA / "resumo_por_genero.csv", index=False)
    resumo_usuario.to_csv(PASTA_SAIDA / "resumo_por_usuario.csv", index=False)
    salvar_graficos(resumo_nota, reviews)
    salvar_relatorio(reviews, resumo_nota, resumo_genero, resumo_usuario)
    print(f"Análise inicial concluída: {PASTA_SAIDA}")
    print(f"Reviews refinadas: {len(reviews):,}")


if __name__ == "__main__":
    main()
