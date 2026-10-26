#!/usr/bin/env python3
"""Gera os dados agregados usados pelo site de exploração e recomendação."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ENTRADA = BASE_DIR / "processed" / "07_sentimento" / "reviews_sentimento.parquet"
SAIDA = BASE_DIR / "data" / "books.json"

STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "an", "and", "are", "as",
    "at", "be", "because", "been", "being", "but", "by", "book", "books",
    "can", "could", "did", "do", "does", "for", "from", "get", "had", "has",
    "have", "he", "her", "here", "him", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "just", "like", "me", "more", "most", "much", "my",
    "no", "not", "of", "on", "one", "only", "or", "other", "our", "out", "over",
    "read", "really", "she", "should", "so", "some", "than", "that", "the",
    "their", "them", "there", "these", "they", "this", "through", "to", "too",
    "was", "we", "were", "what", "when", "which", "who", "will", "with",
    "would", "you", "your",
}
TOKEN_RE = re.compile(r"^[a-z][a-z'-]{2,}$")


def texto(valor: object) -> str:
    if valor is None or pd.isna(valor):
        return ""
    return str(valor).strip()


def capa_url(row: pd.Series) -> str:
    for coluna in ("isbn13", "isbn"):
        identificador = texto(row.get(coluna)).replace("-", "")
        if identificador:
            return f"https://covers.openlibrary.org/b/isbn/{identificador}-L.jpg"
    return ""


def palavras(tokens: pd.Series, limite: int = 8) -> list[dict[str, object]]:
    contador: Counter[str] = Counter()
    for valor in tokens.dropna():
        for token in str(valor).split("|"):
            token = token.casefold()
            if TOKEN_RE.match(token) and token not in STOPWORDS:
                contador[token] += 1
    return [{"word": palavra, "count": quantidade} for palavra, quantidade in contador.most_common(limite)]


def agregar(df: pd.DataFrame) -> list[dict[str, object]]:
    livros: list[dict[str, object]] = []
    for book_id, grupo in df.groupby("book_id", sort=False):
        livro = grupo.iloc[0]
        validas = grupo[grupo["rating_valido"].notna()]
        total = len(grupo)
        proporcoes = grupo["vader_label"].value_counts(normalize=True)
        entry = {
            "id": str(book_id),
            "title": texto(livro.get("title")) or "Título não informado",
            "author": texto(livro.get("openlibrary_authors")) or texto(livro.get("google_authors")) or "Autor não informado",
            "cover_url": capa_url(livro),
            "cover_fallback_url": (
                capa_url(livro).replace("-L.jpg", "-M.jpg")
                if capa_url(livro)
                else ""
            ),
            "isbn": texto(livro.get("isbn13")) or texto(livro.get("isbn")),
            "genres": [valor for valor in [texto(livro.get(f"genre_{indice}")) for indice in range(1, 4)] if valor],
            "publication_year": int(livro["publication_year"]) if pd.notna(livro.get("publication_year")) else None,
            "average_rating": round(float(validas["rating_valido"].mean()), 2) if not validas.empty else None,
            "review_count": total,
            "average_words": round(float(grupo["refined_word_count"].mean()), 1),
            "sentiment": {
                "positive": round(float(proporcoes.get("positive", 0)), 3),
                "neutral": round(float(proporcoes.get("neutral", 0)), 3),
                "negative": round(float(proporcoes.get("negative", 0)), 3),
            },
            "sentiment_score": round(float(grupo["sentiment_score_per_100_words"].mean()), 3),
            "common_words": palavras(grupo["review_tokens"]),
            "positive_words": palavras(grupo.loc[grupo["vader_label"] == "positive", "review_tokens"]),
            "negative_words": palavras(grupo.loc[grupo["vader_label"] == "negative", "review_tokens"]),
        }
        livros.append(entry)
    return sorted(livros, key=lambda item: (-item["review_count"], item["title"].casefold()))


def main() -> None:
    if not ENTRADA.exists():
        raise FileNotFoundError(f"Arquivo de sentimento não encontrado: {ENTRADA}")
    df = pd.read_parquet(ENTRADA)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_from": str(ENTRADA.relative_to(BASE_DIR)),
        "book_count": int(df["book_id"].nunique()),
        "review_count": int(len(df)),
        "books": agregar(df),
    }
    SAIDA.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Dados do site gerados: {SAIDA}")
    print(f"Livros: {payload['book_count']:,}; reviews: {payload['review_count']:,}")


if __name__ == "__main__":
    main()
