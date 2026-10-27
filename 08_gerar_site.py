#!/usr/bin/env python3
"""Gera os dados agregados usados pelo site de exploração e recomendação."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


BASE_DIR = Path(__file__).resolve().parent
ENTRADA = BASE_DIR / "processed" / "07_sentimento" / "reviews_sentimento.parquet"
SAIDA = BASE_DIR / "data" / "books.json"

# Stopwords do inglês mais termos genéricos de qualquer resenha de livro.
STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "book", "books", "read", "reading", "really", "just", "like", "story",
    "good", "great", "time", "way", "lot", "did", "does", "got", "know",
    "think", "felt", "thing", "things", "pretty", "author", "end", "page",
    "pages", "chapter", "chapters", "review", "reviews", "star", "stars",
} | {
    # Palavras funcionais dos idiomas mais comuns entre as resenhas fora do
    # inglês (espanhol, português, italiano, francês, alemão e indonésio).
    "que", "del", "los", "las", "por", "para", "una", "con", "como", "pero",
    "mas", "muy", "uma", "nao", "mais", "muito", "isso", "ele", "ela", "che",
    "per", "non", "della", "sono", "les", "des", "est", "pas", "une", "qui",
    "dans", "und", "der", "die", "das", "ist", "nicht", "ein", "eine", "yang",
    "dan", "ini", "itu", "tidak", "dengan", "aku", "saya",
}
# Contrações ("it's", "didn't") ficam fora: são frequentes e não informativas.
TOKEN_RE = re.compile(r"^[a-z][a-z-]{2,}$")
# Uma palavra só entra se aparece nas resenhas de pelo menos dois livros:
# elimina erros de digitação e permite conectar títulos no grafo.
DOC_FREQ_MINIMA = 2


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


def contar(tokens: pd.Series) -> Counter[str]:
    contador: Counter[str] = Counter()
    for valor in tokens.dropna():
        for token in str(valor).split("|"):
            token = token.casefold()
            if TOKEN_RE.match(token) and token not in STOPWORDS:
                contador[token] += 1
    return contador


def frequencia_documental(df: pd.DataFrame) -> tuple[Counter[str], int]:
    """Em quantos livros cada palavra aparece (o livro é o documento)."""
    frequencia: Counter[str] = Counter()
    for _, grupo in df.groupby("book_id", sort=False):
        frequencia.update(set(contar(grupo["review_tokens"])))
    return frequencia, df["book_id"].nunique()


def palavras(
    tokens: pd.Series, frequencia: Counter[str], total_livros: int, limite: int = 8
) -> list[dict[str, object]]:
    """Palavras mais características do livro, por TF-IDF.

    A contagem exibida continua sendo a ocorrência bruta nas resenhas; o peso
    IDF só decide a ordem, rebaixando termos que aparecem em quase todo livro.
    """
    contador = contar(tokens)
    pontuadas = [
        (palavra, quantidade * math.log(total_livros / frequencia[palavra]))
        for palavra, quantidade in contador.items()
        if frequencia[palavra] >= DOC_FREQ_MINIMA
    ]
    pontuadas.sort(key=lambda item: (-item[1], item[0]))
    return [{"word": palavra, "count": contador[palavra]} for palavra, _ in pontuadas[:limite]]


def agregar(df: pd.DataFrame) -> list[dict[str, object]]:
    frequencia, total_livros = frequencia_documental(df)
    livros: list[dict[str, object]] = []
    for book_id, grupo in df.groupby("book_id", sort=False):
        livro = grupo.iloc[0]
        validas = grupo[grupo["rating_valido"].notna()]
        total = len(grupo)
        # O transformer é um modelo para inglês: as demais resenhas entram na
        # contagem e nas palavras, mas não no perfil de sentimento.
        ingles = grupo[grupo["text_is_english"] & grupo["transformer_label"].isin(["positive", "negative"])]
        proporcoes = ingles["transformer_label"].value_counts(normalize=True)
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
            "sentiment_reviews": len(ingles),
            "sentiment": {
                "positive": round(float(proporcoes.get("positive", 0)), 3),
                "negative": round(float(proporcoes.get("negative", 0)), 3),
            },
            "sentiment_score": (
                round(float(ingles["transformer_positive_prob"].mean()), 3) if len(ingles) else None
            ),
            "common_words": palavras(grupo["review_tokens"], frequencia, total_livros),
            "positive_words": palavras(
                ingles.loc[ingles["transformer_label"] == "positive", "review_tokens"],
                frequencia, total_livros,
            ),
            "negative_words": palavras(
                ingles.loc[ingles["transformer_label"] == "negative", "review_tokens"],
                frequencia, total_livros,
            ),
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
        "sentiment_model": "siebert/sentiment-roberta-large-english",
        "english_review_count": int(df["text_is_english"].sum()),
        "books": agregar(df),
    }
    SAIDA.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Dados do site gerados: {SAIDA}")
    print(f"Livros: {payload['book_count']:,}; reviews: {payload['review_count']:,}")


if __name__ == "__main__":
    main()
