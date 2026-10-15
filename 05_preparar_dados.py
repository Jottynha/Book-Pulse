#!/usr/bin/env python3
"""Prepara as bases confiáveis para a etapa de análise."""

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PASTA_PROCESSED = BASE_DIR / "processed"
PASTA_VALIDACAO = PASTA_PROCESSED / "04_validacao"
PASTA_SAIDA = PASTA_PROCESSED / "05_dados_prontos"

ARQUIVO_BOOKS = PASTA_VALIDACAO / "goodreads_books_validated_100k.parquet"
ARQUIVO_REVIEWS = PASTA_VALIDACAO / "goodreads_reviews_validated_100k.parquet"
STATUS_CONFIAVEIS = {"MUITO_ALTO", "ALTO", "MEDIO"}

# Termos genéricos como "fiction", "books" e "novels" não recebem uma
# categoria própria; eles não descrevem o gênero com precisão suficiente.
GENEROS = {
    "science_fiction": ["science fiction", "sci fi", "sci-fi", "scifi", "dystopia", "dystopian"],
    "fantasy": ["fantasy", "high fantasy", "urban fantasy", "epic fantasy"],
    "romance": ["romance", "romantic", "contemporary romance", "historical romance"],
    "mystery": ["mystery", "crime", "detective", "cozy mystery"],
    "thriller": ["thriller", "suspense", "psychological thriller"],
    "horror": ["horror", "gothic horror", "ghost story"],
    "historical_fiction": ["historical fiction", "historical novel", "historical"],
    "contemporary_fiction": ["contemporary fiction", "contemporary"],
    "young_adult": ["young adult", "ya fiction", "ya books", "teen fiction", "teen"],
    "children": ["children", "childrens", "middle grade", "picture book"],
    "adventure": ["adventure", "action adventure", "survival"],
    "literary_fiction": ["literary fiction", "literature"],
    "classics": ["classic", "classics"],
    "nonfiction": ["non-fiction", "nonfiction", "non fiction"],
    "biography": ["biography", "memoir", "autobiography"],
    "history": ["world history", "history"],
    "poetry": ["poetry", "poems", "poem"],
    "drama": ["drama", "plays", "play"],
    "philosophy": ["philosophy", "philosophical"],
    "religion": ["religion", "christian", "spirituality"],
    "self_help": ["self help", "self-help", "personal development"],
}
def limpar_texto(valor: object) -> str:
    if valor is None or pd.isna(valor):
        return ""
    texto = html.unescape(str(valor))
    texto = unicodedata.normalize("NFKC", texto)
    texto = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def limpar_lista(valor: object) -> str:
    itens = [limpar_texto(item) for item in str(valor or "").split("|")]
    itens = sorted({item for item in itens if item}, key=str.casefold)
    return "|".join(itens)


def numero(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie, errors="coerce")


def normalizar_rotulo(valor: str) -> str:
    valor = unicodedata.normalize("NFKD", valor.lower())
    valor = "".join(char for char in valor if not unicodedata.combining(char))
    valor = re.sub(r"[^a-z0-9]+", " ", valor)
    return re.sub(r"\s+", " ", valor).strip()


ALIASES_GENEROS = {
    normalizar_rotulo(alias): genero
    for genero, aliases in GENEROS.items()
    for alias in aliases
}


def rotulos(valor: object) -> list[str]:
    texto = limpar_texto(valor)
    partes = re.split(r"[|,;/]+", texto)
    return [normalizar_rotulo(parte) for parte in partes if limpar_texto(parte)]


def encontrar_generos(row: pd.Series) -> dict[str, object]:
    pontuacao: dict[str, int] = {}
    evidencias: dict[str, set[str]] = {}
    fontes = [
        ("popular_shelves", row.get("popular_shelves"), 1),
        ("openlibrary_subjects", row.get("openlibrary_subjects"), 2),
        ("google_categories", row.get("google_categories"), 2),
    ]
    for fonte, valor, peso in fontes:
        for rotulo in rotulos(valor):
            generos_encontrados = {
                genero
                for alias, genero in ALIASES_GENEROS.items()
                if alias == rotulo or alias in rotulo
            }
            for genero in generos_encontrados:
                pontuacao[genero] = pontuacao.get(genero, 0) + peso
                evidencias.setdefault(genero, set()).add(f"{fonte}:{rotulo}")

    ordenados = sorted(pontuacao, key=lambda genero: (-pontuacao[genero], genero))[:3]
    resultado = {
        "genre_1": ordenados[0] if len(ordenados) > 0 else "",
        "genre_2": ordenados[1] if len(ordenados) > 1 else "",
        "genre_3": ordenados[2] if len(ordenados) > 2 else "",
        "genres_assigned": "|".join(ordenados),
        "genre_evidence": "|".join(
            f"{genero}:{','.join(sorted(evidencias[genero]))}"
            for genero in ordenados
        ),
        "genre_count": len(ordenados),
        "genre_assignment_status": "assigned" if ordenados else "unclassified",
    }
    return resultado


def preparar_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    reviews = reviews[reviews["match_status"].isin(STATUS_CONFIAVEIS)].copy()
    reviews = reviews.drop_duplicates("review_id", keep="first")
    reviews["review_text"] = reviews["review_text"].map(limpar_texto)
    reviews["rating"] = numero(reviews["rating"])
    reviews["rating_valido"] = reviews["rating"].where(reviews["rating"].between(1, 5))
    reviews["n_votes"] = numero(reviews["n_votes"]).fillna(0).clip(lower=0)
    reviews["n_comments"] = numero(reviews["n_comments"]).fillna(0).clip(lower=0)
    for coluna in ["date_added", "date_updated", "read_at", "started_at"]:
        reviews[coluna] = pd.to_datetime(
            reviews[coluna], errors="coerce", format="mixed", utc=True
        )
    reviews["review_chars"] = reviews["review_text"].str.len()
    reviews["review_words"] = reviews["review_text"].str.split().str.len()
    reviews["review_is_empty"] = reviews["review_text"].eq("")
    reviews["review_has_non_latin"] = reviews["review_text"].map(
        lambda texto: any(ord(char) > 127 for char in texto)
    )
    colunas = [
        "review_id", "user_id", "book_id", "rating", "rating_valido", "review_text",
        "date_added", "date_updated", "read_at", "started_at", "n_votes",
        "n_comments", "review_chars", "review_words", "review_is_empty",
        "review_has_non_latin", "match_status",
    ]
    return reviews[colunas]


def preparar_books(books: pd.DataFrame) -> pd.DataFrame:
    books = books.drop_duplicates("book_id", keep="first").copy()
    colunas_texto = [
        "title", "title_without_series", "description", "publisher", "language_code",
        "country_code", "format", "popular_shelves", "series", "openlibrary_subjects",
        "openlibrary_authors", "google_categories", "google_authors",
    ]
    for coluna in colunas_texto:
        books[coluna] = books[coluna].map(limpar_texto)
    books["author_ids"] = books["author_ids"].map(limpar_lista)
    for coluna in [
        "publication_year", "num_pages", "average_rating", "ratings_count",
        "text_reviews_count", "openlibrary_pages", "google_page_count",
        "google_average_rating", "google_ratings_count",
    ]:
        books[coluna] = numero(books[coluna])
    books["is_ebook"] = books["is_ebook"].astype(str).str.lower().eq("true")
    generos = books.apply(encontrar_generos, axis=1, result_type="expand")
    books = pd.concat([books.reset_index(drop=True), generos], axis=1)
    colunas = [
        "book_id", "title", "title_without_series", "author_ids", "isbn", "isbn13",
        "publication_year", "num_pages", "description", "publisher", "language_code",
        "country_code", "format", "is_ebook", "average_rating", "ratings_count",
        "text_reviews_count", "popular_shelves", "series", "openlibrary_subjects",
        "openlibrary_authors", "google_categories", "google_authors", "genre_1",
        "genre_2", "genre_3", "genres_assigned", "genre_evidence", "genre_count",
        "genre_assignment_status", "match_status",
    ]
    return books[colunas]


def salvar_relatorio(reviews: pd.DataFrame, books: pd.DataFrame, base: pd.DataFrame) -> None:
    generos = books[books["genre_count"] > 0]
    relatorio = f"""# Dados preparados para análise

A etapa final usa somente matches `MUITO_ALTO`, `ALTO` e `MEDIO`.

## Cobertura

- Reviews prontas: {len(reviews):,}
- Livros prontos: {len(books):,}
- Reviews sem texto: {int(reviews['review_is_empty'].sum()):,}
- Reviews com caracteres não ASCII: {int(reviews['review_has_non_latin'].sum()):,}
- Livros com pelo menos um gênero atribuído: {len(generos):,}
- Livros sem gênero específico: {int((books['genre_count'] == 0).sum()):,}
- Livros com 1 gênero: {int((books['genre_count'] == 1).sum()):,}
- Livros com 2 gêneros: {int((books['genre_count'] == 2).sum()):,}
- Livros com 3 gêneros: {int((books['genre_count'] == 3).sum()):,}

## Critério de gêneros

Foram selecionados no máximo três gêneros por livro. A atribuição combina
`popular_shelves` do Goodreads, assuntos da Open Library e categorias do Google
Books. Assuntos/categorias externos recebem peso maior que shelves sociais.

`genre_evidence` preserva os rótulos que sustentaram cada atribuição. Quando não
há evidência específica suficiente, o livro fica como `unclassified`; nenhum
rótulo genérico é inventado para preencher a lacuna.

## Arquivos

- `reviews_prontas.parquet`: reviews limpas e filtradas;
- `livros_prontos.parquet`: livros normalizados com até três gêneros;
- `base_analise.parquet`: reviews e livros prontos unidos por `book_id`;
- `generos_livros.csv`: distribuição dos gêneros atribuídos;
- `relatorio_preparacao.md`: cobertura e decisões de limpeza.
"""
    (PASTA_SAIDA / "relatorio_preparacao.md").write_text(relatorio, encoding="utf-8")
    generos.value_counts("genres_assigned").rename("livros").rename_axis(
        "genres_assigned"
    ).reset_index().to_csv(
        PASTA_SAIDA / "generos_livros.csv", index=False, encoding="utf-8"
    )


def main() -> None:
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    reviews = preparar_reviews(pd.read_parquet(ARQUIVO_REVIEWS))
    books = preparar_books(pd.read_parquet(ARQUIVO_BOOKS))
    book_ids = set(reviews["book_id"])
    books = books[books["book_id"].isin(book_ids)].copy()
    base = reviews.merge(books, on="book_id", how="inner", suffixes=("_review", "_book"))
    reviews.to_parquet(PASTA_SAIDA / "reviews_prontas.parquet", index=False)
    books.to_parquet(PASTA_SAIDA / "livros_prontos.parquet", index=False)
    base.to_parquet(PASTA_SAIDA / "base_analise.parquet", index=False)
    salvar_relatorio(reviews, books, base)
    print(f"Dados preparados: {PASTA_SAIDA}")
    print(f"Reviews: {len(reviews):,}")
    print(f"Livros: {len(books):,}")


if __name__ == "__main__":
    main()
