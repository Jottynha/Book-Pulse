#!/usr/bin/env python3
"""Gera uma EDA inicial da amostra validada de reviews e livros."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PASTA_PROCESSED = BASE_DIR / "processed"
ARQUIVO_BOOKS = PASTA_PROCESSED / "goodreads_books_validated_100k.parquet"
ARQUIVO_REVIEWS = PASTA_PROCESSED / "goodreads_reviews_validated_100k.parquet"
PASTA_SAIDA = PASTA_PROCESSED / "eda_amostra"
ARQUIVOS_LEGADOS = {
    "resumo_geral.csv",
    "distribuicao_notas.csv",
    "top_usuarios.csv",
    "top_livros.csv",
    "top_shelves.csv",
    "shelves_por_tipo.csv",
    "top_generos_proxy.csv",
    "top_generos_proxy_por_reviews.csv",
    "05_idiomas.png",
}

STATUS_CONFIAVEIS = ["MUITO_ALTO", "ALTO", "MEDIO"]
SHELVES_STATUS = {
    "to-read",
    "currently-reading",
    "owned",
    "books-i-own",
    "owned-books",
    "to-buy",
    "wish-list",
    "dnf",
    "did-not-finish",
    "maybe",
    "abandoned",
    "borrowed",
    "reviewed",
    "tbr",
    "have",
    "i-own",
    "own-it",
    "my-library",
    "my-books",
    "library",
    "library-books",
    "read",
    "re-read",
    "read-in-2012",
    "read-in-2013",
    "read-in-2014",
    "read-in-2015",
    "read-in-2016",
    "read-in-2017",
    "read-in-2018",
    "read-in-2019",
    "read-in-2020",
    "read-in-2021",
    "read-in-2022",
    "read-in-2023",
    "read-in-2024",
    "read-in-2025",
    "read-in-2026",
    "read-2012",
    "read-2013",
    "read-2014",
    "read-2015",
    "read-2016",
    "read-2017",
}
SHELVES_FORMATO = {
    "ebook",
    "ebooks",
    "kindle",
    "audiobook",
    "audiobooks",
    "audio",
    "audio-books",
    "audio-book",
    "audible",
    "e-book",
    "e-books",
}
SHELVES_OUTROS = {
    "favorites",
    "favourites",
    "favorite",
    "favorite-books",
    "default",
    "books",
    "books-about-books",
    "series",
    "book-club",
    "5-stars",
    "4-stars",
    "3-stars",
    "2-stars",
    "1-star",
    "english",
    "arc",
}


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera tabelas, relatório e gráficos da EDA da amostra."
    )
    parser.add_argument(
        "--only-confident",
        action="store_true",
        help="Usa somente reviews com match_status MUITO_ALTO, ALTO ou MEDIO.",
    )
    return parser.parse_args()


def numero(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie, errors="coerce")


def proporcao(valor: int, total: int) -> float:
    return round(valor / total * 100, 2) if total else 0.0


def preparar_reviews(reviews: pd.DataFrame, only_confident: bool) -> pd.DataFrame:
    reviews = reviews.copy()
    if only_confident:
        reviews = reviews[reviews["match_status"].isin(STATUS_CONFIAVEIS)].copy()

    reviews["rating"] = numero(reviews["rating"])
    reviews["rating_valido"] = reviews["rating"].where(reviews["rating"].between(1, 5))
    reviews["n_votes"] = numero(reviews["n_votes"]).fillna(0)
    reviews["n_comments"] = numero(reviews["n_comments"]).fillna(0)
    reviews["review_text"] = reviews["review_text"].fillna("").astype(str)
    reviews["review_chars"] = reviews["review_text"].str.len()
    reviews["review_words"] = reviews["review_text"].str.split().str.len()
    reviews["date_added"] = pd.to_datetime(
        reviews["date_added"], errors="coerce", format="mixed", utc=True
    )
    reviews["review_year"] = reviews["date_added"].dt.year
    return reviews


def preparar_books(books: pd.DataFrame) -> pd.DataFrame:
    books = books.copy()
    for column in [
        "publication_year",
        "num_pages",
        "average_rating",
        "ratings_count",
        "text_reviews_count",
    ]:
        books[column] = numero(books[column])
    books["is_ebook"] = books["is_ebook"].astype(str).str.lower().eq("true")
    return books


def salvar_csv(dataframe: pd.DataFrame, nome: str) -> None:
    dataframe.to_csv(PASTA_SAIDA / nome, index=False, encoding="utf-8")


def limpar_saidas_legadas() -> None:
    for nome in ARQUIVOS_LEGADOS:
        caminho = PASTA_SAIDA / nome
        if caminho.exists():
            caminho.unlink()


def classificar_shelf(shelf: str) -> str:
    shelf_normalizada = shelf.lower()
    if (
        shelf_normalizada in SHELVES_STATUS
        or shelf_normalizada.startswith("read-in-")
        or shelf_normalizada.startswith("read-")
        or "to-read" in shelf_normalizada
        or "owned" in shelf_normalizada
    ):
        return "status_leitura"
    if (
        shelf_normalizada in SHELVES_FORMATO
        or "kindle" in shelf_normalizada
        or "ebook" in shelf_normalizada
        or "e-book" in shelf_normalizada
        or "audio" in shelf_normalizada
        or "audible" in shelf_normalizada
    ):
        return "formato"
    if (
        shelf_normalizada in SHELVES_OUTROS
        or "favorite" in shelf_normalizada
        or "favourite" in shelf_normalizada
        or "-stars" in shelf_normalizada
    ):
        return "outro"
    return "genre_proxy"


def contar_shelves(books: pd.DataFrame) -> pd.DataFrame:
    shelves = (
        books[["book_id", "popular_shelves"]]
        .assign(popular_shelves=lambda df: df["popular_shelves"].fillna(""))
        .assign(popular_shelves=lambda df: df["popular_shelves"].str.split("|"))
        .explode("popular_shelves")
        .rename(columns={"popular_shelves": "shelf"})
    )
    shelves["shelf"] = shelves["shelf"].fillna("").astype(str).str.strip()
    shelves = shelves[shelves["shelf"].ne("")]
    shelves = shelves.drop_duplicates(["book_id", "shelf"])
    shelves["shelf_type"] = shelves["shelf"].map(classificar_shelf)
    shelves["is_genre_proxy"] = shelves["shelf_type"].eq("genre_proxy")
    return shelves


def escrever_grafico(figura: plt.Figure, nome: str) -> None:
    figura.tight_layout()
    figura.savefig(PASTA_SAIDA / nome, dpi=150, bbox_inches="tight")
    plt.close(figura)


def gerar_graficos(reviews: pd.DataFrame) -> None:
    figura, eixo = plt.subplots(figsize=(8, 5))
    reviews["rating"].value_counts().sort_index().plot.bar(ax=eixo, color="#2f6f8f")
    eixo.set_title("Distribuição das notas")
    eixo.set_xlabel("Nota")
    eixo.set_ylabel("Número de reviews")
    escrever_grafico(figura, "01_distribuicao_notas.png")

    figura, eixos = plt.subplots(1, 2, figsize=(12, 5))
    reviews["review_chars"].clip(upper=reviews["review_chars"].quantile(0.99)).plot.hist(
        bins=40, ax=eixos[0], color="#d97941"
    )
    eixos[0].set_title("Tamanho das reviews (até P99)")
    eixos[0].set_xlabel("Caracteres")
    eixos[0].set_ylabel("Número de reviews")
    reviews["n_votes"].clip(upper=reviews["n_votes"].quantile(0.99)).plot.hist(
        bins=30, ax=eixos[1], color="#597a55"
    )
    eixos[1].set_title("Votos de utilidade (até P99)")
    eixos[1].set_xlabel("Votos")
    eixos[1].set_ylabel("Número de reviews")
    escrever_grafico(figura, "02_texto_e_votos.png")

    top_books = reviews["book_id"].value_counts().head(15).sort_values()
    figura, eixo = plt.subplots(figsize=(9, 6))
    rotulos = top_books.index.astype(str)
    top_books.index = rotulos
    top_books.plot.barh(ax=eixo, color="#6d597a")
    eixo.set_title("Livros com mais reviews na amostra")
    eixo.set_xlabel("Número de reviews")
    eixo.set_ylabel("book_id")
    escrever_grafico(figura, "03_livros_mais_resenhados.png")

    rating_books = (
        reviews.groupby("book_id", as_index=False)
        .agg(review_count=("review_id", "count"), mean_rating=("rating", "mean"))
        .query("review_count >= 2")
    )
    figura, eixo = plt.subplots(figsize=(8, 6))
    eixo.scatter(
        rating_books["review_count"],
        rating_books["mean_rating"],
        alpha=0.45,
        s=18,
        color="#355070",
    )
    eixo.set_xscale("log")
    eixo.set_title("Volume de reviews e nota média por livro")
    eixo.set_xlabel("Reviews por livro (escala log)")
    eixo.set_ylabel("Nota média")
    escrever_grafico(figura, "04_volume_e_nota_por_livro.png")

def preparar_base_consolidada(
    reviews: pd.DataFrame,
    books: pd.DataFrame,
    shelves: pd.DataFrame,
):
    genres = (
        shelves[shelves["is_genre_proxy"]]
        .groupby("book_id")["shelf"]
        .agg(lambda values: "|".join(sorted(set(values))))
        .rename("genres_proxy")
    )
    book_columns = [
        "book_id", "title", "author_ids", "publication_year", "num_pages",
        "language_code", "is_ebook", "average_rating", "ratings_count",
        "text_reviews_count",
    ]
    base = reviews.merge(books[book_columns], on="book_id", how="left")
    base = base.merge(genres, on="book_id", how="left")
    base["genres_proxy"] = base["genres_proxy"].fillna("")
    return base


def gerar_tabelas_consolidadas(
    base: pd.DataFrame,
    shelves: pd.DataFrame,
) -> None:
    base_columns = [
        "review_id", "user_id", "book_id", "rating", "rating_valido", "review_chars",
        "review_words", "n_votes", "n_comments", "review_year",
        "match_status", "title", "author_ids", "publication_year",
        "num_pages", "language_code", "is_ebook", "average_rating",
        "ratings_count", "text_reviews_count", "genres_proxy",
    ]
    base[base_columns].to_parquet(
        PASTA_SAIDA / "base_eda_consolidada.parquet", index=False
    )

    perfil = pd.DataFrame([
        ("reviews", len(base), "registros"),
        ("livros_unicos", base["book_id"].nunique(), "livros"),
        ("usuarios_unicos", base["user_id"].nunique(), "usuarios"),
        ("nota_media_valida", round(base["rating_valido"].mean(), 4), "escala 1 a 5"),
        ("percentual_rating_valido", round(base["rating_valido"].notna().mean() * 100, 2), "%"),
        ("mediana_caracteres_review", base["review_chars"].median(), "caracteres"),
        ("media_votos", round(base["n_votes"].mean(), 4), "votos por review"),
        ("media_comentarios", round(base["n_comments"].mean(), 4), "comentarios por review"),
        ("percentual_rating_zero", round(base["rating"].eq(0).mean() * 100, 2), "%"),
        ("percentual_com_ano", round(base["publication_year"].notna().mean() * 100, 2), "% dos reviews"),
        ("percentual_com_paginas", round(base["num_pages"].notna().mean() * 100, 2), "% dos reviews"),
    ], columns=["indicador", "valor", "unidade"])
    salvar_csv(perfil, "perfil_eda.csv")

    livros = base.groupby("book_id", as_index=False).agg(
        title=("title", "first"),
        reviews=("review_id", "count"),
        usuarios=("user_id", "nunique"),
        notas_validas=("rating_valido", "count"),
        nota_media=("rating_valido", "mean"),
        nota_mediana=("rating_valido", "median"),
        caracteres_medianos=("review_chars", "median"),
        votos_medios=("n_votes", "mean"),
        comentarios_medios=("n_comments", "mean"),
        publication_year=("publication_year", "first"),
        num_pages=("num_pages", "first"),
        language_code=("language_code", "first"),
        is_ebook=("is_ebook", "first"),
        average_rating_goodreads=("average_rating", "first"),
        ratings_count=("ratings_count", "first"),
        genres_proxy=("genres_proxy", "first"),
    )
    salvar_csv(livros.sort_values(["reviews", "nota_media"], ascending=False), "analise_livros.csv")

    usuarios = base.groupby("user_id", as_index=False).agg(
        reviews=("review_id", "count"),
        livros=("book_id", "nunique"),
        notas_validas=("rating_valido", "count"),
        nota_media=("rating_valido", "mean"),
        caracteres_medianos=("review_chars", "median"),
        votos_medios=("n_votes", "mean"),
        comentarios_medios=("n_comments", "mean"),
    )
    salvar_csv(usuarios.sort_values("reviews", ascending=False), "analise_usuarios.csv")

    genre_books = shelves[shelves["is_genre_proxy"]][["book_id", "shelf"]].drop_duplicates()
    generos = genre_books.merge(
        base[["book_id", "review_id", "rating_valido"]], on="book_id", how="inner"
    ).groupby("shelf", as_index=False).agg(
        livros=("book_id", "nunique"),
        reviews=("review_id", "nunique"),
        notas_validas=("rating_valido", "count"),
        nota_media=("rating_valido", "mean"),
        nota_mediana=("rating_valido", "median"),
    )
    salvar_csv(generos.sort_values("reviews", ascending=False), "analise_generos.csv")


def gerar_relatorio(
    base: pd.DataFrame,
    shelves: pd.DataFrame,
    only_confident: bool,
) -> None:
    total_reviews = len(base)
    genre_shelves = shelves[shelves["is_genre_proxy"]]
    shelf_type_counts = shelves["shelf_type"].value_counts()

    linhas = [
        "# EDA inicial da amostra",
        "",
        f"Filtro aplicado: {'somente matches confiáveis' if only_confident else 'todas as reviews da amostra' }.",
        "",
        "## Perfil da base",
        "",
        f"- Reviews analisadas: {total_reviews:,}",
        f"- Livros únicos analisados: {base['book_id'].nunique():,}",
        f"- Usuários únicos: {base['user_id'].nunique():,}",
        "As tabelas consolidadas são a fonte principal para as análises seguintes.",
        "",
        "## Notas e comportamento",
        "",
        f"- Nota média válida (1 a 5): {base['rating_valido'].mean():.2f}",
        f"- Mediana da nota válida: {base['rating_valido'].median():.1f}",
        f"- Percentual sem nota (`rating == 0`): {proporcao(int(base['rating'].eq(0).sum()), total_reviews):.2f}%",
        f"- Mediana de caracteres por review: {base['review_chars'].median():.0f}",
        f"- Média de votos: {base['n_votes'].mean():.2f}",
        f"- Média de comentários: {base['n_comments'].mean():.2f}",
        "",
        "## Gêneros e shelves",
        "",
        "As `popular_shelves` são rótulos sociais e podem se sobrepor. Nesta EDA,",
        "cada shelf é classificada como `genre_proxy`, `status_leitura`, `formato`",
        "ou `outro`. Apenas `genre_proxy` é usado na tabela de gêneros; a classificação",
        "é exploratória e não equivale a uma taxonomia editorial definitiva.",
        "",
        f"- Associações livro-shelf classificadas como proxy de gênero: {int(shelf_type_counts.get('genre_proxy', 0)):,}",
        f"- Associações livro-shelf classificadas como status de leitura: {int(shelf_type_counts.get('status_leitura', 0)):,}",
        f"- Associações livro-shelf classificadas como formato: {int(shelf_type_counts.get('formato', 0)):,}",
        "",
        "## Artefatos",
        "",
        "- `base_eda_consolidada.parquet`: uma linha por review com variáveis do livro.",
        "- `perfil_eda.csv`: indicadores essenciais da amostra.",
        "- `analise_livros.csv`: uma linha por livro, com métricas agregadas.",
        "- `analise_usuarios.csv`: uma linha por usuário, com métricas agregadas.",
        "- `analise_generos.csv`: gêneros proxy com livros, reviews e notas.",
        "- `01_` a `04_*.png`: gráficos essenciais.",
        "",
        "## Perguntas para a próxima análise",
        "",
        "- A nota está associada ao tamanho da review ou aos votos recebidos?",
        "- A distribuição de notas muda entre livros com diferentes volumes de reviews?",
        "- Usuários mais ativos avaliam de forma diferente dos usuários com uma única review?",
        "- Quais gêneros/shelves concentram mais reviews e quais têm maior nota média?",
    ]
    (PASTA_SAIDA / "relatorio_eda.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def main() -> None:
    global PASTA_SAIDA

    args = argumentos()
    if not ARQUIVO_BOOKS.exists() or not ARQUIVO_REVIEWS.exists():
        raise FileNotFoundError(
            "Execute as etapas 01 a 05 antes da EDA para gerar os Parquets validados."
        )
    nome_saida = "eda_amostra_confiante" if args.only_confident else "eda_amostra"
    PASTA_SAIDA = PASTA_PROCESSED / nome_saida
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    limpar_saidas_legadas()
    books = preparar_books(pd.read_parquet(ARQUIVO_BOOKS))
    reviews = preparar_reviews(pd.read_parquet(ARQUIVO_REVIEWS), args.only_confident)
    shelves = contar_shelves(books)
    base = preparar_base_consolidada(reviews, books, shelves)
    gerar_tabelas_consolidadas(base, shelves)
    gerar_relatorio(base, shelves, args.only_confident)
    gerar_graficos(base)
    print(f"EDA concluída: {PASTA_SAIDA}")
    print(f"Reviews analisadas: {len(reviews):,}")
    print(f"Livros analisados: {reviews['book_id'].nunique():,}")


if __name__ == "__main__":
    main()
