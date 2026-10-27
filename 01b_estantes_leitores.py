#!/usr/bin/env python3
"""Extrai as estantes dos leitores, com contagens, para os livros da amostra.

O passo 1 guardou apenas os nomes das `popular_shelves`. Aqui o arquivo
original é lido de novo para recuperar quantos leitores colocaram cada livro em
cada estante. Essa contagem é o voto dos leitores usado pelo passo 5 para
atribuir gêneros.

Arquivo de entrada: goodreads_books.json.gz (UCSD Book Graph), informado por
ARQUIVO_BOOKS ou --books.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
AMOSTRA_BOOKS = BASE_DIR / "processed" / "01_amostra" / "goodreads_books_100k.parquet"
SAIDA = BASE_DIR / "processed" / "01_amostra" / "estantes_leitores.parquet"
BOOK_ID_RE = re.compile(r'"book_id":\s*"(\d+)"')


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--books",
        default=os.getenv(
            "ARQUIVO_BOOKS",
            "/media/joao/8cb5cf8c-261d-4ef9-965c-485a78e63f21/kaggle/goodreads_books.json.gz",
        ),
    )
    parser.add_argument("--output", type=Path, default=SAIDA)
    return parser.parse_args()


def main() -> None:
    args = argumentos()
    alvo = set(pd.read_parquet(AMOSTRA_BOOKS, columns=["book_id"])["book_id"].astype(str))
    linhas: list[dict[str, object]] = []
    encontrados: set[str] = set()
    with gzip.open(args.books, "rt", encoding="utf-8") as arquivo:
        for linha in arquivo:
            # Filtra pelo id antes de decodificar o JSON inteiro.
            achado = BOOK_ID_RE.search(linha)
            if not achado or achado.group(1) not in alvo:
                continue
            livro = json.loads(linha)
            book_id = str(livro["book_id"])
            encontrados.add(book_id)
            for estante in livro.get("popular_shelves") or []:
                nome = str(estante.get("name") or "").strip()
                contagem = pd.to_numeric(estante.get("count"), errors="coerce")
                if nome and pd.notna(contagem):
                    linhas.append({"book_id": book_id, "shelf": nome, "count": int(contagem)})
            if len(encontrados) == len(alvo):
                break
    estantes = pd.DataFrame(linhas, columns=["book_id", "shelf", "count"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    estantes.to_parquet(args.output, index=False)
    print(f"Estantes extraídas: {args.output}")
    print(f"Livros encontrados: {len(encontrados):,} de {len(alvo):,}; linhas: {len(estantes):,}")


if __name__ == "__main__":
    main()
