# Dados preparados para análise

A etapa final usa somente matches `MUITO_ALTO`, `ALTO` e `MEDIO`.

## Cobertura

- Reviews prontas: 8,033
- Livros prontos: 7,031
- Reviews sem texto: 0
- Reviews com caracteres não ASCII: 0
- Livros com pelo menos um gênero atribuído: 6,971
- Livros sem gênero específico: 60
- Livros com 1 gênero: 252
- Livros com 2 gêneros: 536
- Livros com 3 gêneros: 6,183

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
