# Passo 6 — Refinamento textual e análise inicial

## Escopo

Esta etapa usa `base_analise.parquet`, que já contém somente matches confiáveis.
O texto original (`review_text`) é preservado. A coluna `review_text_refined`
remove HTML, URLs, caracteres de controle, diferenças de caixa e pontuação,
mantendo apenas tokens para facilitar as análises seguintes.

- Reviews processadas: 8,033
- Reviews com texto refinado: 7,983
- Reviews sem texto: 50
- Média de palavras: 122.2
- Usuários na amostra: 7,350

## Primeiros indicadores comportamentais

Os arquivos `resumo_por_nota.csv`, `resumo_por_genero.csv`,
`resumo_por_usuario.csv` e `correlacoes_com_rating.csv` permitem investigar:

- se reviews com notas diferentes têm extensões e padrões de linguagem distintos;
- se gênero está associado a nota, tamanho ou votos de utilidade;
- se usuários mais ativos na amostra escrevem reviews diferentes;
- a associação entre nota, votos, comentários, tamanho e popularidade do livro.

### Gêneros mais frequentes

- `fantasy`: 1,032 reviews, nota média 3.96
- `romance`: 935 reviews, nota média 3.95
- `science_fiction`: 700 reviews, nota média 3.85
- `children`: 696 reviews, nota média 3.93
- `mystery`: 659 reviews, nota média 3.84
- `young_adult`: 653 reviews, nota média 3.79
- `thriller`: 421 reviews, nota média 3.81
- `biography`: 417 reviews, nota média 3.84
- `nonfiction`: 398 reviews, nota média 3.85
- `literary_fiction`: 389 reviews, nota média 3.78

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
