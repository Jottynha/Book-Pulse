# EDA inicial da amostra

Filtro aplicado: somente matches confiáveis.

## Perfil da base

- Reviews analisadas: 8,033
- Livros únicos analisados: 7,031
- Usuários únicos: 7,350
As tabelas consolidadas são a fonte principal para as análises seguintes.

## Notas e comportamento

- Nota média válida (1 a 5): 3.87
- Mediana da nota válida: 4.0
- Percentual sem nota (`rating == 0`): 3.49%
- Mediana de caracteres por review: 334
- Média de votos: 1.10
- Média de comentários: 0.24

## Gêneros e shelves

As `popular_shelves` são rótulos sociais e podem se sobrepor. Nesta EDA,
cada shelf é classificada como `genre_proxy`, `status_leitura`, `formato`
ou `outro`. Apenas `genre_proxy` é usado na tabela de gêneros; a classificação
é exploratória e não equivale a uma taxonomia editorial definitiva.

- Associações livro-shelf classificadas como proxy de gênero: 538,835
- Associações livro-shelf classificadas como status de leitura: 148,726
- Associações livro-shelf classificadas como formato: 61,766

## Artefatos

- `base_eda_consolidada.parquet`: uma linha por review com variáveis do livro.
- `perfil_eda.csv`: indicadores essenciais da amostra.
- `analise_livros.csv`: uma linha por livro, com métricas agregadas.
- `analise_usuarios.csv`: uma linha por usuário, com métricas agregadas.
- `analise_generos.csv`: gêneros proxy com livros, reviews e notas.
- `01_` a `04_*.png`: gráficos essenciais.

## Perguntas para a próxima análise

- A nota está associada ao tamanho da review ou aos votos recebidos?
- A distribuição de notas muda entre livros com diferentes volumes de reviews?
- Usuários mais ativos avaliam de forma diferente dos usuários com uma única review?
- Quais gêneros/shelves concentram mais reviews e quais têm maior nota média?
