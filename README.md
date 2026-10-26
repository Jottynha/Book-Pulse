# Book Review Data Analysis

Projeto de Ciência de Dados (CEFET-MG) para analisar o comportamento de leitores
a partir de uma amostra de resenhas do Goodreads, enriquecida com metadados da
Open Library e do Google Books.

## Objetivo e escopo

O objetivo é construir uma base confiável para análise exploratória: preservar as
informações comportamentais do Goodreads, acrescentar características dos livros
e documentar a qualidade dos cruzamentos. O resultado não representa todos os
reviews do dataset original e não garante que todo match externo esteja correto.

O dataset original tem aproximadamente 15 milhões de resenhas. A amostra de
10.000 resenhas representa cerca de 0,067% desse volume. Esse tamanho permite
executar o fluxo completo em ambiente local, reduz custo e chamadas às APIs,
produz uma base manejável para EDA e torna o processamento reproduzível.

A seleção usa reservoir sampling em uma única passagem pelo arquivo compactado,
com `seed=42`. Assim, cada registro tem a mesma oportunidade de entrar na
amostra sem carregar os aproximadamente 15 milhões de registros na memória. A
amostra é adequada para exploração e validação do pipeline; conclusões sobre a
população inteira devem declarar essa limitação.

## Estado atual e encerramento do cruzamento

O cruzamento foi concluído para a amostra definida:

| Indicador | Resultado |
| --- | ---: |
| Reviews Goodreads | 10.000 |
| Livros únicos | 8.916 |
| Livros com match Open Library | 7.654 (85,85%) |
| Livros com match Google Books | 4.424 (49,62%) |
| Livros com ambas as APIs | 3.852 |
| Livros com confiança `MUITO_ALTO`, `ALTO` ou `MEDIO` | 7.031 (78,86%) |
| Reviews com confiança adequada | 8.033 |
| Reviews com ambas as APIs e confiança adequada | 4.293 |

Reviews são mais numerosos que livros porque um mesmo livro pode receber várias
resenhas. A diferença de cobertura entre as APIs não indica, sozinha, que uma
fonte seja melhor: os catálogos, identificadores, edições e políticas de
preenchimento são diferentes.

Esta fase de cruzamento está fechada. A análise principal seguirá somente com
reviews cujo `match_status` seja `MUITO_ALTO`, `ALTO` ou `MEDIO`. A base completa
será mantida para auditoria e análise de sensibilidade, mas não será usada como
referência dos resultados principais porque contém matches baixos, rejeitados e
livros sem match.

```python
import pandas as pd

df = pd.read_parquet(
    "processed/04_validacao/goodreads_reviews_validated_100k.parquet"
)
df_confiavel = df[
    df["match_status"].isin(["MUITO_ALTO", "ALTO", "MEDIO"])
].copy()
```

Para comparar as duas APIs, além do filtro de confiança, exigir
`openlibrary_match == True` e `google_match == True`.

## Justificativa do enriquecimento

O Goodreads é a fonte principal das resenhas e avaliações, mas seus metadados
não são suficientes para todas as análises. O enriquecimento foi feito no nível
de livro, usando os `book_id` encontrados na amostra, para evitar consultas de
livros que não participam da análise e manter o vínculo com cada review.

- **Open Library:** prioriza ISBN-13 e ISBN-10 em lote; sem resultado, tenta
  título + autor e depois título.
- **Google Books:** funciona como fonte independente de conferência e de
  atributos editoriais; prioriza ISBN e encerra a busca quando encontra um
  resultado adequado.

As consultas usam cache persistente, retentativas para erros de rede e rate limit,
salvamento incremental e retomada. A ausência de match significa apenas que a
fonte ou a estratégia não retornou um registro utilizável.

## Validação dos matches

O validador compara, quando disponíveis:

- ISBN-13 e ISBN-10, com maior peso para os identificadores;
- similaridade de título após normalização de acentos e pontuação;
- interseção de autores;
- ano de publicação, com tolerância de até três anos para diferenças de edição;
- concordância entre Open Library e Google Books.

Cada livro recebe `match_status` em `MUITO_ALTO`, `ALTO`, `MEDIO`, `BAIXO`,
`REJEITAR` ou `SEM_MATCH`. Os casos fracos ou conflitantes ficam em
`processed/04_validacao/matches_para_revisao.csv`.

A fila de revisão manual inclui agora todos os livros `BAIXO`, `REJEITAR` e
`SEM_MATCH`, totalizando 1.885 registros na execução atual:

| Situação | Quantidade |
| --- | ---: |
| `SEM_MATCH` nas duas fontes | 690 |
| `REJEITAR` | 1.067 |
| `BAIXO` | 128 |

Cada registro recebe `review_priority`, `review_reason`, presença em cada fonte,
indicadores de ISBN, autores, anos e conflito entre títulos. A prioridade 1 é
reservada para ausência total de match; prioridades maiores indicam conflito ou
evidência fraca. Nenhum match é promovido automaticamente durante essa revisão:
uma eventual correção manual deve ser uma decisão registrada e reproduzível.

## Pipeline

```text
01_amostra_reviews.py
    -> processed/01_amostra/
02_cruzar_openlibrary.py
    -> processed/02_openlibrary/
03_cruzar_google_books.py
    -> processed/03_google_books/
04_validar_matches.py
    -> processed/04_validacao/
    -> relatório de qualidade e fila de revisão
05_preparar_dados.py
    -> processed/05_dados_prontos/
06_analisar_reviews.py
    -> processed/06_analise/
```

As etapas externas trabalham somente com os livros únicos da amostra. A análise
principal usa exclusivamente o resultado validado e os matches confiáveis.

> **Nota:** o sufixo `100k` é histórico e foi mantido para não quebrar o pipeline.
> A execução documentada contém 10.000 reviews, não 100.000.

## Preparação final dos dados

O script `05_preparar_dados.py` é a etapa final antes da análise. Ele usa somente
matches `MUITO_ALTO`, `ALTO` e `MEDIO` e produz dados limpos, sem duplicatas de
reviews/livros, com tipos numéricos e datas normalizadas, texto higienizado e
variáveis de tamanho da resenha.

```bash
python3 05_preparar_dados.py
```

As `popular_shelves`, assuntos da Open Library e categorias do Google Books são
combinados para atribuir no máximo três gêneros específicos por livro. A saída
preserva `genre_evidence`, isto é, os rótulos que sustentaram a atribuição. Livros
sem evidência suficiente ficam como `unclassified`, sem gênero inventado.

Na execução atual, 6.971 dos 7.031 livros receberam ao menos um gênero, 6.183
receberam três gêneros e 60 ficaram sem evidência específica. A ordem dos gêneros
é determinada pela força e pela quantidade de evidências, com assuntos/categorias
das fontes externas pesando mais que shelves sociais do Goodreads. Essas
categorias são variáveis analíticas controladas, não uma taxonomia editorial
definitiva.

Ratings zero são preservados como ausência de avaliação e não entram em métricas
que usem `rating_valido`. Reviews em outros idiomas ou com caracteres não ASCII
continuam na base, mas recebem indicadores para tratamento específico na análise.

Arquivos finais:

- `reviews_prontas.parquet`: reviews limpas e confiáveis;
- `livros_prontos.parquet`: livros associados à amostra confiável;
- `base_analise.parquet`: reviews e livros unidos por `book_id`;
- `generos_livros.csv`: gêneros atribuídos e frequência;
- `relatorio_preparacao.md`: cobertura e decisões de limpeza.

## Passo 6: refinamento textual e análise inicial

O script `06_analisar_reviews.py` inicia a análise comportamental. Ele preserva
`review_text` e cria uma versão refinada (`review_text_refined`) sem HTML, URLs,
caracteres de controle, pontuação e diferenças de caixa. Também calcula
quantidade de tokens, diversidade lexical, frases e um score de sentimento
lexical transparente, baseado em um vocabulário pequeno em inglês.

O score de sentimento é somente uma linha de base: não resolve negação, ironia
ou contexto. Ele serve para orientar a exploração e deve ser substituído ou
comparado com uma abordagem de NLP mais robusta antes de conclusões finais.

```bash
python3 06_analisar_reviews.py
```

Saídas em `processed/06_analise/`:

- `reviews_refinadas.parquet`: reviews com variáveis textuais;
- `resumo_por_nota.csv`: extensão, sentimento, votos e comentários por nota;
- `resumo_por_genero.csv`: indicadores por gênero principal;
- `resumo_por_usuario.csv`: atividade agregada por usuário;
- `correlacoes_com_rating.csv`: correlações exploratórias;
- `visao_geral_reviews.png` e `relatorio_analise.md`.

## Arquivos principais

```text
processed/
├── 01_amostra/
├── 02_openlibrary/
├── 03_google_books/
├── 04_validacao/
├── 05_dados_prontos/
└── 06_analise/
```

## Execução completa

```bash
python3 -m pip install -r requirements.txt
python3 01_amostra_reviews.py
python3 02_cruzar_openlibrary.py
python3 03_cruzar_google_books.py
python3 04_validar_matches.py
python3 05_preparar_dados.py
python3 06_analisar_reviews.py
```

Para o Google Books, configure `GOOGLE_BOOKS_API_KEY` no ambiente ou em `.env`.
As etapas externas reaproveitam os caches persistidos em `processed/`.

## Próximas análises

- Relação entre nota e sentimento do texto da resenha.
- Efeito de tamanho, gênero e popularidade do livro sobre a nota.
- Relação entre extensão da resenha, votos de utilidade e comentários.
- Comparação entre usuários ocasionais e usuários mais ativos.
