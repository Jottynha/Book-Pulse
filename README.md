# BookPulse — análise comportamental de resenhas

Projeto de Ciência de Dados (CEFET-MG) para estudar o comportamento de leitores
a partir de resenhas do Goodreads, enriquecidas com metadados da Open Library e
do Google Books. O projeto combina preparação de dados, análise textual,
sentimento exploratório e uma interface de recomendação baseada em padrões
agregados das resenhas.

## 1. Pergunta e limites

O objetivo é investigar como notas, texto, gênero, popularidade, votos,
comentários e atividade dos usuários se relacionam nas resenhas. O dataset
original tem aproximadamente 15 milhões de resenhas; esta execução usa uma
amostra reprodutível de 10.000 reviews, selecionada por reservoir sampling com
`seed=42`. Portanto, os resultados descrevem a amostra e não devem ser
generalizados automaticamente para todos os usuários do Goodreads.

É importante separar três conceitos:

- **sentimento textual:** polaridade expressa no texto;
- **nota Goodreads:** avaliação numérica dada pelo usuário;
- **predição de nota:** tarefa diferente, que estima a nota a partir de dados.

Uma nota alta não comprova que todo o texto é positivo. As correlações deste
projeto são associações observacionais; não demonstram causalidade.

## 2. Cobertura dos dados confiáveis

O fluxo externo foi encerrado antes da análise principal. Apenas livros com
`match_status` `MUITO_ALTO`, `ALTO` ou `MEDIO` entram nos resultados principais.

| Indicador | Quantidade |
| --- | ---: |
| Reviews na amostra original | 10.000 |
| Livros únicos na amostra | 8.916 |
| Matches Open Library | 7.654 (85,85%) |
| Matches Google Books | 4.424 (49,62%) |
| Livros com ambas as fontes | 3.852 |
| Livros com confiança adequada | 7.031 |
| Reviews com confiança adequada | 8.033 |
| Reviews confiáveis com ambas as APIs | 4.293 |

Open Library e Google Books têm coberturas diferentes por causa de edições,
identificadores e políticas de catálogo. Ausência de match não significa que o
livro seja inválido. Os registros fracos continuam preservados em
`processed/04_validacao/` para auditoria.

## 3. Pipeline reproduzível

```text
01_amostra_reviews.py
  -> processed/01_amostra/
02_cruzar_openlibrary.py
  -> processed/02_openlibrary/
03_cruzar_google_books.py
  -> processed/03_google_books/
04_validar_matches.py
  -> processed/04_validacao/
05_preparar_dados.py
  -> processed/05_dados_prontos/
06_analisar_reviews.py
  -> processed/06_analise/
07_analise_sentimento.py
  -> processed/07_sentimento/
08_gerar_site.py
  -> site/data/books.json
```

O sufixo `100k` dos arquivos é histórico; a execução documentada contém 10.000
reviews na amostra inicial.

## 4. Preparação e enriquecimento

O script `05_preparar_dados.py` filtra os matches confiáveis, remove duplicatas,
normaliza datas e números, preserva o texto original e cria indicadores de
tamanho. Ratings zero são mantidos como ausência de avaliação e não entram em
métricas que exigem `rating_valido`.

Os gêneros são inferidos combinando `popular_shelves` do Goodreads, assuntos da
Open Library e categorias do Google Books. Fontes externas recebem peso maior,
são atribuídos no máximo três gêneros e `genre_evidence` registra os rótulos que
deram suporte à decisão. Na execução atual:

- 6.971 dos 7.031 livros têm ao menos um gênero;
- 6.183 têm três gêneros;
- 60 ficam como `unclassified`, sem gênero inventado.

Saídas principais:

- `reviews_prontas.parquet`: reviews limpas e confiáveis;
- `livros_prontos.parquet`: livros normalizados;
- `base_analise.parquet`: união review-livro por `book_id`;
- `generos_livros.csv`: frequência de gêneros.

## 5. Análise exploratória textual

`06_analisar_reviews.py` preserva `review_text` e cria uma representação para
EDA com tokens, contagem de palavras e frases, diversidade lexical e score
lexical. O baseline usa um vocabulário explícito em inglês e uma regra simples
para negadores nas três palavras anteriores. O score também é normalizado por
100 palavras.

Na execução atual, foram processadas 8.033 reviews, das quais 7.983 têm texto
refinado e 50 não têm tokens utilizáveis. O baseline é interpretável, mas não
resolve ironia, sarcasmo, escopo complexo, contexto literário ou sentimentos
mistos. Ele deve orientar a exploração, não ser apresentado como sentimento
validado.

Os resumos incluem análises por nota, gênero e usuário, além de correlações
Pearson e Spearman com nota, tamanho, votos, comentários e popularidade. Essas
medidas não controlam confundidores como livro, usuário, gênero ou popularidade.

## 6. Análise de sentimento e validação

`07_analise_sentimento.py` cria uma etapa separada, sem substituir os resultados
anteriores. Ela combina:

1. baseline lexical do passo 6;
2. VADER, com `neg`, `neu`, `pos`, `compound` e limiares convencionais
   `-0,05` e `0,05`;
3. transformer opcional `siebert/sentiment-roberta-large-english`;
4. TF-IDF com unigramas e bigramas + regressão logística, condicionado à
   existência de anotações humanas.

Sem rótulos humanos, o script não calcula acurácia, precisão, recall ou F1. A
nota pode ser usada como proxy em uma análise complementar, mas não como
verdade de referência do sentimento.

O script gera uma amostra estratificada e sem a nota original para anotação
manual. As classes principais são `negative`, `neutral` e `positive`; `mixed` e
`unclear` podem ser registrados durante a anotação e devem ter tratamento
documentado antes da avaliação. Recomenda-se que dois anotadores classifiquem
parte da amostra independentemente. Com as colunas `annotator_a` e
`annotator_b`, o script calcula concordância observada e Cohen's kappa.

Com anotações preenchidas, o comando abaixo produz métricas em um split
estratificado separado:

```bash
py -3 07_analise_sentimento.py --annotations caminho\para\anotacoes.csv
```

O transformer não é baixado por padrão:

```bash
py -3 07_analise_sentimento.py --run-transformer
```

Antes de usar um checkpoint, deve-se conferir model card, idioma, domínio,
licença, limite de entrada e requisitos computacionais. Reviews longas são
divididas em trechos sobrepostos; a quantidade de trechos é registrada.

## 7. BookPulse: exploração e recomendação

O site em `site/` transforma os resultados agregados em uma interface de
exploração. Ele permite:

- buscar livros pelo título ou autor;
- carregar a capa por URL da Open Library, usando ISBN-13 ou ISBN-10;
- visualizar nota média, quantidade de reviews e extensão média;
- comparar proporções de resenhas positivas, neutras e negativas;
- consultar palavras recorrentes no conjunto de reviews;
- separar termos associados às resenhas positivas e negativas;
- explorar um grafo interativo ligando o livro às palavras mais frequentes e a
  outros títulos que compartilham esses termos;
- recomendar livros com gêneros compartilhados e perfil de sentimento próximo.

As recomendações são exploratórias: não representam uma avaliação editorial,
não usam o texto bruto das resenhas na página e não implicam que livros
semelhantes sejam necessariamente melhores. As capas priorizam o endpoint grande (`-L`) da Open Library e fazem fallback
automático para o tamanho médio (`-M`) ou para uma capa gerada quando não há
ISBN ou imagem disponível. No grafo, o livro selecionado fica no centro, as
palavras no primeiro anel e os títulos relacionados no anel externo. A
espessura das linhas indica a frequência do termo compartilhado.

Para gerar os dados do site e servir a interface:

```bash
py -3 08_gerar_site.py
py -3 -m http.server 8000 --directory site
```

Abra <http://localhost:8000>. O servidor local é necessário porque o navegador
bloqueia `fetch` de arquivos JSON quando a página é aberta diretamente por
`file://`.

## 8. Saídas e organização

Os parquets, CSVs e caches em `processed/` são artefatos do pipeline. A
documentação metodológica foi consolidada neste README para evitar relatórios
fragmentados e desatualizados. O site mantém apenas os dados agregados
necessários à visualização.

## 9. Execução completa

```bash
py -3 -m pip install -r requirements.txt
py -3 01_amostra_reviews.py
py -3 02_cruzar_openlibrary.py
py -3 03_cruzar_google_books.py
py -3 04_validar_matches.py
py -3 05_preparar_dados.py
py -3 06_analisar_reviews.py
py -3 07_analise_sentimento.py
py -3 08_gerar_site.py
```

O Google Books exige `GOOGLE_BOOKS_API_KEY` no ambiente ou em `.env`. As etapas
externas reaproveitam os caches persistidos em `processed/`.

## 10. Referências metodológicas

- Pang, Lee e Vaithyanathan (2002), classificação supervisionada de reviews:
  <https://doi.org/10.3115/1118693.1118704>
- Hutto e Gilbert (2014), VADER:
  <https://doi.org/10.1609/icwsm.v8i1.14550>
- Socher et al. (2013), composição semântica:
  <https://aclanthology.org/D13-1170/>
- Devlin et al. (2019), BERT:
  <https://doi.org/10.18653/v1/N19-1423>
- scikit-learn, validação cruzada e métricas:
  <https://scikit-learn.org/stable/modules/cross_validation.html> e
  <https://scikit-learn.org/stable/modules/model_evaluation.html>
- Hugging Face, classificação textual:
  <https://huggingface.co/docs/transformers/en/tasks/sequence_classification>
