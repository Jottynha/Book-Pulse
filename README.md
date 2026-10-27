# BookPulse — análise comportamental de resenhas

Projeto de Ciência de Dados (CEFET-MG) para estudar o comportamento de leitores
a partir de resenhas do Goodreads, enriquecidas com metadados da Open Library e
do Google Books. O projeto combina preparação de dados, análise textual,
sentimento com validação, predição de nota e uma interface de recomendação
baseada em padrões agregados das resenhas.

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

A amostragem é feita por resenha, o desenho adequado para perguntas sobre
resenhas. Como consequência, 92% dos livros aparecem com uma única resenha: os
perfis por livro do site são ruidosos, embora não enviesados.

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
| Reviews confiáveis em inglês | 7.182 |
| Reviews confiáveis com ambas as APIs | 4.293 |

Open Library e Google Books têm coberturas diferentes por causa de edições,
identificadores e políticas de catálogo. Ausência de match não significa que o
livro seja inválido. Os registros fracos continuam preservados em
`processed/04_validacao/` para auditoria.

**Viés de seleção do filtro de match.** O filtro favorece livros populares e bem
catalogados, e descarta resenhas com nota ligeiramente maior:

| Indicador | Mantidas (8.033) | Descartadas (1.967) |
| --- | ---: | ---: |
| Nota média | 3,87 | 4,02 |
| Popularidade mediana do livro (ratings) | 2.938 | 359 |
| Palavras por resenha (mediana) | 60 | 70 |

Os resultados valem, portanto, para livros com metadados externos confiáveis,
com tendência a títulos mais populares.

## 3. Pipeline reproduzível

```text
01_amostra_reviews.py
  -> processed/01_amostra/
01b_estantes_leitores.py
  -> processed/01_amostra/estantes_leitores.parquet
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
  -> data/books.json
09_predicao_nota.py
  -> processed/09_predicao_nota/
```

O sufixo `100k` dos arquivos é histórico; a execução documentada contém 10.000
reviews na amostra inicial. Os relatórios Markdown gerados pelos scripts em
`processed/` não são versionados; a documentação fica neste README.

## 4. Preparação e gêneros pelo voto dos leitores

O script `05_preparar_dados.py` filtra os matches confiáveis, remove duplicatas,
normaliza datas e números, preserva o texto original e cria indicadores de
tamanho. Ratings zero são mantidos como ausência de avaliação e não entram em
métricas que exigem `rating_valido`.

Categorias de editora (BISAC, no Google Books) são definidas para o mercado, e
não pelo leitor. Por isso, os gêneros são atribuídos pelo **voto dos leitores**:
`01b_estantes_leitores.py` recupera do arquivo original do UCSD Book Graph
quantas pessoas colocaram cada livro em cada estante (`popular_shelves`), e o
passo 5 conta esses votos:

- o voto de um gênero é a **maior** contagem entre as estantes que o indicam,
  não a soma, para não contar duas vezes quem usa "sci-fi" e
  "science-fiction";
- um gênero entra com pelo menos 3 votos e 10% dos votos do gênero mais votado
  do livro, até três gêneros;
- `nonfiction` é tratado como categoria ampla: um gênero específico que passe
  no corte (biografia, história etc.) vem antes dele;
- empates, comuns em estantes compostas como "mystery-thriller", são decididos
  pelo catálogo;
- sem votos suficientes, vale o critério anterior por presença de rótulos
  (estantes, Open Library e Google Books), sinalizado em `genre_source`.

Na execução atual, 6.416 livros têm gêneros pelo voto dos leitores, 550 pelo
critério de rótulos e 65 ficam como `unclassified`, sem gênero inventado. A
classificação de catálogo é preservada em `catalog_genres`: **leitores e
catálogo concordam no gênero principal em apenas 49,7% dos livros**. A maior
divergência é `children` → `young_adult` (348 livros): a categoria editorial
"Juvenile Fiction" agrupa o que os leitores separam em infantil e juvenil.

Saídas principais:

- `reviews_prontas.parquet`: reviews limpas e confiáveis;
- `livros_prontos.parquet`: livros normalizados, com votos e evidências;
- `base_analise.parquet`: união review-livro por `book_id`;
- `generos_livros.csv`: frequência de gêneros.

## 5. Análise exploratória textual

`06_analisar_reviews.py` preserva `review_text` e cria uma representação para
EDA com tokens, contagem de palavras e frases, diversidade lexical e score
lexical. O baseline usa um vocabulário explícito em inglês e uma regra simples
para negadores nas três palavras anteriores. O score também é normalizado por
100 palavras.

Os resumos incluem análises por nota, gênero e usuário, além de correlações
Pearson e Spearman com nota, tamanho, votos, comentários e popularidade. Essas
medidas não controlam confundidores como livro, usuário, gênero ou popularidade.

## 6. Análise de sentimento e validação

`07_analise_sentimento.py` combina:

1. baseline lexical do passo 6;
2. VADER, com limiares convencionais `-0,05` e `0,05`;
3. transformer `siebert/sentiment-roberta-large-english` (Hartmann et al.,
   2023), binário, com acurácia publicada de 93,2% em 15 bases e 92,5% em
   resenhas de livros. Reviews longas são divididas em trechos sobrepostos e a
   probabilidade positiva é a média dos trechos, ponderada pelo tamanho;
4. TF-IDF + regressão logística, treinado quando há anotações humanas, com
   split agrupado por usuário e livro.

O idioma de cada review é detectado (`text_language`): os métodos são para
inglês, então 851 reviews em outras línguas ficam fora da amostra de anotação,
da predição de nota e do perfil de sentimento do site.

Nas 7.182 reviews em inglês, o transformer classifica 80,9% como positivas. Sua
probabilidade positiva tem correlação de Spearman ρ = 0,558 com a nota, contra
0,238 do VADER e 0,253 do léxico. A fração de reviews negativas cai de forma
monotônica com a nota: 95,9% (nota 1), 81,2% (2), 33,9% (3), 5,0% (4) e 2,0% (5).

**Anotação humana.** O script gera uma amostra estratificada de 601 reviews em
inglês, sem a nota original. A página `anotar.html` exibe cada review como um
cartão (arrastar para a direita = positiva, esquerda = negativa; botões para
neutra, mista e incerta), com tradução automática para português como apoio
(OPUS-MT `opus-mt-tc-big-en-pt`, rodando no navegador; vale o texto original).
Cada classificação é gravada numa planilha Google por um App da Web
(`anotacoes_apps_script.gs`). Os dois anotadores devem fazer as mesmas
resenhas, na ordem, sem conversar. Com a URL da planilha, o passo 7 separa os
anotadores, calcula Cohen's kappa e avalia todos os métodos:

```bash
py -3 07_analise_sentimento.py --run-transformer --annotations "<URL do App da Web>?format=csv"
```

Também aceita dois CSVs exportados pela página ou um CSV de referência com
`review_id` e `label`.

## 7. Predição de nota com rótulo fraco

`09_predicao_nota.py` usa a nota como rótulo fraco (1–2 = negativa, 4–5 =
positiva, 3 fora) para responder "o texto prevê a nota?", que é diferente de
validar o sentimento. Validação cruzada em 5 dobras agrupadas por componente
usuário-livro (reviews que compartilham usuário ou livro nunca se dividem entre
treino e teste), 5.485 reviews em inglês:

| Método | Acurácia balanceada | Macro-F1 | F1 negativa |
| --- | ---: | ---: | ---: |
| Transformer (siebert) | 0,910 ± 0,023 | 0,896 | 0,822 |
| TF-IDF + regressão logística | 0,772 ± 0,017 | 0,776 | 0,612 |
| VADER (95,7% de cobertura) | 0,669 ± 0,007 | 0,661 | 0,417 |
| Léxico (74,3% de cobertura) | 0,633 ± 0,014 | 0,663 | 0,383 |
| Classe majoritária | 0,500 | 0,464 | 0,000 |

Sem agrupamento, o TF-IDF obtém 0,774: o vazamento é pequeno nesta amostra,
porque quase todo usuário aparece uma vez. `termos_tfidf.csv` lista os termos
mais informativos (negativos: *not, didn't, boring, disappointing,
unfortunately*; positivos: *loved, great, enjoyed, amazing, wonderful*).

## 8. Coerência metodológica e estado dos resultados

O projeto distingue preparação, exploração, inferência e produto visual. Os
resultados numéricos principais são descritivos e observacionais. A
concordância com a nota é um rótulo fraco: mostra que o transformer acompanha a
avaliação numérica, não que acerta o sentimento de cada texto. A validação
humana (kappa entre anotadores e métricas contra os rótulos de consenso) é a
etapa que fecha essa lacuna. O site é uma prova de conceito de recomendação
baseada em similaridade textual agregada, não um sistema validado em usuários.

O artigo em [relatorio.tex](relatorio.tex) registra essa delimitação, descreve
os resultados efetivamente observados e separa conclusões de propostas futuras.

## 9. BookPulse: exploração e recomendação

O site na raiz (`index.html`, `styles.css`, `app.js` e `data/books.json`)
transforma os resultados agregados em uma interface de exploração. Ele permite:

- buscar livros pelo título ou autor (com navegação por teclado) e abrir um
  livro por link direto (`?livro=<book_id>`);
- navegar entre a página inicial e os livros selecionados com os controles
  `Início` e `Voltar`, além do histórico do navegador;
- carregar a capa por URL da Open Library, usando ISBN-13 ou ISBN-10;
- visualizar nota média, quantidade de reviews e extensão média;
- comparar a distribuição das notas, a quantidade de textos disponíveis e
  excertos de resenhas da amostra;
- comparar proporções de resenhas positivas e negativas segundo o transformer,
  apenas nas resenhas em inglês;
- consultar as palavras mais características das resenhas de cada livro,
  ranqueadas por TF-IDF (o livro é o documento), e os termos de resenhas
  positivas e negativas;
- explorar um grafo que liga o livro a essas palavras e a outros títulos que as
  compartilham; ao passar o mouse sobre uma capa, o site mostra título, autor,
  quantidade de resenhas e gêneros;
- recomendar livros com gêneros compartilhados (voto dos leitores) e
  probabilidade positiva média próxima.

Livros com menos de três resenhas exibem um aviso: o perfil reflete poucas
pessoas. As recomendações são exploratórias e não usam o texto bruto das
resenhas na página. A página `anotar.html` é a ferramenta de anotação descrita
na seção 6.

Para gerar os dados do site e servir a interface:

```bash
py -3 08_gerar_site.py
py -3 -m http.server 8000
```

Abra <http://localhost:8000>. O servidor local é necessário porque o navegador
bloqueia `fetch` de arquivos JSON quando a página é aberta por `file://`. Essa
mesma estrutura é compatível com o GitHub Pages. Para traduzir sem baixar o
modelo (~520 MB) do Hugging Face, coloque uma cópia em
`models/Douglasrambo/opus-mt-tc-big-en-pt-onnx/` (pasta ignorada pelo Git).

Arquivos temporários Python (`__pycache__`) e gráficos HTML antigos gerados
durante protótipos não fazem parte do produto final. O GitHub Pages depende
somente dos arquivos de site na raiz e de `data/books.json`; as ferramentas de
anotação e os scripts analíticos permanecem no repositório por serem partes
reprodutíveis do método.

## 10. Execução completa

```bash
py -3 -m pip install -r requirements.txt
py -3 01_amostra_reviews.py
py -3 01b_estantes_leitores.py --books caminho/para/goodreads_books.json.gz
py -3 02_cruzar_openlibrary.py
py -3 03_cruzar_google_books.py
py -3 04_validar_matches.py
py -3 05_preparar_dados.py
py -3 06_analisar_reviews.py
py -3 07_analise_sentimento.py --run-transformer
py -3 09_predicao_nota.py
py -3 08_gerar_site.py
py -3 -m pytest tests
```

O Google Books exige `GOOGLE_BOOKS_API_KEY` no ambiente ou em `.env`. As etapas
externas reaproveitam os caches persistidos em `processed/`. O arquivo
`goodreads_books.json.gz` (2 GB) está em
<https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_books.json.gz>.
O transformer exige `torch` e `transformers` (ver `requirements.txt`); sem eles,
`--run-transformer` reaproveita `transformer_cache.parquet`.

## 11. Referências metodológicas

- Pang, Lee e Vaithyanathan (2002), classificação supervisionada de reviews:
  <https://doi.org/10.3115/1118693.1118704>
- Hutto e Gilbert (2014), VADER:
  <https://doi.org/10.1609/icwsm.v8i1.14550>
- Hartmann, Heitmann, Siebert e Schamp (2023), acurácia de análise de
  sentimento: <https://doi.org/10.1016/j.ijresmar.2022.05.005>
- Socher et al. (2013), composição semântica:
  <https://aclanthology.org/D13-1170/>
- Devlin et al. (2019), BERT:
  <https://doi.org/10.18653/v1/N19-1423>
- Wan e McAuley (2018), UCSD Book Graph:
  <https://doi.org/10.1145/3240323.3240369>
- Tiedemann e Thottingal (2020), OPUS-MT:
  <https://aclanthology.org/2020.eamt-1.61/>
- scikit-learn, validação cruzada e métricas:
  <https://scikit-learn.org/stable/modules/cross_validation.html> e
  <https://scikit-learn.org/stable/modules/model_evaluation.html>
