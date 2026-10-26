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
vocabulário pequeno e explícito em inglês. A regra inverte a polaridade quando
há um negador nas três palavras anteriores; `sentiment_score_per_100_words`
reduz o efeito mecânico do tamanho da review. `sentiment_label` classifica o
score como `positive`, `negative` ou `neutral`. Isso continua sendo uma linha
de base auditável, não uma medição validada: não trata adequadamente ironia,
escopo linguístico complexo ou contexto específico de livros.

## Fundamentação e limites metodológicos

O score lexical é mantido como baseline interpretável. A literatura mostra que
classificação de reviews depende de composição e contexto, e não apenas de
contagem de palavras (Pang, Lee e Vaithyanathan, 2002; Socher et al., 2013).
Para uma versão final do trabalho, a recomendação é:

1. anotar manualmente uma amostra estratificada e medir concordância entre
   anotadores (Cohen's kappa);
2. comparar o baseline com VADER e TF-IDF + regressão logística ou Linear SVM;
3. manter um conjunto de teste separado, evitando usar a nota Goodreads como
   verdade absoluta de sentimento;
4. reportar macro-F1, F1 por classe, balanced accuracy, matriz de confusão e
   exemplos de erro;
5. comparar splits por review, usuário e livro para investigar vazamento;
6. interpretar correlações como associações observacionais, não causalidade.

As correlações exportadas usam Pearson e Spearman; são medidas exploratórias e
não controlam confundidores como usuário, livro, gênero ou popularidade.

Um transformer, como DistilBERT ou RoBERTa, pode ser uma extensão posterior,
desde que seja treinado e avaliado com rótulos humanos ou com o proxy de rating
explicitamente identificado como rótulo fraco.

## Referências

- Pang, B.; Lee, L.; Vaithyanathan, S. (2002). *Thumbs up? Sentiment
  Classification using Machine Learning Techniques*.
  https://doi.org/10.3115/1118693.1118704
- Hutto, C. J.; Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model
  for Sentiment Analysis of Social Media Text*.
  https://doi.org/10.1609/icwsm.v8i1.14550
- Socher, R. et al. (2013). *Recursive Deep Models for Semantic
  Compositionality Over a Sentiment Treebank*.
  https://aclanthology.org/D13-1170/
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding*.
  https://doi.org/10.18653/v1/N19-1423
- scikit-learn. *Cross-validation* e *Metrics and scoring*.
  https://scikit-learn.org/stable/modules/cross_validation.html
  e https://scikit-learn.org/stable/modules/model_evaluation.html

## Saídas

- `reviews_refinadas.parquet`: base com tokens e variáveis textuais;
- `resumo_por_nota.csv`: métricas agrupadas pela nota;
- `resumo_por_genero.csv`: métricas agrupadas pelo gênero principal;
- `resumo_por_usuario.csv`: atividade agregada por usuário;
- `correlacoes_com_rating.csv`: correlações exploratórias com a nota;
- `visao_geral_reviews.png`: distribuição de notas e relação inicial com sentimento.
