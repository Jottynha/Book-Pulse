<div align="center">

# 📚 BookPulse

### Descobrindo o que as resenhas revelam sobre os livros

<p>
  <em>Um projeto acadêmico de Ciência de Dados sobre texto, sentimento,
  gêneros, avaliações e descoberta de livros.</em>
</p>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Plotly](https://img.shields.io/badge/visualizações-Plotly-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/python/)
[![GitHub Pages](https://img.shields.io/badge/site-GitHub%20Pages-222?logo=github)](https://jottynha.github.io/Book-Pulse/)
[![License](https://img.shields.io/badge/projeto-acadêmico-0e7770)](https://github.com/Jottynha/Book-Pulse)

<br>

**[🌐 Abrir o BookPulse](https://jottynha.github.io/Book-Pulse/)** ·
**[✍️ Anotar resenhas](https://jottynha.github.io/Book-Pulse/anotar.html)** ·
**[📄 Ler o relatório](./relatorio.tex)**

</div>

<br>

> **Em uma frase:** o BookPulse transforma milhares de resenhas em uma
> experiência interativa que conecta livros, palavras e sentimentos — sem
> confundir uma nota com a complexidade do texto que a acompanha.

<details>
<summary>🧭 Navegação rápida</summary>

- [O projeto](#-o-projeto)
- [O que foi descoberto](#-o-que-foi-descoberto)
- [O site como produto](#-o-site-como-produto)
- [Como os dados foram tratados](#-como-os-dados-foram-tratados)
- [Pipeline reproduzível](#-pipeline-reproduzível)
- [Rodando localmente](#-rodando-localmente)
- [Estrutura do repositório](#-estrutura-do-repositório)
- [Limitações honestas](#-limitações-honestas)
- [Relatório e figuras](#-relatório-e-figuras)
- [Créditos](#-créditos)

</details>

## ✨ O projeto

O BookPulse nasceu de uma pergunta simples:

> **Como os leitores avaliam livros e como essa avaliação aparece nas
> palavras que escolhem para escrever uma resenha?**

Para investigá-la, construímos um pipeline que parte de uma amostra do UCSD
Book Graph/Goodreads, enriquece os livros com catálogos externos, valida os
matches, trata os textos, compara métodos de sentimento e entrega os
resultados em uma interface de exploração.

O projeto combina:

| Camada | O que responde |
| --- | --- |
| 🔎 **Dados** | Quais resenhas e livros podem ser analisados com segurança? |
| 🧹 **Tratamento** | Como preservar o texto e criar variáveis comparáveis? |
| 💬 **Linguagem** | Quais palavras aparecem e como a polaridade varia? |
| 📊 **Métodos** | O que muda entre léxico, VADER e Transformer? |
| 🧭 **Produto** | Como transformar resultados técnicos em exploração compreensível? |

### O que este projeto não promete

“Comportamento” aqui significa **padrões observáveis de avaliação, escrita e
interação registrada na plataforma**. O projeto não diagnostica leitores, não
infere personalidade ou saúde mental e não trata a nota Goodreads como
verdade absoluta sobre o sentimento de uma resenha.

## 📌 O que foi descoberto

Esta execução usa uma amostra reprodutível de **10.000 resenhas**, selecionada
por reservoir sampling com `seed=42`.

| Indicador | Resultado |
| --- | ---: |
| Livros únicos na amostra | **8.916** |
| Livros com correspondência confiável | **7.031** |
| Resenhas confiáveis | **8.033** |
| Resenhas confiáveis em inglês | **7.182** |
| Usuários na base final | **7.350** |
| Matches Open Library | **7.654** |
| Matches Google Books | **4.424** |
| Matches em ambas as APIs | **3.852** |

### Resultados de sentimento

- O Transformer classificou **80,9%** das resenhas em inglês como positivas.
- A probabilidade positiva do Transformer teve Spearman
  **$\rho = 0,558$** com a nota.
- Para comparação: VADER alcançou **$\rho = 0,238$** e o léxico
  **$\rho = 0,253$**.
- A fração negativa caiu de **95,9% na nota 1** para **2,0% na nota 5**.

### Predição da polaridade da nota

| Método | Acurácia balanceada | Macro-F1 | F1 negativo |
| --- | ---: | ---: | ---: |
| Transformer | **0,910 ± 0,023** | **0,896** | **0,822** |
| TF-IDF + regressão logística | 0,772 ± 0,017 | 0,776 | 0,612 |
| VADER | 0,669 ± 0,007 | 0,661 | 0,417 |
| Léxico | 0,633 ± 0,014 | 0,663 | 0,383 |
| Classe majoritária | 0,500 | 0,464 | 0,000 |

> A tarefa acima usa a nota apenas como **rótulo fraco**: notas 1–2 são
> chamadas de negativas, notas 4–5 de positivas e nota 3 é excluída. Isso
> mede se o texto acompanha a nota; não valida sozinho o sentimento textual.

## 🌐 O site como produto

O site é a condensação interativa do pipeline. A pessoa pode começar pela
descoberta e chegar até a evidência textual de um livro:

```text
🏠 Home
  └─ 🔍 busca por título ou autor
       └─ 📖 página do livro
            ├─ 📊 nota, reviews e sentimento
            ├─ 🔤 vocabulário característico
            ├─ 🕸️ grafo de palavras e capas relacionadas
            ├─ 💬 resenhas em destaque
            └─ 🧭 recomendações exploratórias
```

### Uma visita guiada

1. **Descobrir** — a home apresenta a proposta, os livros em destaque e a
   busca com navegação por teclado.
2. **Selecionar** — a página da obra mostra capa, autor, gêneros, nota média,
   quantidade de resenhas e extensão média.
3. **Interpretar** — gráficos de sentimento, distribuição de notas, palavras
   positivas/negativas e excertos ajudam a contextualizar os números.
4. **Relacionar** — o grafo coloca o livro no centro, termos no primeiro anel
   e títulos que compartilham palavras no anel externo.
5. **Validar** — a ferramenta de anotação oculta a nota original e permite
   comparar a leitura humana com os modelos automáticos.

<div align="center">

| 🏠 Página inicial | 📖 Página de livro |
| --- | --- |
| Acesso à busca, destaques e anotação humana | Perfil textual, sentimento e recomendações |
| *Captura opcional: `figures/site_home.png`* | *Captura opcional: `figures/site_livro.png`* |

</div>

O site está na raiz para preservar o GitHub Pages:

```text
index.html  ·  styles.css  ·  app.js  ·  favicon.svg
data/books.json  ·  anotar.html  ·  anotar.js  ·  anotar.css
```

As recomendações são exploratórias: combinam gêneros compartilhados e
proximidade da probabilidade positiva média. O grafo representa
**compartilhamento lexical**, não uma distância semântica validada.

## 🧪 Como os dados foram tratados

### Enriquecimento e validação

Open Library foi consultada priorizando ISBN-13 e ISBN-10, seguida por título
e autor. Google Books funcionou como fonte independente. O validador combinou
ISBN, similaridade de título, autores, ano de publicação e concordância entre
fontes. A análise principal mantém apenas `MUITO_ALTO`, `ALTO` e `MEDIO`.

O filtro melhora a confiabilidade, mas introduz seleção: os livros mantidos
são mais populares, enquanto as resenhas descartadas têm nota média
ligeiramente maior.

### Gêneros pelo voto dos leitores

Os gêneros não são simplesmente copiados do catálogo. O pipeline recupera as
estantes do Goodreads e:

- usa a maior contagem entre estantes equivalentes;
- exige pelo menos 3 votos e 10% do gênero mais votado;
- aceita até 3 gêneros por livro;
- prioriza gêneros específicos em vez de `nonfiction`;
- registra a origem em `genre_source`.

Na execução atual, **6.416 livros** receberam gênero pelo voto dos leitores,
**550** pelo fallback de rótulos e **65** ficaram sem classificação. A
concordância entre gênero principal de leitores e catálogo foi de **49,7%**.

### Sentimento e validação

São comparados três níveis de complexidade:

1. **Léxico:** transparente, barato e com regra simples de negação.
2. **VADER:** adiciona regras de intensidade, pontuação e linguagem informal.
3. **Transformer RoBERTa:** usa representação contextual pré-treinada.

As resenhas fora do inglês permanecem na base, mas não entram nos modelos
desenvolvidos para inglês. Reviews longas são divididas em trechos de até 350
palavras, com sobreposição de 50 palavras.

A anotação humana usa **600 resenhas** estratificadas. O CSV possui 601 linhas
físicas porque uma delas é o cabeçalho. A nota original é ocultada durante a
anotação, que pode marcar uma resenha como positiva, negativa, neutra, mista
ou incerta.

## 🔁 Pipeline reproduzível

```text
01_amostra_reviews.py        → processed/01_amostra/
01b_estantes_leitores.py     → estantes e votos de gênero
02_cruzar_openlibrary.py     → processed/02_openlibrary/
03_cruzar_google_books.py    → processed/03_google_books/
04_validar_matches.py        → processed/04_validacao/
05_preparar_dados.py         → processed/05_dados_prontos/
06_analisar_reviews.py       → processed/06_analise/
07_analise_sentimento.py     → processed/07_sentimento/
09_predicao_nota.py          → processed/09_predicao_nota/
08_gerar_site.py             → data/books.json
```

### Parâmetros principais

| Parâmetro | Valor |
| --- | ---: |
| Semente de amostragem | `42` |
| Amostra inicial | `10.000` reviews |
| Mínimo de votos para gênero | `3` |
| Corte relativo de gênero | `10%` |
| Máximo de gêneros por livro | `3` |
| Tamanho máximo do trecho | `350` palavras |
| Sobreposição entre trechos | `50` palavras |
| Dobras agrupadas | `5` |

## 🚀 Rodando localmente

### 1. Instalar dependências

```bash
py -3 -m pip install -r requirements.txt
```

### 2. Executar o pipeline

```bash
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

O Google Books exige `GOOGLE_BOOKS_API_KEY` no ambiente ou em `.env`. As
etapas externas reutilizam caches em `processed/`. O dataset original pode
ser obtido no [UCSD Book Graph](https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/).

### 3. Servir o site

```bash
py -3 -m http.server 8000
```

Abra <http://localhost:8000>. O servidor local é necessário porque o navegador
bloqueia `fetch` de JSON em `file://`.

### 4. Gerar as figuras do artigo

```bash
py -3 gerar_figuras_relatorio.py
```

O script usa Plotly + Kaleido e grava PDFs para o LaTeX e SVGs para inspeção
em [figures/](./figures/). Não usa Matplotlib ou Seaborn nessa etapa.

## 🗂️ Estrutura do repositório

```text
Book-Pulse/
├── 01_...09_*.py          # etapas reprodutíveis do pipeline
├── index.html              # entrada do GitHub Pages
├── app.js / styles.css     # interface e interações
├── anotar.html             # anotação humana
├── data/books.json         # dados agregados do site
├── processed/              # resultados intermediários e auditoria
├── figures/                # gráficos do relatório
├── gerar_figuras_relatorio.py
├── relatorio.tex           # artigo IEEE
├── requirements.txt
└── tests/
```

## ⚠️ Limitações honestas

- A amostragem é por resenha: **92% dos livros aparecem uma única vez**.
- O filtro de correspondência favorece livros populares e bem catalogados.
- A nota Goodreads é um indicador relacionado, não ground truth de sentimento.
- O Transformer foi treinado em domínio externo.
- O modelo binário não representa bem textos neutros ou mistos.
- O grafo mede compartilhamento de palavras, não semelhança semântica.
- A recomendação ainda não foi avaliada por ranking ou estudo com usuários.
- Ainda não foram calculados intervalos de confiança por bootstrap.
- A validação humana independente ainda precisa ser concluída.
- O relatório LaTeX ainda está em um arquivo único; sua modularização está
  planejada, mas não foi feita para preservar uma compilação simples.

Essas limitações não invalidam a exploração; elas definem o alcance correto das
conclusões.

## 📄 Relatório e figuras

O [relatório IEEE](./relatorio.tex) documenta dados, metodologia, resultados,
ética, validade e o site como condensação interativa do projeto.

Ele reserva espaços para quatro capturas opcionais. Para preenchê-los no
Overleaf, envie para `figures/`:

```text
site_home.png
site_livro.png
site_grafo.png
site_anotacao.png
```

Sem essas imagens, o relatório exibe caixas indicativas e continua compilável.

## 🤝 Créditos

Projeto acadêmico desenvolvido por:

- **João Pedro Rodrigues Silva** — [@Jottynha](https://github.com/Jottynha)
- **Henrique de Freitas Araujo** — [@Ak4ai](https://github.com/Ak4ai)

🐛 Encontrou um problema ou tem uma sugestão? Abra uma
[issue](https://github.com/Jottynha/Book-Pulse/issues).

<div align="center">

<sub>Feito para transformar dados de leitura em perguntas melhores — e não em respostas simplistas.</sub>

</div>
