"""Testes das regras do pipeline que não dependem dos dados processados."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest


RAIZ = Path(__file__).resolve().parents[1]


def carregar(nome: str):
    spec = importlib.util.spec_from_file_location(nome.replace(".py", ""), RAIZ / nome)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


@pytest.fixture(scope="module")
def preparar():
    return carregar("05_preparar_dados.py")


@pytest.fixture(scope="module")
def sentimento():
    return carregar("07_analise_sentimento.py")


@pytest.fixture(scope="module")
def site():
    return carregar("08_gerar_site.py")


@pytest.fixture(scope="module")
def traducao():
    return carregar("10_traducao_reviews.py")


def test_generos_por_palavra_inteira(preparar):
    assert preparar.generos_do_rotulo("teen fiction") == {"young_adult"}
    assert preparar.generos_do_rotulo("eighteenth century") == set()
    assert preparar.generos_do_rotulo("display") == set()


def test_voto_dos_leitores_usa_maior_contagem_e_corte(preparar):
    estantes = [
        ("to-read", 50000), ("dystopian", 15000), ("science-fiction", 7000),
        ("young-adult", 12000), ("fantasy", 4000), ("romance", 900),
    ]
    generos, votos, _ = preparar.generos_por_leitores(estantes)
    assert votos["science_fiction"] == 15000  # máximo, não 22000
    assert generos == ["science_fiction", "young_adult", "fantasy"]
    assert "romance" not in generos  # abaixo de 10% do mais votado


def test_nonfiction_cede_para_genero_especifico(preparar):
    estantes = [("non-fiction", 900), ("biography", 300), ("memoir", 250)]
    generos, _, _ = preparar.generos_por_leitores(estantes)
    assert generos[0] == "biography"
    assert "nonfiction" in generos


def test_empate_decidido_pelo_catalogo(preparar):
    estantes = [("mystery-thriller", 500)]
    generos, _, _ = preparar.generos_por_leitores(estantes, catalogo=["thriller"])
    assert generos[0] == "thriller"


def test_sem_votos_usa_rotulos_e_sinaliza(preparar):
    livro = pd.Series({
        "book_id": "1", "popular_shelves": "fantasy|to-read",
        "openlibrary_subjects": "", "google_categories": "Fiction / Romance",
    })
    resultado = preparar.encontrar_generos(livro, None)
    assert resultado["genre_source"] == "labels_fallback"
    assert set(resultado["genres_assigned"].split("|")) == {"fantasy", "romance"}


@pytest.mark.parametrize(
    ("texto", "idioma"),
    [
        ("I really loved this book, the characters were wonderful.", "en"),
        ("Loved it!", "en"),
        ("Este libro es un ensayo deslumbrante sobre la evolucion del concepto.", "es"),
        ("", "und"),
    ],
)
def test_detectar_idioma(sentimento, texto, idioma):
    assert sentimento.detectar_idioma(texto) == idioma


def test_componentes_usuario_livro_agrupa_por_usuario_ou_livro(sentimento):
    dados = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u3"],
        "book_id": ["b1", "b2", "b2", "b9"],
    })
    grupos = sentimento.componentes_usuario_livro(dados)
    assert grupos.iloc[0] == grupos.iloc[1] == grupos.iloc[2]
    assert grupos.iloc[3] != grupos.iloc[0]


def test_fragmentos_cobrem_o_texto_com_sobreposicao(sentimento):
    palavras = [f"w{i}" for i in range(700)]
    partes = sentimento.fragmentos(" ".join(palavras), limite=350, sobreposicao=50)
    assert partes[0].split()[0] == "w0"
    assert partes[-1].split()[-1] == "w699"
    assert all(len(parte.split()) <= 350 for parte in partes)


def test_dois_anotadores_numa_planilha(sentimento, tmp_path):
    planilha = tmp_path / "planilha.csv"
    pd.DataFrame({
        "review_id": ["r1", "r2", "r1", "r2", "r1"],
        "label": ["positive", "negative", "positive", "neutral", "negative"],
        "annotator": ["ana", "ana", "bia", "bia", "ana"],
        "annotated_at": ["1", "1", "1", "1", "2"],
    }).to_csv(planilha, index=False)
    dados = sentimento.carregar_anotacoes([str(planilha)]).set_index("review_id")
    # A reclassificação mais recente de "ana" (r1 -> negative) prevalece.
    assert dados.loc["r1", "annotator_a"] == "negative"
    assert pd.isna(dados.loc["r1", "label"])  # divergência fica sem referência
    assert pd.isna(dados.loc["r2", "label"])


def test_palavras_tfidf_rebaixam_termos_de_todo_livro(site):
    tokens = pd.Series(["katniss|katniss|series|series|series"])
    frequencia = Counter({"katniss": 3, "series": 100})
    resultado = site.palavras(tokens, frequencia, total_livros=100)
    # "series" aparece mais vezes, mas está em todo livro (idf = 0) e cai para trás.
    assert [item["word"] for item in resultado] == ["katniss", "series"]
    assert resultado[1]["count"] == 3  # a contagem exibida continua bruta


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("kmlt rwyh qryh Zlmh lmHmd kml Hsyn w lq byn wbyn lktb", True),
        ("slwb jyd lkn qS@ mml@ ...", True),
        ("Este libro es un ensayo deslumbrante sobre el amor.", False),
        ("Dreadful.", False),
        ("539 - 2016", False),
    ],
)
def test_detecta_arabe_transliterado(traducao, texto, esperado):
    assert traducao.transliterado(texto) is esperado


def test_rotas_de_traducao(traducao):
    assert traducao.definir_rota("Muy divertido", "es", 0.95, 0.5) == "translate"
    assert traducao.definir_rota("Dreadful.", "en", 0.99, 0.5) == "english"
    assert traducao.definir_rota("3.5", "en", 0.2, 0.5) == "undetermined"
    assert traducao.definir_rota("texto", "xx", 0.99, 0.5) == "undetermined"
    assert traducao.definir_rota("", "es", 0.99, 0.5) == "undetermined"


def test_segmentar_respeita_limite_e_ordem(traducao):
    texto = "Uma frase curta. " + " ".join(f"p{i}" for i in range(250)) + ". Fim."
    trechos = traducao.segmentar(texto, limite=120)
    assert all(len(trecho.split()) <= 120 for trecho in trechos)
    assert " ".join(trechos).split() == texto.split()


def test_rotulo_fraco_e_acuracia_balanceada(traducao):
    import numpy as np

    rotulo = traducao.rotulo_fraco(pd.Series([1.0, 3.0, 5.0, None]))
    assert rotulo.tolist()[0] == "negative" and rotulo.tolist()[2] == "positive"
    assert pd.isna(rotulo.iloc[1]) and pd.isna(rotulo.iloc[3])
    verdade = np.array(["negative", "positive", "positive", "positive"])
    previsto = np.array(["negative", "positive", "negative", "negative"])
    # recall negativo 1,0 e positivo 1/3
    assert traducao.acuracia_balanceada(verdade, previsto) == pytest.approx(2 / 3)
