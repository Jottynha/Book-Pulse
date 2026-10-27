#!/usr/bin/env python3
"""Sentimento das reviews fora do inglês, por tradução automática.

O passo 7 classifica só reviews em inglês, porque o transformer usado
(siebert/sentiment-roberta-large-english) é monolíngue. Este passo trata as
demais em quatro rotas:

1. ``english``: o fastText (LID-176) reconhece como inglês uma review que o
   langdetect não reconheceu (em geral, textos curtos como "Dreadful.").
   O texto original vai direto ao classificador.
2. ``translate``: idioma identificado com confiança e suportado pelo NLLB-200.
   A review é traduzida para o inglês e a tradução vai ao mesmo classificador.
3. ``transliterated``: árabe ou persa escrito em caracteres latinos por
   transliteração (o langdetect costuma chamar de galês). Nenhum dos modelos
   foi treinado nesse formato; essas reviews ficam fora.
4. ``undetermined``: idioma indeterminado (links, números, textos curtos
   demais) ou sem suporte no tradutor. Também ficam fora.

A avaliação usa o rótulo fraco do passo 9 (nota 1-2 = negative, 4-5 =
positive) e compara, nas mesmas reviews traduzidas: o classificador no texto
original, o classificador na tradução e um modelo multilíngue
(cardiffnlp/twitter-xlm-roberta-base-sentiment) no texto original. Os
resultados ficam à parte da análise principal, que segue só em inglês.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ENTRADA_PADRAO = BASE_DIR / "processed" / "07_sentimento" / "reviews_sentimento.parquet"
SAIDA_PADRAO = BASE_DIR / "processed" / "10_traducao"
LID_PADRAO = BASE_DIR / "models" / "lid.176.ftz"
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
TRADUTOR_PADRAO = "facebook/nllb-200-distilled-600M"
CLASSIFICADOR_PADRAO = "siebert/sentiment-roberta-large-english"
MULTILINGUE_PADRAO = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
CLASSES = ["negative", "positive"]

# Códigos ISO 639-1 do fastText -> códigos FLORES-200 do NLLB.
NLLB_CODIGOS = {
    "af": "afr_Latn", "ar": "arb_Arab", "bg": "bul_Cyrl", "ca": "cat_Latn",
    "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn", "de": "deu_Latn",
    "el": "ell_Grek", "es": "spa_Latn", "et": "est_Latn", "eu": "eus_Latn",
    "fa": "pes_Arab", "fi": "fin_Latn", "fr": "fra_Latn", "gl": "glg_Latn",
    "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn", "hu": "hun_Latn",
    "id": "ind_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "lt": "lit_Latn", "lv": "lvs_Latn", "ms": "zsm_Latn", "nl": "nld_Latn",
    "no": "nob_Latn", "pl": "pol_Latn", "pt": "por_Latn", "ro": "ron_Latn",
    "ru": "rus_Cyrl", "sk": "slk_Latn", "sl": "slv_Latn", "so": "som_Latn",
    "sq": "als_Latn", "sr": "srp_Cyrl", "sv": "swe_Latn", "sw": "swh_Latn",
    "tl": "tgl_Latn", "tr": "tur_Latn", "uk": "ukr_Cyrl", "vi": "vie_Latn",
    "zh": "zho_Hans",
}


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ENTRADA_PADRAO)
    parser.add_argument("--output", type=Path, default=SAIDA_PADRAO)
    parser.add_argument("--lid-model", type=Path, default=LID_PADRAO)
    parser.add_argument("--min-lid-prob", type=float, default=0.5)
    parser.add_argument("--translation-model", default=TRADUTOR_PADRAO)
    parser.add_argument("--transformer-model", default=CLASSIFICADOR_PADRAO)
    parser.add_argument("--multilingual-model", default=MULTILINGUE_PADRAO)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--translation-batch-size", type=int, default=8)
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def carregar_modulo(nome: str) -> Any:
    caminho = BASE_DIR / nome
    spec = importlib.util.spec_from_file_location(nome.replace(".py", ""), caminho)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Não foi possível carregar: {caminho}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def sem_links(texto: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"https?://\S+|www\.\S+", " ", texto)).strip()


def transliterado(texto: str) -> bool:
    """Árabe ou persa em caracteres latinos (ex.: "qr't wSf jmyl jdan").

    A transliteração usa ` e @ no lugar de letras árabes e quase não escreve
    vogais; nenhum idioma de escrita latina tem esse perfil.
    """
    palavras = texto.split()
    letras = re.findall(r"[A-Za-z]", texto)
    if not palavras or not letras:
        return False
    especiais = (texto.count("`") + texto.count("@")) / len(palavras)
    vogais = sum(letra in "aeiouAEIOU" for letra in letras) / len(letras)
    return especiais >= 0.05 or (vogais < 0.25 and len(letras) >= 20)


def identificar_idiomas(textos: list[str], caminho_modelo: Path) -> tuple[list[str], list[float]]:
    try:
        import fasttext
    except ImportError as exc:
        raise RuntimeError("fastText indisponível. Instale fasttext-wheel.") from exc
    if not caminho_modelo.exists():
        caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
        print(f"Baixando LID-176 em {caminho_modelo}")
        urllib.request.urlretrieve(LID_URL, caminho_modelo)
    modelo = fasttext.load_model(str(caminho_modelo))
    rotulos, probs = modelo.predict([texto.replace("\n", " ") for texto in textos], k=1)
    return [r[0].removeprefix("__label__") for r in rotulos], [float(p[0]) for p in probs]


def definir_rota(texto: str, idioma: str, prob: float, min_prob: float) -> str:
    if not texto:
        return "undetermined"
    if transliterado(texto):
        return "transliterated"
    if prob < min_prob:
        return "undetermined"
    if idioma == "en":
        return "english"
    return "translate" if idioma in NLLB_CODIGOS else "undetermined"


def segmentar(texto: str, limite: int = 120) -> list[str]:
    """Divide em frases e as agrupa em trechos de até `limite` palavras.

    O NLLB foi treinado com frases; trechos longos tendem a ser resumidos ou
    truncados na tradução.
    """
    frases = [frase for frase in re.split(r"(?<=[.!?])\s+|\n+", texto) if frase.strip()]
    trechos: list[str] = []
    atual: list[str] = []
    for frase in frases:
        palavras = frase.split()
        for inicio in range(0, len(palavras), limite):
            pedaco = palavras[inicio:inicio + limite]
            if atual and len(atual) + len(pedaco) > limite:
                trechos.append(" ".join(atual))
                atual = []
            atual.extend(pedaco)
    if atual:
        trechos.append(" ".join(atual))
    return trechos


def confianca(modelo: Any, entrada: Any, saida: Any, pad: int) -> list[float]:
    """Log-probabilidade média por token de cada tradução gerada.

    Recalculada com uma passagem do modelo sobre a saída: pedir os scores ao
    `generate` guardaria os logits de todo o vocabulário (256 mil tokens) a
    cada passo do beam search, o que não cabe numa GPU de 6 GB.
    """
    import torch

    codificado = modelo.get_encoder()(
        input_ids=entrada["input_ids"], attention_mask=entrada["attention_mask"]
    )
    resultado = []
    for linha in range(saida.shape[0]):
        logits = modelo(
            encoder_outputs=(codificado.last_hidden_state[linha:linha + 1],),
            attention_mask=entrada["attention_mask"][linha:linha + 1],
            decoder_input_ids=saida[linha:linha + 1, :-1],
        ).logits[0].float()
        alvo = saida[linha, 1:]
        # A primeira posição é o código de idioma forçado; não conta.
        mascara = alvo.ne(pad)
        mascara[0] = False
        logprobs = torch.log_softmax(logits, dim=-1).gather(1, alvo[:, None])[:, 0]
        resultado.append(float(logprobs[mascara].mean()) if mascara.any() else float("nan"))
    return resultado


def traduzir(
    reviews: pd.DataFrame, modelo_nome: str, batch_size: int, beams: int, cache: Path
) -> pd.DataFrame:
    """Traduz `text_clean` de cada review para o inglês.

    A confiança da tradução é a log-probabilidade média por token da saída,
    ponderada pelo número de palavras de cada trecho. Valores muito baixos
    indicam texto que o modelo não entendeu.
    """
    colunas = ["review_id", "translation", "translation_logprob", "translation_segments"]
    anterior = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=colunas)
    if anterior.attrs.get("model", modelo_nome) != modelo_nome:
        anterior = pd.DataFrame(columns=colunas)
    faltantes = reviews[~reviews["review_id"].isin(anterior["review_id"])]
    if faltantes.empty:
        print(f"Tradução: reaproveitando cache {cache}")
        return anterior[colunas]

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    usar_gpu = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
    modelo = AutoModelForSeq2SeqLM.from_pretrained(
        modelo_nome, dtype=torch.float16 if usar_gpu else torch.float32
    ).to("cuda" if usar_gpu else "cpu").eval()
    destino = tokenizer.convert_tokens_to_ids("eng_Latn")
    modelo.generation_config.max_length = None

    trechos = [
        (posicao, indice, origem, trecho)
        for posicao, (texto, origem) in enumerate(
            zip(faltantes["text_clean"], faltantes["nllb_source"])
        )
        for indice, trecho in enumerate(segmentar(texto))
    ]
    # Lotes de uma só língua de origem, ordenados por tamanho.
    trechos.sort(key=lambda item: (item[2], len(item[3])))
    traduzidos: dict[int, list[tuple[int, str, float, int]]] = {}
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    lotes = []
    for inicio in range(0, len(trechos), batch_size):
        lote = trechos[inicio:inicio + batch_size]
        origens = {item[2] for item in lote}
        if len(origens) > 1:
            for origem in sorted(origens):
                lotes.append([item for item in lote if item[2] == origem])
        else:
            lotes.append(lote)
    for lote in tqdm(lotes, desc="Tradução", unit="lote") if tqdm else lotes:
        tokenizer.src_lang = lote[0][2]
        entrada = tokenizer(
            [item[3] for item in lote], return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        ).to(modelo.device)
        with torch.no_grad():
            saida = modelo.generate(
                **entrada,
                forced_bos_token_id=destino,
                num_beams=beams,
                max_new_tokens=int(entrada["input_ids"].shape[1] * 1.6) + 10,
                no_repeat_ngram_size=4,
            )
            scores = confianca(modelo, entrada, saida, tokenizer.pad_token_id)
        textos = tokenizer.batch_decode(saida, skip_special_tokens=True)
        for (posicao, indice, _, trecho), texto, score in zip(lote, textos, scores):
            traduzidos.setdefault(posicao, []).append(
                (indice, texto, float(score), len(trecho.split()))
            )

    linhas = []
    for posicao, review_id in enumerate(faltantes["review_id"]):
        # Os trechos voltam à ordem do texto (os lotes os ordenaram por tamanho).
        partes = sorted(traduzidos.get(posicao, []))
        peso = sum(p for _, _, _, p in partes)
        linhas.append({
            "review_id": review_id,
            "translation": " ".join(t for _, t, _, _ in partes),
            "translation_logprob": sum(s * p for _, _, s, p in partes) / peso if peso else None,
            "translation_segments": len(partes),
        })
    novos = pd.DataFrame(linhas)
    resultado = pd.concat([anterior[colunas], novos], ignore_index=True)
    resultado.attrs["model"] = modelo_nome
    resultado.to_parquet(cache, index=False)
    return resultado


def inferir_multilingue(
    textos: pd.Series, modelo_nome: str, batch_size: int, fragmentos: Any
) -> pd.DataFrame:
    """Classificador multilíngue de três classes direto no texto original.

    Mesma agregação do passo 7: média das probabilidades dos trechos,
    ponderada pelo número de palavras. A classe final é a de maior
    probabilidade, e "neutral" conta como não decidida na avaliação.
    """
    import torch
    from transformers import pipeline

    usar_gpu = torch.cuda.is_available()
    classificador = pipeline(
        "text-classification", model=modelo_nome, device=0 if usar_gpu else -1,
        dtype=torch.float16 if usar_gpu else torch.float32,
    )
    trechos = [
        (posicao, parte)
        for posicao, texto in enumerate(textos)
        for parte in fragmentos(texto, limite=200, sobreposicao=30)
    ]
    trechos.sort(key=lambda item: len(item[1]))
    acumulado: dict[int, list[tuple[dict[str, float], int]]] = {}
    for inicio in range(0, len(trechos), batch_size):
        lote = trechos[inicio:inicio + batch_size]
        previsoes = classificador(
            [parte for _, parte in lote], batch_size=batch_size, truncation=True,
            max_length=512, top_k=None,
        )
        for (posicao, parte), scores in zip(lote, previsoes):
            probs = {str(s["label"]).lower(): float(s["score"]) for s in scores}
            acumulado.setdefault(posicao, []).append((probs, len(parte.split())))
    linhas = []
    for posicao in range(len(textos)):
        partes = acumulado.get(posicao, [])
        peso = sum(p for _, p in partes)
        if not peso:
            linhas.append({"multilingual_label": "", "multilingual_positive_prob": None})
            continue
        media = {
            classe: sum(probs.get(classe, 0.0) * p for probs, p in partes) / peso
            for classe in ("negative", "neutral", "positive")
        }
        linhas.append({
            "multilingual_label": max(media, key=media.get),
            "multilingual_positive_prob": media["positive"],
        })
    return pd.DataFrame(linhas, index=textos.index)


def rotulo_fraco(notas: pd.Series) -> pd.Series:
    rotulo = pd.Series(pd.NA, index=notas.index, dtype="object")
    rotulo[notas <= 2] = "negative"
    rotulo[notas >= 4] = "positive"
    return rotulo


def acuracia_balanceada(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = [
        np.mean(y_pred[y_true == classe] == classe)
        for classe in CLASSES if np.any(y_true == classe)
    ]
    return float(np.mean(recalls)) if recalls else float("nan")


def avaliar(
    y_true: pd.Series, y_pred: pd.Series, reamostragens: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Métricas no rótulo fraco, com IC de 95% por bootstrap.

    Como no passo 9, "neutral" e rótulos vazios contam como não decididos:
    entram na cobertura, não na acurácia.
    """
    from sklearn.metrics import f1_score

    validos = y_true.notna()
    y_true, y_pred = y_true[validos].astype(str), y_pred[validos].fillna("").astype(str)
    decididas = y_pred.isin(CLASSES)
    verdade, previsto = y_true[decididas].to_numpy(), y_pred[decididas].to_numpy()
    amostras = [
        acuracia_balanceada(verdade[indices], previsto[indices])
        for indices in (rng.integers(0, len(verdade), len(verdade)) for _ in range(reamostragens))
    ] if len(verdade) else [float("nan")]
    return {
        "rows": int(validos.sum()),
        "negatives": int((y_true == "negative").sum()),
        "coverage": float(decididas.mean()) if len(decididas) else float("nan"),
        "balanced_accuracy": acuracia_balanceada(verdade, previsto),
        "ci_low": float(np.nanpercentile(amostras, 2.5)),
        "ci_high": float(np.nanpercentile(amostras, 97.5)),
        "f1_negative": float(f1_score(
            verdade, previsto, pos_label="negative", average="binary", zero_division=0
        )) if len(verdade) else float("nan"),
        "negative_share_pred": float(np.mean(previsto == "negative")) if len(previsto) else float("nan"),
    }


def diferenca_pareada(
    y_true: pd.Series, pred_a: pd.Series, pred_b: pd.Series,
    reamostragens: int, rng: np.random.Generator,
) -> dict[str, float]:
    """Diferença de acurácia balanceada A - B nas mesmas reviews (bootstrap pareado)."""
    validos = y_true.notna() & pred_a.isin(CLASSES) & pred_b.isin(CLASSES)
    verdade = y_true[validos].astype(str).to_numpy()
    a, b = pred_a[validos].to_numpy(), pred_b[validos].to_numpy()
    diferencas = []
    for _ in range(reamostragens):
        indices = rng.integers(0, len(verdade), len(verdade))
        diferencas.append(
            acuracia_balanceada(verdade[indices], a[indices])
            - acuracia_balanceada(verdade[indices], b[indices])
        )
    return {
        "rows": int(validos.sum()),
        "difference": acuracia_balanceada(verdade, a) - acuracia_balanceada(verdade, b),
        "ci_low": float(np.nanpercentile(diferencas, 2.5)),
        "ci_high": float(np.nanpercentile(diferencas, 97.5)),
    }


def main() -> None:
    args = argumentos()
    args.output.mkdir(parents=True, exist_ok=True)
    sentimento = carregar_modulo("07_analise_sentimento.py")
    rng = np.random.default_rng(args.seed)

    reviews = pd.read_parquet(args.input)
    reviews["weak_label"] = rotulo_fraco(reviews["rating_valido"])
    ingles = reviews[reviews["text_is_english"]].copy()
    outras = reviews[~reviews["text_is_english"] & ~reviews["text_is_empty"]].copy()
    outras["text_clean"] = outras["text_for_inference"].map(sem_links)
    outras["lid_language"], outras["lid_prob"] = identificar_idiomas(
        outras["text_clean"].tolist(), args.lid_model
    )
    outras["route"] = [
        definir_rota(texto, idioma, prob, args.min_lid_prob)
        for texto, idioma, prob in zip(outras["text_clean"], outras["lid_language"], outras["lid_prob"])
    ]
    outras["nllb_source"] = outras["lid_language"].map(NLLB_CODIGOS).where(
        outras["route"].eq("translate")
    )
    print("Rotas:", outras["route"].value_counts().to_dict())

    traducoes = traduzir(
        outras[outras["route"].eq("translate")], args.translation_model,
        args.translation_batch_size, args.beams, args.output / "traducao_cache.parquet",
    )
    outras = outras.merge(traducoes, on="review_id", how="left")

    # Classificador do passo 7 aplicado às traduções (cache próprio).
    traduzidas = outras.loc[outras["route"].eq("translate"), ["review_id", "translation"]]
    traduzidas = traduzidas.rename(columns={"translation": "text_for_inference"})
    classificadas = sentimento.inferir_transformer(
        traduzidas, args.transformer_model, args.batch_size,
        args.output / "transformer_traducao_cache.parquet",
    )
    classificadas = classificadas[[
        "review_id", "transformer_label", "transformer_positive_prob", "transformer_chunks"
    ]].rename(columns=lambda coluna: coluna.replace("transformer_", "translated_"))
    outras = outras.merge(classificadas, on="review_id", how="left")

    # Modelo multilíngue nas reviews traduzidas (texto original) e, como
    # referência do próprio modelo, nas reviews em inglês com rótulo fraco.
    cache_multi = args.output / "multilingue_cache.parquet"
    alvo_multi = pd.concat([
        outras.loc[outras["route"].eq("translate"), ["review_id", "text_for_inference"]],
        ingles.loc[ingles["weak_label"].notna(), ["review_id", "text_for_inference"]],
    ], ignore_index=True)
    multi = pd.read_parquet(cache_multi) if cache_multi.exists() else None
    if multi is None or not set(alvo_multi["review_id"]) <= set(multi["review_id"]):
        multi = inferir_multilingue(
            alvo_multi["text_for_inference"], args.multilingual_model,
            args.batch_size * 2, sentimento.fragmentos,
        )
        multi.insert(0, "review_id", alvo_multi["review_id"].to_numpy())
        multi.to_parquet(cache_multi, index=False)
    outras = outras.merge(multi, on="review_id", how="left")
    ingles = ingles.merge(multi, on="review_id", how="left")

    # Avaliação no rótulo fraco.
    rotas = outras.set_index("review_id")
    trad = rotas[rotas["route"].eq("translate")]
    recuperadas = rotas[rotas["route"].eq("english")]
    linhas = []
    for conjunto, dados, metodo, coluna in [
        ("english", ingles, "siebert", "transformer_label"),
        ("english", ingles, "xlm_t_multilingual", "multilingual_label"),
        ("recovered_english", recuperadas, "siebert", "transformer_label"),
        ("translated", trad, "siebert_on_translation", "translated_label"),
        ("translated", trad, "siebert_on_original", "transformer_label"),
        ("translated", trad, "xlm_t_multilingual", "multilingual_label"),
    ]:
        linhas.append({"set": conjunto, "method": metodo,
                       **avaliar(dados["weak_label"], dados[coluna], args.bootstrap, rng)})
    avaliacao = pd.DataFrame(linhas)
    avaliacao.to_csv(args.output / "avaliacao_rotulo_fraco.csv", index=False)

    comparacoes = {
        "translation_vs_original": diferenca_pareada(
            trad["weak_label"], trad["translated_label"], trad["transformer_label"],
            args.bootstrap, rng,
        ),
        "translation_vs_multilingual": diferenca_pareada(
            trad["weak_label"], trad["translated_label"], trad["multilingual_label"],
            args.bootstrap, rng,
        ),
    }

    por_idioma = pd.DataFrame([
        {"language": idioma, "reviews": len(dados),
         **avaliar(dados["weak_label"], dados["translated_label"], args.bootstrap, rng)}
        for idioma, dados in trad.groupby("lid_language")
    ]).sort_values("reviews", ascending=False)
    por_idioma.to_csv(args.output / "avaliacao_por_idioma.csv", index=False)

    # Amostra para conferência manual da tradução.
    trad.reset_index().sample(min(40, len(trad)), random_state=args.seed)[[
        "review_id", "lid_language", "translation_logprob", "text_clean", "translation",
        "rating_valido", "translated_label",
    ]].to_csv(args.output / "amostra_traducoes.csv", index=False)

    colunas = [
        "review_id", "book_id", "rating_valido", "text_language", "lid_language", "lid_prob",
        "route", "nllb_source", "translation", "translation_logprob", "translation_segments",
        "translated_label", "translated_positive_prob", "translated_chunks",
        "multilingual_label", "multilingual_positive_prob",
    ]
    outras[colunas].to_parquet(args.output / "reviews_outros_idiomas.parquet", index=False)
    (args.output / "metricas.json").write_text(json.dumps({
        "label": "rótulo fraco: nota 1-2 = negative, 4-5 = positive, nota 3 excluída",
        "routes": outras["route"].value_counts().to_dict(),
        "translated_languages": trad["lid_language"].value_counts().to_dict(),
        "models": {
            "language_id": "fastText lid.176.ftz",
            "translation": args.translation_model,
            "classifier": args.transformer_model,
            "multilingual": args.multilingual_model,
        },
        "min_lid_prob": args.min_lid_prob,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "evaluation": avaliacao.round(4).to_dict(orient="records"),
        "paired_differences": {
            nome: {chave: round(valor, 4) for chave, valor in valores.items()}
            for nome, valores in comparacoes.items()
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Tradução concluída: {args.output}")
    print(avaliacao[["set", "method", "rows", "negatives", "coverage", "balanced_accuracy", "ci_low", "ci_high"]].round(3).to_string(index=False))
    print(json.dumps(comparacoes, indent=2))
    print(por_idioma[["language", "reviews", "rows", "negatives", "balanced_accuracy"]].head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
