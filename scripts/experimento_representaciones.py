"""Comparacion de representaciones para agrupar narrativas (B-cubed contra practica).

Prueba TF-IDF (palabras y caracteres), SVD, embeddings y combinaciones, con k fijo y
k elegido por silueta. Mide B-cubed sobre comentarios de contenido.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import calibracion, embeddings, ingesta, narrativas, ruido as ruido_mod  # noqa: E402


def bcubed(verdad: dict, pred: dict) -> float:
    pares = Counter()
    tam_t = Counter()
    tam_p = Counter()
    for item, t in verdad.items():
        p = pred.get(item, f"__solo__{item}")
        pares[(p, t)] += 1
        tam_t[t] += 1
        tam_p[p] += 1
    n = len(verdad)
    prec = sum(c * c / tam_p[p] for (p, _), c in pares.items()) / n
    rec = sum(c * c / tam_t[t] for (_, t), c in pares.items()) / n
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def evaluar(X, k, peso, uniq, df, mask, verdad_nar, nombre):
    etiquetas = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=10,
                                batch_size=8192, max_iter=500).fit_predict(
        X, sample_weight=peso)
    idx = dict(zip(uniq["norm"], etiquetas))
    lab = df["norm"].map(idx)
    pred = {c: f"n{l}" for c, l, m in zip(df["comentario_id"], lab, mask) if not m and not pd.isna(l)}
    f1 = bcubed(verdad_nar, pred)
    print(f"  {nombre} k={k}: B3={f1:.4f}")
    return f1


def main():
    art = calibracion.construir(verbose=False)
    df = ingesta.cargar_comentarios(RAIZ / "datos" / "practica")
    verdad = pd.read_csv(RAIZ / "datos" / "practica" / "verdad" / "comentarios.csv.gz",
                         dtype=str, keep_default_na=False)
    verdad_nar = {c: y for c, y in zip(verdad["comentario_id"], verdad["narrativa_id"])
                  if y != "ninguna"}
    mask = ruido_mod.predecir(df, art["modelo_ruido"])
    uniq = narrativas.textos_unicos(df, ~mask)
    peso = uniq["peso"].to_numpy(float)
    textos = uniq["norm"].tolist()

    E = embeddings.embeddings_unicos(textos, etiqueta="unicos", verbose=False)
    Xe = narrativas.escalar(E, art["w"])
    del E

    for k in (13, 15, 20):
        evaluar(Xe, k, peso, uniq, df, mask, verdad_nar, "embeddings+metrica")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True,
                          max_features=400000)
    Xt = vec.fit_transform(textos)
    print("  tfidf palabras:", Xt.shape, "nnz/doc", Xt.nnz / Xt.shape[0])
    for k in (13, 15, 20):
        evaluar(Xt, k, peso, uniq, df, mask, verdad_nar, "tfidf palabras")
    del Xt

    vec2 = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           max_features=300000, sublinear_tf=True)
    Xc = vec2.fit_transform(textos)
    print("  tfidf caracteres:", Xc.shape, "nnz/doc", Xc.nnz / Xc.shape[0])
    for k in (13, 15, 20):
        evaluar(Xc, k, peso, uniq, df, mask, verdad_nar, "tfidf caracteres")
    del Xc

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True,
                          max_features=400000)
    Xt = vec.fit_transform(textos)
    svd = TruncatedSVD(200, random_state=0)
    Xs = normalize(svd.fit_transform(Xt)).astype(np.float32)
    print("  varianza SVD:", round(float(svd.explained_variance_ratio_.sum()), 3))
    del Xt
    for k in (13, 15, 20):
        evaluar(Xs, k, peso, uniq, df, mask, verdad_nar, "svd200 tfidf")

    Xm = np.hstack([Xe, Xs * 1.0]).astype(np.float32)
    Xm = normalize(Xm)
    for k in (13, 15, 20):
        evaluar(Xm, k, peso, uniq, df, mask, verdad_nar, "emb+svd")


if __name__ == "__main__":
    main()
