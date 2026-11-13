"""Experimento: KMeans fino (alta pureza) + fusion por fuerza de conexion entre clusters."""
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from evaluacion.puntuar import bcubed  # noqa: E402

ABREV = {"q": "que", "xq": "porque", "x": "por", "tmb": "tambien", "pa": "para",
         "tdo": "todo", "d": "de", "xfa": "por favor"}
STOP = set("""a al algo con como de del el ella en es esa ese eso esta este esto la las le lo los mas me mi
muy no nos o para pero por que se si sin su sus te todo todos un una uno y ya ahi aca hay""".split())


def normalize(t):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[^a-z0-9\u00f1\s]", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1", t)
    return " ".join(ABREV.get(w, w) for w in t.split())


def topk_coseno(X, k=20, bs=384):
    n = X.shape[0]
    XT = X.T.tocsr()
    filas, cols, vals = [], [], []
    for start in range(0, n, bs):
        end = min(start + bs, n)
        S = (X[start:end] @ XT).tocsr()
        for r in range(S.shape[0]):
            i = start + r
            lo, hi = S.indptr[r], S.indptr[r + 1]
            if hi <= lo:
                continue
            ind = S.indices[lo:hi]
            dat = S.data[lo:hi]
            keep = ind != i
            ind, dat = ind[keep], dat[keep]
            if dat.size == 0:
                continue
            if dat.size > k:
                sel = np.argpartition(-dat, k)[:k]
                ind, dat = ind[sel], dat[sel]
            filas.append(np.full(ind.size, i, dtype=np.int32))
            cols.append(ind.astype(np.int32))
            vals.append(dat.astype(np.float32))
    return (np.concatenate(filas), np.concatenate(cols), np.concatenate(vals))


def afinidad_entre_clusters(lab, k, filas, cols, vals):
    """Fuerza de conexion simetrica entre clusters (flujo de aristas normalizado)."""
    i, j = lab[filas], lab[cols]
    m = i != j
    i, j, w = i[m], j[m], vals[m]
    C = np.zeros((k, k), dtype=np.float64)
    np.add.at(C, (i, j), w)
    salida = C.sum(axis=1, keepdims=True) + 1e-9
    P = C / salida
    S = (P + P.T) / 2.0
    return S


def evaluar(lab_txt, verdad, cid_norm, norm2idx, nombre):
    pred = {cid: f"n{lab_txt[norm2idx[nm]]}" for cid, nm in cid_norm.items()}
    p, r, f1 = bcubed(verdad, pred)
    tam = np.bincount(lab_txt)
    print(f"  {nombre}: grupos={len(tam)} max={tam.max()} P={p:.3f} R={r:.3f} F1={f1:.3f}")
    return f1


def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str,
                    keep_default_na=False, usecols=["comentario_id", "texto"])
    v = pd.read_csv(RAIZ / "datos/practica/verdad/comentarios.csv.gz", dtype=str,
                    keep_default_na=False, usecols=["comentario_id", "narrativa_id"])
    d = c.merge(v, on="comentario_id", how="left")
    d["nt"] = d.texto.str.split().str.len()
    d["norm"] = d.texto.map(normalize)
    mask = d.nt >= 9
    uniq = (d[mask].groupby("norm", as_index=False)
            .agg(narrativa_id=("narrativa_id", "first"), peso=("comentario_id", "size")))
    norm2idx = {x: i for i, x in enumerate(uniq.norm)}
    cid_norm = dict(zip(d.comentario_id[mask], d.norm[mask]))
    verdad = dict(zip(d.comentario_id[mask], d.narrativa_id[mask]))
    peso = uniq.peso.values.astype(float)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True,
                          stop_words=list(STOP), max_features=400000)
    X = l2(vec.fit_transform(uniq.norm)).astype(np.float32)
    print("tfidf", X.shape)
    t0 = time.time()
    filas, cols, vals = topk_coseno(X, k=20)
    print(f"vecinos {len(filas)} en {time.time()-t0:.1f}s")

    for kf in (80, 120, 160, 220):
        t0 = time.time()
        km = MiniBatchKMeans(n_clusters=kf, random_state=0, n_init=5, batch_size=8192, max_iter=400)
        fine = km.fit_predict(X, sample_weight=peso)
        S = afinidad_entre_clusters(fine, kf, filas, cols, vals)
        # distancia simetrica valida
        D = 1.0 - S / (S.max() + 1e-12)
        np.fill_diagonal(D, 0.0)
        D = (D + D.T) / 2.0
        Z = linkage(squareform(D, checks=False), method="average")
        print(f"\nkf={kf} ({time.time()-t0:.1f}s)")
        for nc in (11, 13, 15, 18, 22, 26):
            g = fcluster(Z, t=nc, criterion="maxclust")
            # remapear fine -> grupo
            mapa = g
            lab_txt = mapa[fine]
            evaluar(lab_txt, verdad, cid_norm, norm2idx, f"maxclust={nc}")


if __name__ == "__main__":
    main()
