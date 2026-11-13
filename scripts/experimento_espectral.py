"""Experimento: clustering espectral sobre el grafo kNN de textos unicos."""
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
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
    print("textos unicos:", len(uniq))

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True,
                          stop_words=list(STOP), max_features=400000)
    X = l2(vec.fit_transform(uniq.norm)).astype(np.float32)
    print("tfidf", X.shape)

    for nvec in (10, 20):
        for nc in (13, 16, 20, 26):
            t0 = time.time()
            sc = SpectralClustering(n_clusters=nc, affinity="nearest_neighbors",
                                    n_neighbors=nvec, assign_labels="kmeans",
                                    random_state=0, n_jobs=-1)
            try:
                lab = sc.fit_predict(X)
            except Exception as exc:
                print(f"nvec={nvec} nc={nc}: ERROR {exc}")
                continue
            pred = {cid: f"n{lab[norm2idx[nm]]}" for cid, nm in cid_norm.items()}
            p, r, f1 = bcubed(verdad, pred)
            tam = np.bincount(lab)
            print(f"nvec={nvec} nc={nc}: grupos={len(tam)} max={tam.max()} "
                  f"P={p:.3f} R={r:.3f} F1={f1:.3f} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
