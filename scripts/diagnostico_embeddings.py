"""Diagnostico del espacio de embeddings: pureza kNN, techo con centroides oraculo y confusion."""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as l2

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

ABREV = {"q": "que", "xq": "porque", "x": "por", "tmb": "tambien", "pa": "para",
         "tdo": "todo", "d": "de"}
STOP = set()


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
    y = np.asarray(uniq.narrativa_id, dtype=object)
    w = uniq.peso.values.astype(float)
    E = np.load(RAIZ / "salida" / "embeddings_unicos.npz")["E"].astype(np.float32)
    print("embeddings", E.shape)

    print("\n=== kNN purity (embeddings) ===")
    nn = NearestNeighbors(n_neighbors=21, metric="cosine", n_jobs=1).fit(E)
    dist, idx = nn.kneighbors(E)
    for k in (1, 5, 10, 20):
        lab = y[idx[:, 1:k + 1]]
        coincide = (lab == y[:, None]).astype(float)
        print(f"  top-{k}: pureza ponderada = {float((coincide.mean(1) * w).sum() / w.sum()):.3f}")
    print("  sim vecino-1", round(float(1 - dist[:, 1].mean()), 3),
          "| p10", round(float(np.percentile(1 - dist[:, 1], 10)), 3))

    print("\n=== techo oraculo: centroides por narrativa ===")
    clases = np.unique(y)
    C = l2(np.vstack([E[y == cl].mean(0) for cl in clases]))
    S = E @ C.T
    pred = clases[S.argmax(1)]
    acc = float((pred == y).mean())
    accw = float(((pred == y) * w).sum() / w.sum())
    print("  exactitud oraculo (textos unicos):", round(acc, 3), "| ponderada:", round(accw, 3))
    # por narrativa
    for cl in clases:
        m = y == cl
        print(f"    {cl}: acc={float((pred[m]==cl).mean()):.3f} (n={int(m.sum())})")

    print("\n=== confusion kmeans k=13 ===")
    km = MiniBatchKMeans(n_clusters=13, random_state=0, n_init=10, batch_size=8192).fit_predict(E)
    df = pd.crosstab(pd.Series(km, name="cluster"), pd.Series(y, name="narrativa"))
    print(df.to_string())
    print("\npureza por cluster:")
    print((df.max(axis=1) / df.sum(axis=1)).round(3).to_string())


if __name__ == "__main__":
    main()
