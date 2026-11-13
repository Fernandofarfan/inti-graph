"""Metrica supervisada de 'misma narrativa' (aprende que dimensiones importan) + clustering."""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize as l2

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from evaluacion.puntuar import bcubed  # noqa: E402

ABREV = {"q": "que", "xq": "porque", "x": "por", "tmb": "tambien", "pa": "para",
         "tdo": "todo", "d": "de"}


def normalize(t):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[^a-z0-9\u00f1\s]", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1", t)
    return " ".join(ABREV.get(w, w) for w in t.split())


class UF:
    def __init__(self, n):
        self.p = np.arange(n, dtype=np.int64)

    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


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
    norm2idx = {x: i for i, x in enumerate(uniq.norm)}
    cid_norm = dict(zip(d.comentario_id[mask], d.norm[mask]))
    verdad = dict(zip(d.comentario_id[mask], d.narrativa_id[mask]))
    E = np.load(RAIZ / "salida" / "embeddings_unicos.npz")["E"].astype(np.float32)
    n = len(uniq)
    print("unicos", n)

    rng = np.random.default_rng(0)
    idx_por_clase = {cl: np.where(y == cl)[0] for cl in np.unique(y)}
    clases = list(idx_por_clase)
    pares_pos, pares_neg = [], []
    for _ in range(120000):
        cl = clases[rng.integers(len(clases))]
        a, b = rng.choice(idx_por_clase[cl], 2, replace=False)
        pares_pos.append((a, b))
    for _ in range(120000):
        c1, c2 = rng.choice(len(clases), 2, replace=False)
        a = rng.choice(idx_por_clase[clases[c1]])
        b = rng.choice(idx_por_clase[clases[c2]])
        pares_neg.append((a, b))
    pares = np.array(pares_pos + pares_neg)
    etiqueta = np.array([1] * len(pares_pos) + [0] * len(pares_neg))
    perm = rng.permutation(len(pares))
    pares, etiqueta = pares[perm], etiqueta[perm]
    print("pares", pares.shape)

    F = E[pares[:, 0]] * E[pares[:, 1]]
    # features extra: coseno, longitudes
    cos = (E[pares[:, 0]] * E[pares[:, 1]]).sum(1, keepdims=True)
    largo = np.abs(np.log1p(uniq.peso.values[pares[:, 0]]) -
                   np.log1p(uniq.peso.values[pares[:, 1]]))[:, None]
    Xp = np.hstack([F, cos, largo]).astype(np.float32)
    corte = int(0.8 * len(Xp))
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(Xp[:corte], etiqueta[:corte])
    prob = clf.predict_proba(Xp[corte:])[:, 1]
    print("AUC validacion:", round(float(roc_auc_score(etiqueta[corte:], prob)), 4))

    w = np.clip(clf.coef_[0][:E.shape[1]], 0, None)
    print("dimensiones con peso alto:", int((w > np.percentile(w, 95)).sum()),
          "| peso max", round(float(w.max()), 3))
    Esc = l2(E * np.sqrt(w + 1e-6)).astype(np.float32)

    def ev(lab, nombre):
        pred = {cid: f"n{lab[norm2idx[nm]]}" for cid, nm in cid_norm.items()}
        p, r, f1 = bcubed(verdad, pred)
        print(f"  {nombre}: grupos={len(np.unique(lab))} P={p:.3f} R={r:.3f} F1={f1:.3f}")

    for k in (13, 16, 20):
        lab = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=10, batch_size=8192).fit_predict(Esc)
        ev(lab, f"kmeans metrico k={k}")
    for mcs in (100, 300, 800):
        lab = HDBSCAN(min_cluster_size=mcs, min_samples=10, n_jobs=1).fit_predict(Esc)
        print(f"  (hdbscan mcs={mcs} ruido={int((lab==-1).sum())})", end=" ")
        lab2 = lab.copy()
        if (lab2 == -1).any():
            ids = np.array([x for x in np.unique(lab2) if x != -1])
            cent = np.vstack([Esc[lab2 == x].mean(0) for x in ids])
            ruido = lab2 == -1
            d2 = ((Esc[ruido][:, None, :] - cent[None, :, :]) ** 2).sum(-1)
            lab2[ruido] = ids[d2.argmin(1)]
        ev(lab2, f"hdbscan mcs={mcs}")

    # grafo de componentes con similitud metrica
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=25, metric="cosine", n_jobs=1).fit(Esc)
    dist, idx = nn.kneighbors(Esc)
    sim = 1 - dist
    for t in (0.6, 0.7, 0.8, 0.85):
        uf = UF(n)
        m = sim[:, 1:] >= t
        filas = np.repeat(np.arange(n), m.sum(1))
        cols = idx[:, 1:][m]
        for a, b in zip(filas, cols):
            uf.union(int(a), int(b))
        raiz = np.array([uf.find(i) for i in range(n)])
        _, lab = np.unique(raiz, return_inverse=True)
        ev(lab, f"componentes t={t}")


if __name__ == "__main__":
    main()
