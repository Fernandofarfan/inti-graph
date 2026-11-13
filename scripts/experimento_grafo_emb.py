"""Grafo kNN sobre embeddings: componentes conexas y kNN mutuo a distintos umbrales."""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from evaluacion.puntuar import bcubed  # noqa: E402

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
    norm2idx = {x: i for i, x in enumerate(uniq.norm)}
    cid_norm = dict(zip(d.comentario_id[mask], d.norm[mask]))
    verdad = dict(zip(d.comentario_id[mask], d.narrativa_id[mask]))
    y = np.asarray(uniq.narrativa_id, dtype=object)
    w = uniq.peso.values.astype(float)
    E = np.load(RAIZ / "salida" / "embeddings_unicos.npz")["E"].astype(np.float32)

    K = 25
    nn = NearestNeighbors(n_neighbors=K, metric="cosine", n_jobs=1).fit(E)
    dist, idx = nn.kneighbors(E)
    sim = 1.0 - dist
    print("sim: min", round(float(sim[:,1].min()),3), "median", round(float(np.median(sim[:,1])),3))

    def ev(lab, nombre):
        pred = {cid: f"n{lab[norm2idx[nm]]}" for cid, nm in cid_norm.items()}
        p, r, f1 = bcubed(verdad, pred)
        tam = np.bincount(lab)
        # pureza ponderada
        purezas = []
        for g in np.unique(lab):
            m = lab == g
            vc = pd.Series(y[m]).value_counts()
            purezas.append((vc.iloc[0] / vc.sum(), w[m].sum()))
        pur = float(sum(pp * ww for pp, ww in purezas) / sum(ww for _, ww in purezas))
        print(f"  {nombre}: grupos={len(tam)} mayor={tam.max()} singletons={int((tam==1).sum())} "
              f"P={p:.3f} R={r:.3f} F1={f1:.3f} pureza={pur:.3f}")
        return f1

    print("\n=== componentes conexas (grafo kNN) ===")
    for t in (0.70, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94):
        uf = UF(len(uniq))
        m = sim[:, 1:] >= t
        filas = np.repeat(np.arange(len(uniq)), m.sum(1))
        cols = idx[:, 1:][m]
        for a, b in zip(filas, cols):
            uf.union(int(a), int(b))
        raiz = np.array([uf.find(i) for i in range(len(uniq))])
        _, lab = np.unique(raiz, return_inverse=True)
        ev(lab, f"t={t}")

    print("\n=== kNN mutuo ===")
    mutuo = np.zeros(sim.shape, dtype=bool)
    conjuntos = [set(map(int, fila[1:])) for fila in idx]
    for i in range(len(uniq)):
        for j in range(1, K):
            b = int(idx[i, j])
            if i in conjuntos[b]:
                mutuo[i, j] = True
    for t in (0.80, 0.85, 0.88, 0.90, 0.92):
        uf = UF(len(uniq))
        m = mutuo & (sim >= t)
        filas = np.repeat(np.arange(len(uniq)), m.sum(1))
        cols = idx[m]
        for a, b in zip(filas, cols):
            uf.union(int(a), int(b))
        raiz = np.array([uf.find(i) for i in range(len(uniq))])
        _, lab = np.unique(raiz, return_inverse=True)
        ev(lab, f"mutuo t={t}")


if __name__ == "__main__":
    main()
