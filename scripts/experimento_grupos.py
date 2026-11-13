"""Experimento: agrupar subgrupos puros (single-linkage estricto) y fusionarlos midiendo B3.

Objetivo: ver si existe un criterio de fusion que recupere las 13 narrativas.
"""
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
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


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
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

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True,
                          stop_words=list(STOP), max_features=400000)
    X = l2(vec.fit_transform(uniq.norm)).astype(np.float32)
    filas, cols, vals = topk_coseno(X, k=20)
    print("tfidf", X.shape, "aristas", len(filas))

    def evaluar(lab_txt, nombre):
        pred = {cid: f"n{lab_txt[norm2idx[nm]]}" for cid, nm in cid_norm.items()}
        p, r, f1 = bcubed(verdad, pred)
        print(f"    {nombre}: grupos={len(np.unique(lab_txt))} P={p:.3f} R={r:.3f} F1={f1:.3f}")
        return f1

    for t in (0.5, 0.55, 0.6):
        uf = UF(len(uniq))
        m = vals >= t
        for a, b in zip(filas[m], cols[m]):
            uf.union(int(a), int(b))
        raiz = np.array([uf.find(i) for i in range(len(uniq))], dtype=np.int32)
        raices, fine = np.unique(raiz, return_inverse=True)
        G = len(raices)
        tam = np.bincount(fine)
        print(f"\n=== t={t}: {G} subgrupos puros (max {tam.max()}, median {np.median(tam):.0f}) ===")

        # centroides de subgrupo (indicadora dispersa x X)
        import scipy.sparse as sp
        M = sp.csr_matrix((np.ones(len(uniq), dtype=np.float32),
                           (fine, np.arange(len(uniq)))), shape=(G, len(uniq)))
        C = l2(M @ X).toarray().astype(np.float32)
        # similitud entre subgrupos
        S = np.asarray(C @ C.T, dtype=np.float32)
        np.fill_diagonal(S, 1.0)
        D = np.clip(1.0 - S, 0, None)
        np.fill_diagonal(D, 0.0)
        D = ((D + D.T) / 2.0).astype(np.float64)
        cond = squareform(D, checks=False)

        for metodo in ("average", "complete", "ward"):
            Z = linkage(cond, method=metodo)
            for nc in (13, 16, 20, 26):
                g = fcluster(Z, t=nc, criterion="maxclust")
                evaluar(g[fine], f"{metodo} nc={nc}")

        # agrupamiento sobre docs dentro de cada subgrupo ya es exacto; probar grafo de subgrupos
        # por propagacion de etiquetas (Chinese Whispers)
        W = np.where(S >= 0.12, S, 0.0).astype(np.float32)
        lab = np.arange(G)
        rng = np.random.default_rng(0)
        for it in range(30):
            cambios = 0
            orden = rng.permutation(G)
            for g in orden:
                vecinos = np.where(W[g] > 0)[0]
                if len(vecinos) == 0:
                    continue
                pesos = {}
                for h in vecinos:
                    pesos[lab[h]] = pesos.get(lab[h], 0.0) + float(W[g, h])
                nueva = max(pesos.items(), key=lambda kv: kv[1])[0]
                if nueva != lab[g]:
                    lab[g] = nueva
                    cambios += 1
            if cambios == 0:
                break
        # compactar
        _, lab = np.unique(lab, return_inverse=True)
        print(f"    ChineseWhispers: grupos={len(np.unique(lab))}", end=" ")
        evaluar(lab[fine], "cw")


if __name__ == "__main__":
    main()
