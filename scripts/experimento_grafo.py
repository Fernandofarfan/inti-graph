"""Experimento: clustering por grafo de vecinos (TF-IDF) — textos unicos ponderados."""
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
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
    """Top-k vecinos por coseno calculado en bloques (memoria acotada)."""
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


class UnionFind:
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


def etiquetar(uf, n):
    raices = {}
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        r = uf.find(i)
        if r not in raices:
            raices[r] = len(raices)
        out[i] = raices[r]
    return out


def main():
    t0 = time.time()
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str,
                    keep_default_na=False, usecols=["comentario_id", "texto", "respuesta_a"])
    v = pd.read_csv(RAIZ / "datos/practica/verdad/comentarios.csv.gz", dtype=str,
                    keep_default_na=False, usecols=["comentario_id", "narrativa_id"])
    d = c.merge(v, on="comentario_id", how="left")
    d["nt"] = d.texto.str.split().str.len()
    d["norm"] = d.texto.map(normalize)
    cont_mask = d.nt >= 9
    print(f"carga {time.time()-t0:.1f}s | contenido={int(cont_mask.sum())}")

    uniq = (d[cont_mask].groupby("norm", as_index=False)
            .agg(narrativa_id=("narrativa_id", "first"), peso=("comentario_id", "size")))
    print("textos unicos:", len(uniq))
    norm2idx = {x: i for i, x in enumerate(uniq.norm)}

    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True,
                          stop_words=list(STOP), max_features=400000)
    X = l2(vec.fit_transform(uniq.norm)).astype(np.float32)
    print(f"tfidf {X.shape} {time.time()-t0:.1f}s nnz={X.nnz}")

    t0 = time.time()
    filas, cols, vals = topk_coseno(X, k=20)
    print(f"vecinos {len(filas)} aristas {time.time()-t0:.1f}s")

    # mapear verdad a los textos unicos (para diagnostico)
    y = np.asarray(uniq.narrativa_id, dtype=object)
    pesos_txt = uniq.peso.values.astype(float)

    # evaluacion sobre TODOS los comentarios de contenido
    cid_norm = dict(zip(d.comentario_id[cont_mask], d.norm[cont_mask]))
    verdad = dict(zip(d.comentario_id[cont_mask], d.narrativa_id[cont_mask]))

    for t in (0.35, 0.45, 0.55, 0.65, 0.75):
        uf = UnionFind(len(uniq))
        m = vals >= t
        for a, b in zip(filas[m], cols[m]):
            uf.union(int(a), int(b))
        lab = etiquetar(uf, len(uniq))
        grupos = len(set(lab.tolist()))
        tam = np.bincount(lab)
        pred = {cid: f"n{lab[norm2idx[nm]]}" for cid, nm in cid_norm.items()}
        p, r, f1 = bcubed(verdad, pred)
        # pureza ponderada (diagnostico)
        df = pd.DataFrame({"lab": lab, "y": y, "w": pesos_txt})
        pur_w = float((df.groupby("lab").y.apply(lambda s: s.value_counts(normalize=True).iloc[0]) *
                       df.groupby("lab").w.sum()).sum() / pesos_txt.sum())
        print(f"t={t}: grupos={grupos} max={tam.max()} singleton={int((tam==1).sum())} "
              f"B3 P={p:.3f} R={r:.3f} F1={f1:.3f} pureza_w={pur_w:.3f}")

    # referencia: cuantos textos por narrativa
    print("\ntextos unicos por narrativa:")
    print(uniq.narrativa_id.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
