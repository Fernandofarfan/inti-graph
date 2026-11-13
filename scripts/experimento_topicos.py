"""NMF / LDA: topicos partes-based y asignacion por argmax."""
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from evaluacion.puntuar import bcubed  # noqa: E402

ABREV = {"q": "que", "xq": "porque", "x": "por", "tmb": "tambien", "pa": "para",
         "tdo": "todo", "d": "de"}
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
    print("unicos", len(uniq))

    vectores = {
        "tfidf": TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True,
                                 stop_words=list(STOP), max_features=200000),
        "tfidf_uni": TfidfVectorizer(ngram_range=(1, 1), min_df=5, max_df=0.3, sublinear_tf=True,
                                     stop_words=list(STOP), max_features=200000),
        "counts": CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, binary=True,
                                  stop_words=list(STOP), max_features=200000),
    }

    def ev(pesos, nombre):
        lab = pesos.argmax(1)
        pred = {cid: f"n{lab[norm2idx[nm]]}" for cid, nm in cid_norm.items()}
        p, r, f1 = bcubed(verdad, pred)
        print(f"  {nombre}: P={p:.3f} R={r:.3f} F1={f1:.3f}", flush=True)

    for vn, vec in vectores.items():
        X = vec.fit_transform(uniq.norm)
        print(f"\n{vn} {X.shape}", flush=True)
        for k in (13, 20, 26):
            t0 = time.time()
            nmf = NMF(n_components=k, init="nndsvda", solver="cd", beta_loss="frobenius",
                      max_iter=120, random_state=0)
            W = nmf.fit_transform(X)
            ev(W, f"NMF k={k} ({time.time()-t0:.0f}s)")
            if vn == "tfidf" and k == 13:
                terms = np.array(vec.get_feature_names_out())
                for i in range(k):
                    print(f"    t{i:02d}: " + " ".join(terms[np.argsort(-nmf.components_[i])[:8]]))

    # LDA sobre counts
    Xc = vectores["counts"].fit_transform(uniq.norm)
    for k in (13, 20):
        t0 = time.time()
        lda = LatentDirichletAllocation(n_components=k, random_state=0, max_iter=15,
                                        learning_method="online", batch_size=4096, n_jobs=1)
        W = lda.fit_transform(Xc)
        ev(W, f"LDA k={k} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
