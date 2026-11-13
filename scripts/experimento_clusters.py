"""Experimentos de clustering de narrativas (se conserva como bitácora de I+D)."""
import sys, time, unicodedata, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from evaluacion.puntuar import bcubed  # noqa: E402

ABREV = {"q": "que", "xq": "porque", "x": "por", "tmb": "tambien", "pa": "para", "tdo": "todo",
         "x": "por", "d": "de", "q": "que", "xfa": "por favor"}
STOP = set("""a al algo con como de del el ella en es esa ese eso esta este esto la las le lo los mas me mi
muy no nos o para pero por que se si sin su sus te todo todos un una uno y ya ahi aca hay
""".split())

def normalize(t):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[^a-z0-9ñ\s]", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1", t)
    return " ".join(ABREV.get(w, w) for w in t.split())

def main():
    t0 = time.time()
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/comentarios.csv.gz", dtype=str, keep_default_na=False)
    d = c.merge(v, on="comentario_id")
    d["nt"] = d.texto.str.split().str.len()
    d["norm"] = d.texto.map(normalize)
    d["rep"] = d.respuesta_a != ""
    print("carga", round(time.time() - t0, 1), "s", d.shape)

    contenido = d[d.nt >= 9].copy()
    print("contenido", len(contenido), "verdad narrativa en contenido:",
          (contenido.narrativa_id != "ninguna").mean())
    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.4, sublinear_tf=True,
                          stop_words=list(STOP), max_features=200000)
    X = vec.fit_transform(contenido.norm)
    print("tfidf", X.shape, round(time.time() - t0, 1), "s")
    t0 = time.time()
    svd = TruncatedSVD(n_components=120, random_state=0)
    Xs = svd.fit_transform(X)
    print("svd", round(time.time() - t0, 1), "s", "var", round(svd.explained_variance_ratio_.sum(), 3))

    for k in [11, 13, 15]:
        t0 = time.time()
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=5, batch_size=4096)
        lab = km.fit_predict(Xs)
        pred = dict(zip(contenido.comentario_id, [f"k{x:02d}" for x in lab]))
        truth = {r.comentario_id: r.narrativa_id for r in d.itertuples() if r.narrativa_id != "ninguna"}
        print(f"k={k} bcubed={bcubed(truth, pred)[2]:.4f} ({round(time.time()-t0,1)}s)")
        if k == 13:
            contenido["lab"] = lab
            for l in range(k):
                terms = vec.get_feature_names_out()[np.argsort(-km.cluster_centers_[l] if False else -np.asarray(X[l].mean(0)).ravel())[:1]]
            # top terms por tfidf medio del cluster
            import scipy.sparse as sp
            Xc = X[lab == l]
            means = np.asarray(Xc.mean(axis=0)).ravel()
            top = vec.get_feature_names_out()[np.argsort(-means)[:10]]
            sub = contenido[lab == l]
            print(f"  k{l:02d} n={len(sub)} narr={sub.narrativa_id.value_counts().head(3).to_dict()} | {' '.join(top[:8])}")

if __name__ == "__main__":
    main()
