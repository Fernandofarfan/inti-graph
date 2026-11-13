"""Experimento: clustering fino + fusión de centroides (sin usar la verdad)."""
import sys, time, unicodedata, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2
from sklearn.metrics import pairwise_distances

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from evaluacion.puntuar import bcubed  # noqa: E402

ABREV = {"q": "que", "xq": "porque", "x": "por", "tmb": "tambien", "pa": "para", "tdo": "todo",
         "d": "de", "xfa": "por favor"}
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
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/comentarios.csv.gz", dtype=str, keep_default_na=False)
    d = c.merge(v, on="comentario_id")
    d["nt"] = d.texto.str.split().str.len()
    d["norm"] = d.texto.map(normalize)
    cont = d[d.nt >= 9].reset_index(drop=True)
    y = np.asarray(cont.narrativa_id, dtype=object)
    print("contenido", len(cont), "unicos", cont.norm.nunique())

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True,
                          stop_words=list(STOP), max_features=400000)
    X = l2(vec.fit_transform(cont.norm))
    print("tfidf", X.shape)

    # 1) separación de centroides por verdad
    clases = np.unique(y)
    cents_truth = l2(np.vstack([np.asarray(X[y == cl].mean(0)).ravel() for cl in clases]))
    S = cents_truth @ cents_truth.T
    print("min sim entre narrativas distintas:", round((S - np.eye(len(clases)) * 9).max(), 3))

    truth = dict(zip(cont.comentario_id, y))

    # 2) k-means fino en sparse normalizado
    for k in [40, 80, 120]:
        t0 = time.time()
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=8192, max_iter=200)
        lab = km.fit_predict(X)
        cents = l2(km.cluster_centers_)
        # pureza
        df = pd.DataFrame({"lab": lab, "y": y})
        pureza = df.groupby("lab").y.apply(lambda s: s.value_counts(normalize=True).iloc[0]).mean()
        print(f"k={k} pureza media={pureza:.3f} tiempo={time.time()-t0:.1f}s")

        # fusionar centroides por similitud (union-find con umbral)
        for umbral in [0.30, 0.40, 0.50, 0.60, 0.70]:
            Sm = cents @ cents.T
            padre = list(range(k))
            def find(a):
                while padre[a] != a:
                    padre[a] = padre[padre[a]]
                    a = padre[a]
                return a
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    padre[rb] = ra
            for i in range(k):
                for j in range(i + 1, k):
                    if Sm[i, j] >= umbral:
                        union(i, j)
            grupos = {}
            for i in range(k):
                grupos.setdefault(find(i), []).append(i)
            mapa = {}
            for g, members in enumerate(grupos.values()):
                for m in members:
                    mapa[m] = f"n{g:03d}"
            pred = dict(zip(cont.comentario_id, [mapa[l] for l in lab]))
            f1 = bcubed(truth, pred)[2]
            print(f"   umbral={umbral} grupos={len(grupos)} bcubed={f1:.4f}")

if __name__ == "__main__":
    main()
