"""Experimento: fusión de clusters por firma de términos (Jaccard)."""
import sys, time, unicodedata, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from evaluacion.puntuar import bcubed  # noqa: E402

ABREV = {"q": "que", "xq": "porque", "x": "por", "tmb": "tambien", "pa": "para", "tdo": "todo", "d": "de"}
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

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True,
                          stop_words=list(STOP), max_features=400000)
    X = l2(vec.fit_transform(cont.norm))
    terms = np.array(vec.get_feature_names_out())
    truth = dict(zip(cont.comentario_id, y))
    print("matriz", X.shape)

    for k in [60, 100, 150]:
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=8192, max_iter=300)
        lab = km.fit_predict(X)
        print(f"k={k}")
        dfc = pd.DataFrame({"lab": lab, "y": y})
        print("  pureza", round(dfc.groupby("lab").y.apply(lambda s: s.value_counts(normalize=True).iloc[0]).mean(), 3))
        # firma: top 20 términos por tf-idf medio del cluster
        firmas = []
        for i in range(k):
            mask = lab == i
            if not mask.any():
                firmas.append(set())
                continue
            media = np.asarray(X[mask].mean(axis=0)).ravel()
            firmas.append(set(terms[np.argsort(-media)[:20]]))
        for umbral in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
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
                    if not firmas[i] or not firmas[j]:
                        continue
                    jac = len(firmas[i] & firmas[j]) / len(firmas[i] | firmas[j])
                    if jac >= umbral:
                        union(i, j)
            grupos = {}
            for i in range(k):
                grupos.setdefault(find(i), []).append(i)
            mapa = {}
            for g, mem in enumerate(grupos.values()):
                for m in mem:
                    mapa[m] = f"n{g:03d}"
            pred = dict(zip(cont.comentario_id, [mapa[l] for l in lab]))
            print(f"  jac>={umbral}: grupos={len(grupos)} bcubed={bcubed(truth, pred)[2]:.4f}")

if __name__ == "__main__":
    main()
