"""Experimento: barrido de representaciones para clustering de narrativas."""
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

def evaluar(nombre, X, truth, ks):
    print(f"--- {nombre} {X.shape}")
    for k in ks:
        if k > X.shape[0]:
            continue
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=8192, max_iter=300)
        lab = km.fit_predict(X)
        pred = {cid: f"k{l:03d}" for cid, l in zip(truth, lab)}
        print(f"  k={k} bcubed={bcubed(truth, pred)[2]:.4f}")

def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/comentarios.csv.gz", dtype=str, keep_default_na=False)
    d = c.merge(v, on="comentario_id")
    d["nt"] = d.texto.str.split().str.len()
    d["norm"] = d.texto.map(normalize)
    cont = d[d.nt >= 9].reset_index(drop=True)
    truth = {cid: n for cid, n in zip(cont.comentario_id, np.asarray(cont.narrativa_id, dtype=object))}
    print("docs", len(cont))

    configs = [
        ("maxdf05", dict(min_df=3, max_df=0.5)),
        ("maxdf10", dict(min_df=10, max_df=0.10)),
        ("maxdf05", dict(min_df=20, max_df=0.05)),
        ("maxdf03", dict(min_df=20, max_df=0.03)),
        ("maxdf02", dict(min_df=30, max_df=0.02)),
        ("maxdf01", dict(min_df=50, max_df=0.01)),
    ]
    for nombre, kw in configs:
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words=list(STOP),
                              max_features=400000, **kw)
        X = l2(vec.fit_transform(cont.norm))
        evaluar(nombre, X, truth, [13, 16, 20])

    # representación por términos raros: sólo las palabras de menor df
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=3, max_df=0.5, sublinear_tf=True,
                          stop_words=list(STOP), max_features=400000)
    X = vec.fit_transform(cont.norm)
    terms = np.array(vec.get_feature_names_out())
    dfreq = np.asarray((X > 0).sum(0)).ravel()
    for limite in [500, 1000, 2000, 3000]:
        raros = dfreq <= limite
        Xr = l2(X[:, raros])
        evaluar(f"raros<=df{limite} ({raros.sum()} terminos)", Xr, truth, [13, 16])

if __name__ == "__main__":
    main()
