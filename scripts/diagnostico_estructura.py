"""Diagnostico estructural sobre textos UNICOS (con peso = frecuencia)."""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2
from sklearn.neighbors import NearestNeighbors

RAIZ = Path(__file__).resolve().parent.parent

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


def knn_purity(X, y, w, ks=(1, 5, 10, 20), n_neighbors=21, etiqueta=""):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=1).fit(X)
    dist, idx = nn.kneighbors(X)
    for k in ks:
        if k >= n_neighbors:
            continue
        lab = y[idx[:, 1:k + 1]]
        coincide = (lab == y[:, None]).astype(float)
        # pureza ponderada por frecuencia de cada texto
        pureza = float((coincide.mean(axis=1) * w).sum() / w.sum())
        print(f"  {etiqueta} top-{k}: pureza ponderada = {pureza:.3f}")
    print(f"  {etiqueta} sim vecino-1 = {float(dist[:, 1].mean()):.3f} | vecino-20 = {float(dist[:, n_neighbors-1].mean()):.3f}")


def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/comentarios.csv.gz", dtype=str, keep_default_na=False)
    pub = pd.read_csv(RAIZ / "datos/practica/publicaciones.csv.gz", dtype=str, keep_default_na=False)
    d = c.merge(v, on="comentario_id")
    d["nt"] = d.texto.str.split().str.len()
    d["norm"] = d.texto.map(normalize)

    print("=== 1. duplicados exactos ===")
    g = d.groupby("norm").narrativa_id.nunique()
    print("textos unicos:", len(g), "| textos con >1 narrativa:", int((g > 1).sum()),
          f"({(g > 1).mean():.2%})")
    dup = d.groupby("norm").size()
    print("docs que viven en textos duplicados:", int(dup[dup > 1].sum()), "de", len(d))

    cont_full = d[d.nt >= 9].copy()
    y = cont_full.narrativa_id.values

    # textos unicos ponderados por frecuencia (99.9% de los duplicados son puros)
    cont = (cont_full.groupby("norm", as_index=False)
            .agg(narrativa_id=("narrativa_id", "first"), peso=("comentario_id", "size")))
    w = cont.peso.values.astype(float)
    y = np.asarray(cont.narrativa_id, dtype=object)
    print("\n=== 2. kNN purity TF-IDF (textos unicos ponderados) ===")
    print("textos unicos de contenido:", len(cont))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.5, sublinear_tf=True,
                          stop_words=list(STOP), max_features=400000)
    X = l2(vec.fit_transform(cont.norm)).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=21, metric="cosine", algorithm="brute", n_jobs=1).fit(X)
    dist, idx = nn.kneighbors(X)
    for k in (1, 5, 10, 20):
        lab = y[idx[:, 1:k + 1]]
        coincide = (lab == y[:, None]).astype(float)
        print(f"  top-{k}: pureza ponderada = {float((coincide.mean(axis=1) * w).sum() / w.sum()):.3f}")
    print("  sim vecino-1:", round(float(dist[:, 1].mean()), 3),
          "| sim vecino-20:", round(float(dist[:, 20].mean()), 3),
          "| sim vecino-1 p90:", round(float(np.percentile(dist[:, 1], 90)), 3))

    print("\n=== 2b. textos mas frecuentes por narrativa ===")
    for cl in ("L1", "H2", "B1"):
        sub = cont_full[cont_full.narrativa_id == cl].norm.value_counts().head(6)
        print(f"  [{cl}]")
        for txt, n in sub.items():
            print(f"    {n:5d}  {txt}")

    print("\n=== 3. publicaciones vs narrativas ===")
    cc = cont_full.groupby("publicacion_id").narrativa_id.nunique()
    print("publicaciones:", len(pub), "| con contenido:", len(cc))
    print("narrativas por publicacion:", cc.describe()[["mean", "50%", "max"]].round(2).to_dict())
    print("distribucion:", cc.value_counts().sort_index().to_dict())
    top = cont_full.groupby("publicacion_id").narrativa_id.apply(lambda s: s.value_counts(normalize=True).iloc[0])
    print("pureza media por publicacion:", round(float(top.mean()), 3),
          "| publicaciones >=90% una narrativa:", int((top >= 0.9).sum()))

    print("\n=== 4. respuesta_a como grafo ===")
    rep = cont_full[cont_full.respuesta_a != ""]
    print("con respuesta_a:", len(rep), f"({len(rep)/len(cont_full):.1%})")
    if len(rep):
        m = rep.merge(cont_full[["comentario_id", "narrativa_id"]], left_on="respuesta_a",
                      right_on="comentario_id", suffixes=("", "_padre"))
        print("misma narrativa que el padre:", round(float((m.narrativa_id == m.narrativa_id_padre).mean()), 3),
              "de", len(m))

    print("\n=== 5. cuentas vs narrativas ===")
    cn = cont_full.groupby("cuenta_id").narrativa_id.nunique()
    print("cuentas con contenido:", len(cn), "| con 1 narrativa:", int((cn == 1).sum()),
          f"({(cn == 1).mean():.1%})")
    print("narrativas por cuenta:", cn.describe()[["mean", "50%", "max"]].round(2).to_dict())

    print("\n=== 6. tokens ===")
    n = d[d.narrativa_id == "ninguna"]
    print("ruido:", len(n), "| ruido nt<=8:", float((n.nt <= 8).mean()))
    print("contenido con respuesta_a:", float((cont_full.respuesta_a != "").mean()))
    print("regla captura ruido:", float(((n.nt <= 8) & (n.respuesta_a == "")).mean()))

    print("\n=== 7. kNN sobre terminos de lift alto ===")
    cv = TfidfVectorizer(ngram_range=(1, 1), min_df=5, max_df=0.5, sublinear_tf=True,
                         stop_words=list(STOP), max_features=400000)
    Xw = cv.fit_transform(cont.norm)
    best_lift = np.zeros(Xw.shape[1])
    for cl in np.unique(y):
        m = y == cl
        p = np.asarray(Xw[m].mean(0)).ravel()
        q = np.asarray(Xw.mean(0)).ravel()
        best_lift = np.maximum(best_lift, p / (q + 1e-9))
    for umbral in (2.0, 3.0, 4.0):
        sel = best_lift >= umbral
        Xr = l2(Xw[:, sel])
        n2 = NearestNeighbors(n_neighbors=11, metric="cosine", algorithm="brute", n_jobs=1).fit(Xr)
        dd, ii = n2.kneighbors(Xr)
        res = []
        for k in (1, 5, 10):
            lab = y[ii[:, 1:k + 1]]
            res.append(f"top-{k}={(lab == y[:, None]).mean():.3f}")
        print(f"  lift>={umbral} ({int(sel.sum())} terminos): " + " ".join(res))


if __name__ == "__main__":
    main()
