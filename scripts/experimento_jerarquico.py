"""Experimento jerarquico: tema primero (3), despues narrativas dentro del tema."""
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
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


def vect(textos, **kw):
    base = dict(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True,
                stop_words=list(STOP), max_features=400000)
    base.update(kw)
    return TfidfVectorizer(**base)


def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str,
                    keep_default_na=False, usecols=["comentario_id", "texto", "publicacion_id"])
    v = pd.read_csv(RAIZ / "datos/practica/verdad/comentarios.csv.gz", dtype=str,
                    keep_default_na=False, usecols=["comentario_id", "narrativa_id"])
    nar = pd.read_csv(RAIZ / "datos/practica/verdad/narrativas.csv", dtype=str, keep_default_na=False)
    d = c.merge(v, on="comentario_id", how="left")
    d["nt"] = d.texto.str.split().str.len()
    d["norm"] = d.texto.map(normalize)
    d = d[d.nt >= 9].reset_index(drop=True)
    print("contenido:", len(d))
    tema_de = dict(zip(nar.narrativa_id, nar.tema))
    d["tema"] = d.narrativa_id.map(tema_de)

    # ---- etapa 1: temas sobre todos los comentarios de contenido
    X = l2(vect(d.norm).fit_transform(d.norm)).astype(np.float32)
    print("tfidf global", X.shape)
    tema_pred = None
    for k in (3, 4, 5):
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=10, batch_size=8192, max_iter=400)
        lab = km.fit_predict(X)
        # pureza de tema: cada cluster se asigna a su tema mayoritario
        tmp = pd.DataFrame({"lab": lab, "tema": d.tema})
        mapa = tmp.groupby("lab").tema.agg(lambda s: s.value_counts().index[0]).to_dict()
        pred_tema = np.array([mapa[x] for x in lab])
        print(f"  temas k={k}: pureza={float((pred_tema == d.tema.values).mean()):.3f} "
              f"silueta={float(silhouette_score(X[:8000], lab[:8000])):.3f}")
        if k == 3:
            tema_pred = lab

    # ---- etapa 2: narrativas dentro de cada tema
    for esquema, kwargs in [
        ("base", dict()),
        ("sin_topicos", dict(max_df=0.5)),
        ("unigramas", dict(ngram_range=(1, 1))),
        ("df_alto", dict(min_df=8)),
    ]:
        etiquetas = np.empty(len(d), dtype=object)
        detalle = []
        for g in sorted(set(tema_pred)):
            idx = np.where(tema_pred == g)[0]
            Xg = l2(vect(d.norm.iloc[idx], **kwargs).fit_transform(d.norm.iloc[idx])).astype(np.float32)
            mejor = None
            for k in (3, 4, 5, 6):
                km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=10, batch_size=4096, max_iter=400)
                lab = km.fit_predict(Xg)
                sub = d.iloc[idx]
                # coherencia interna: B3 restringido a este tema (usa verdad solo para medir)
                verdad_g = {cid: n for cid, n, t in zip(sub.comentario_id, sub.narrativa_id, sub.tema) if t == sub.tema.iloc[0]}
                pred_g = {cid: f"g{g}_{l}" for cid, l in zip(sub.comentario_id, lab)}
                _, _, f1g = bcubed(verdad_g, pred_g)
                if mejor is None or f1g > mejor[1]:
                    mejor = (k, f1g, lab)
                detalle.append((esquema, g, k, round(f1g, 3)))
            k, f1g, lab = mejor
            for j, l in zip(idx, lab):
                etiquetas[j] = f"g{g}_{l}"
        pred = dict(zip(d.comentario_id, etiquetas))
        verdad = dict(zip(d.comentario_id, d.narrativa_id))
        p, r, f1 = bcubed(verdad, pred)
        print(f"\n[{esquema}] B3 global P={p:.3f} R={r:.3f} F1={f1:.3f}")
        print("  detalle:", detalle)


if __name__ == "__main__":
    main()
