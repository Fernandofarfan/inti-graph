"""Clasificador de ruido: transferencia por publicacion (simula temas nuevos)."""
import pathlib
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

RAIZ = pathlib.Path(__file__).resolve().parent.parent


def norm(t):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[@#]\w+", " ", t)
    t = re.sub(r"[^a-z0-9\u00f1\s]", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1", t)
    return " ".join(t.split())


def f1_bin(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return 2 * p * r / (p + r) if p + r else 0


def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/comentarios.csv.gz", dtype=str, keep_default_na=False)
    d = c.merge(v, on="comentario_id")
    d["norm"] = d.texto.map(norm)
    d["nt"] = d.texto.str.split().str.len()
    d["tiene_rep"] = (d.respuesta_a != "").astype(int)
    d["signos"] = d.texto.str.count(r"[?¿]")
    y = (d.narrativa_id == "ninguna").astype(int).values
    print("ruido:", y.mean().round(3))

    # texto enriquecido con marcas estructurales
    d["entrada"] = (d.norm + " len" + d.nt.clip(0, 15).astype(str)
                    + " rep" + d.tiene_rep.astype(str) + " q" + (d.signos > 0).astype(str))

    gkf = GroupKFold(5)
    oof = np.zeros(len(d))
    for tr, te in gkf.split(d, y, groups=d.publicacion_id):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.7, sublinear_tf=True,
                              max_features=400000)
        Xtr = vec.fit_transform(d.entrada.iloc[tr])
        Xte = vec.transform(d.entrada.iloc[te])
        clf = LogisticRegression(max_iter=1500, C=1.0, class_weight="balanced")
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]

    mejor = None
    for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        pred = (oof >= th).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        f1 = f1_bin(tp, fp, fn)
        print(f"  th={th}: marca={pred.sum()} P={tp/max(pred.sum(),1):.4f} R={tp/y.sum():.4f} F1={f1:.4f}")
        if mejor is None or f1 > mejor[1]:
            mejor = (th, f1)
    print("mejor:", mejor)

    # regla dura + clasificador
    print("\n=== regla dura (nt<=2 y sin respuesta) + clasificador ===")
    th = mejor[0]
    pred = (oof >= th).astype(int)
    duro = ((d.nt <= 2) & (d.tiene_rep == 0)).values
    for nombre, p in [("solo clf", pred), ("duro OR clf", ((pred == 1) | duro).astype(int))]:
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        print(f"  {nombre}: P={tp/max(int((p==1).sum()),1):.4f} R={tp/y.sum():.4f} F1={f1_bin(tp, fp, fn):.4f}")


if __name__ == "__main__":
    main()
