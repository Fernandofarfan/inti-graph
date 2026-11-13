"""Postura con clasificador TF-IDF (validacion cruzada agrupada por narrativa)."""
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
    t = re.sub(r"@\w+|#\w+|https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-z0-9\u00f1\s]", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1", t)
    return " ".join(t.split())


def f1_bin(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return 2 * p * r / (p + r) if p + r else 0


def macro(y, pred):
    out = []
    for cls in ("apoya", "rechaza"):
        tp = int(((y == cls) & (pred == cls)).sum())
        fp = int(((y != cls) & (pred == cls)).sum())
        fn = int(((y == cls) & (pred != cls)).sum())
        out.append(f1_bin(tp, fp, fn))
    return (out[0] + out[1]) / 2, out


def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/comentarios.csv.gz", dtype=str, keep_default_na=False)
    d = c.merge(v, on="comentario_id")
    d = d[d.narrativa_id != "ninguna"].reset_index(drop=True)
    d["n"] = d.texto.map(norm)
    y = d.postura.values
    print("filas", len(d), "| balance", pd.Series(y).value_counts(normalize=True).round(3).to_dict())

    # CV agrupada por narrativa (mide generalizacion a narrativas nuevas)
    gkf = GroupKFold(5)
    oof = np.empty(len(d), dtype=object)
    for tr, te in gkf.split(d, y, groups=d.narrativa_id):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.6, sublinear_tf=True,
                              max_features=300000)
        Xtr = vec.fit_transform(d.n.iloc[tr])
        Xte = vec.transform(d.n.iloc[te])
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict(Xte)
    m, det = macro(y, oof)
    print(f"\nCV agrupada por narrativa: macroF1={m:.4f} apoya={det[0]:.4f} rechaza={det[1]:.4f}")

    # CV aleatoria (mide techo si los templates se repiten)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(d))
    folds = np.array_split(idx, 5)
    oof2 = np.empty(len(d), dtype=object)
    for k in range(5):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.6, sublinear_tf=True,
                              max_features=300000)
        Xtr = vec.fit_transform(d.n.iloc[tr])
        Xte = vec.transform(d.n.iloc[te])
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        clf.fit(Xtr, y[tr])
        oof2[te] = clf.predict(Xte)
    m2, det2 = macro(y, oof2)
    print(f"CV aleatoria: macroF1={m2:.4f} apoya={det2[0]:.4f} rechaza={det2[1]:.4f}")


if __name__ == "__main__":
    main()
