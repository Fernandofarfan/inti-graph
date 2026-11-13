"""Evaluar postura (apoya/rechaza) con marcadores sobre narraciones verdaderas."""
import pathlib
import re
import unicodedata
from collections import Counter

import pandas as pd

RAIZ = pathlib.Path(__file__).resolve().parent.parent


def norm(t):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"@\w+|#\w+|https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-z0-9\u00f1\s]", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1", t)
    return " ".join(t.split())


BASE = ["mentira", "verso", "falso", "fake", "nada que ver", "no es cierto", "trucho",
        "desmintio", "sin pruebas", "inventan", "dejen de", "no hubo", "nadie va a"]

EXT = BASE + [
    "no es verdad", "no fue", "no hay", "no existe", "no se robo", "no robaron", "nunca",
    "jamas", "en realidad", "lo que pasa es", "pero", "aunque", "al contrario",
    "abran los ojos", "lean bien", "informense", "hay que informarse", "no todo es",
    "mucho antes", "desde hace", "no tiene nada que ver", "otra vez con", "dejen de repetir",
    "eso es viejo", "ya lo desmintieron", "no la deben", "no nos engañan", "sin fundamento",
    "no es casualidad", "no es cierto", "no es lo que", "no es la mina", "no es el hospital",
    "no es el boleto", "no es culpa", "no son", "no estaba", "no estaba",
]


def f1_bin(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return 2 * p * r / (p + r) if p + r else 0


def evaluar(d, pred, nombre):
    f1s = []
    for cls in ("apoya", "rechaza"):
        tp = fp = fn = 0
        for t, p in zip(d.postura, pred):
            if t == cls and p == cls:
                tp += 1
            elif t != cls and p == cls:
                fp += 1
            elif t == cls and p != cls:
                fn += 1
        f1s.append(f1_bin(tp, fp, fn))
    print(f"  {nombre}: macroF1={(f1s[0]+f1s[1])/2:.4f} apoya={f1s[0]:.3f} rechaza={f1s[1]:.3f}")


def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/comentarios.csv.gz", dtype=str, keep_default_na=False)
    d = c.merge(v, on="comentario_id")
    d = d[d.narrativa_id != "ninguna"].reset_index(drop=True)
    d["n"] = d.texto.map(norm)
    print("comentarios de narrativa:", len(d))

    evaluar(d, ["rechaza" if any(m in t for m in BASE) else "apoya" for t in d.n], "base")
    evaluar(d, ["rechaza" if any(m in t for m in EXT) else "apoya" for t in d.n], "ext")

    # marcador de primera persona / afirmacion
    print("\n=== marcadores mas frecuentes por postura (lift) ===")
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(ngram_range=(1, 2), min_df=20, binary=True)
    X = cv.fit_transform(d.n)
    terms = cv.get_feature_names_out()
    for cls in ("apoya", "rechaza"):
        m = (d.postura == cls).values
        p = X[m].mean(0).A1
        q = X.mean(0).A1
        lift = p / (q + 1e-9)
        orden = [i for i in lift.argsort()[::-1] if p[i] > 0.06][:25]
        print(f"  {cls}: " + ", ".join(f"{terms[i]}({lift[i]:.2f})" for i in orden))

    print("\n=== ejemplos mal clasificados por base ===")
    pred = ["rechaza" if any(m in t for m in BASE) else "apoya" for t in d.n]
    err = d[[p != t for p, t in zip(pred, d.postura)]]
    for _, r in err.sample(min(12, len(err)), random_state=0).iterrows():
        print(f"  [{r.postura} vs pred] {r.texto[:90]}")


if __name__ == "__main__":
    main()
