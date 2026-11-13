"""Experimento: Â¿quÃ© tan separables son las narrativas? (cota superior supervisada)."""
import sys, time, unicodedata, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import normalize

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

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
    t = re.sub(r"[^a-z0-9Ã±\s]", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1", t)
    return " ".join(ABREV.get(w, w) for w in t.split())

def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/comentarios.csv.gz", dtype=str, keep_default_na=False)
    d = c.merge(v, on="comentario_id")
    d["nt"] = d.texto.str.split().str.len()
    d["norm"] = d.texto.map(normalize)
    cont = d[d.nt >= 9].reset_index(drop=True)
    print("contenido", len(cont))

    for nombre, kwargs in [
        ("word12", dict(analyzer="word", ngram_range=(1, 2))),
        ("char35", dict(analyzer="char_wb", ngram_range=(3, 5))),
    ]:
        t0 = time.time()
        vec = TfidfVectorizer(min_df=5, max_df=0.5, sublinear_tf=True, max_features=300000, **kwargs)
        X = vec.fit_transform(cont.norm)
        y = np.asarray(cont.narrativa_id, dtype=object)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
        clf = LogisticRegression(max_iter=200, C=4.0, n_jobs=-1)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        print(nombre, "acc", round(accuracy_score(yte, pred), 4), "fit", round(time.time() - t0, 1), "s")
        # centroide supervisado
        Xn = normalize(X)
        clases = np.unique(y)
        cents = np.vstack([np.asarray(Xn[y == cl].mean(0)).ravel() for cl in clases])
        cents = normalize(cents)
        sims = Xn @ cents.T
        predc = clases[sims.argmax(1)]
        print(nombre, "centroide acc", round(accuracy_score(y, predc), 4))
        print(classification_report(y, predc, zero_division=0, digits=2))

if __name__ == "__main__":
    main()

