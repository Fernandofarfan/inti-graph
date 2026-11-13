"""Ajuste de la regla de ruido: separar ruido real de comentarios cortos de contenido."""
import pathlib
import re
import unicodedata
from collections import Counter

import pandas as pd

RAIZ = pathlib.Path(__file__).resolve().parent.parent


def norm(t):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[@#]\w+", " ", t)
    t = re.sub(r"[^a-z0-9\u00f1\s]", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1", t)
    return " ".join(t.split())


def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/comentarios.csv.gz", dtype=str, keep_default_na=False)
    d = c.merge(v, on="comentario_id")
    d["nt"] = d.texto.str.split().str.len()
    d["nc"] = d.texto.str.len()
    d["n_norm"] = d.texto.map(lambda t: len(norm(t).split()))
    ruido = d.narrativa_id == "ninguna"

    print("=== comentarios con nt<=8 y sin respuesta ===")
    m = (d.nt <= 8) & (d.respuesta_a == "")
    print("total:", int(m.sum()), "| ruido:", int((m & ruido).sum()), "| contenido:", int((m & ~ruido).sum()))

    print("\n--- ejemplos de contenido corto sin respuesta ---")
    sub = d[m & ~ruido]
    for _, r in sub.sample(25, random_state=1).iterrows():
        print(f"  nt={r.nt} nc={r.nc} [{r.narrativa_id}/{r.postura}] {r.texto[:90]}")

    print("\n--- ejemplos de ruido ---")
    for _, r in d[m & ruido].sample(15, random_state=1).iterrows():
        print(f"  nt={r.nt} nc={r.nc} {r.texto[:90]}")

    print("\n=== separacion por longitud / tokens normalizados ===")
    for col in ("nt", "nc", "n_norm"):
        print(f"  {col}: ruido {d[ruido][col].describe()[['mean','50%','max']].round(1).to_dict()} "
              f"| corto-contenido {sub[col].describe()[['mean','50%','max']].round(1).to_dict()}")

    print("\n=== matriz de decision para umbrales ===")
    for t in range(1, 12):
        mm = (d.n_norm <= t) & (d.respuesta_a == "")
        tp = int((mm & ruido).sum())
        fp = int((mm & ~ruido).sum())
        print(f"  n_norm<={t}: cubre ruido={tp/ruido.sum():.4f} FP={fp}")

    print("\n=== que palabras tienen los cortos de contenido ===")
    print(Counter(w for t in sub.texto.map(norm) for w in t.split()).most_common(25))

    print("\n=== ruido: distribucion de palabras ===")
    print(Counter(w for t in d[ruido].texto.map(norm) for w in t.split()).most_common(25))

    # hay contenido con respuesta_a vacia y n_norm alto
    print("\ncontenido sin respuesta con n_norm<=3:", int(((d.n_norm <= 3) & (d.respuesta_a == "") & ~ruido).sum()))
    print("ruido con n_norm>=6:", int(((d.n_norm >= 6) & ruido).sum()))


if __name__ == "__main__":
    main()
