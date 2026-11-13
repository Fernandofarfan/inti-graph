"""Jerarquia con embeddings: tema (3) y despues frames por tema."""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from evaluacion.puntuar import bcubed  # noqa: E402

ABREV = {"q": "que", "xq": "porque", "x": "por", "tmb": "tambien", "pa": "para",
         "tdo": "todo", "d": "de"}


def normalize(t):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[^a-z0-9\u00f1\s]", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1", t)
    return " ".join(ABREV.get(w, w) for w in t.split())


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
    asignar_tema = dict(zip(nar.narrativa_id, nar.tema))
    d["tema"] = d.narrativa_id.map(asignar_tema)

    uniq = (d.groupby("norm", as_index=False)
            .agg(narrativa_id=("narrativa_id", "first"), tema=("tema", "first"),
                 peso=("comentario_id", "size")))
    norm2idx = {x: i for i, x in enumerate(uniq.norm)}
    cid_norm = dict(zip(d.comentario_id, d.norm))
    verdad = dict(zip(d.comentario_id, d.narrativa_id))
    E = np.load(RAIZ / "salida" / "embeddings_unicos.npz")["E"].astype(np.float32)
    print("unicos", len(uniq), "embeddings", E.shape)

    def ev(lab, nombre):
        pred = {cid: f"n{lab[norm2idx[nm]]}" for cid, nm in cid_norm.items()}
        p, r, f1 = bcubed(verdad, pred)
        print(f"  {nombre}: grupos={len(np.unique(lab))} P={p:.3f} R={r:.3f} F1={f1:.3f}")
        return f1

    # --- etapa temas
    for k in (3, 4):
        lab = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=20, batch_size=8192).fit_predict(E)
        tab = pd.crosstab(pd.Series(lab, name="c"), pd.Series(uniq.tema.values, name="tema"))
        mapa = {c: tab.columns[tab.loc[c].argmax()] for c in tab.index}
        pred = np.array([mapa[x] for x in lab])
        print(f"\ntemas k={k}: pureza={float((pred == uniq.tema.values).mean()):.3f}")
        print(tab.to_string())
        if k == 3:
            lab_tema = lab

    # --- gating con tema verdadero (techo)
    print("\n=== techo: gating por tema verdadero + kmeans de frames ===")
    etiquetas = np.empty(len(uniq), dtype=object)
    for tema in uniq.tema.unique():
        idx = np.where(uniq.tema.values == tema)[0]
        for k in (3, 4, 5):
            lab = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=10, batch_size=4096).fit_predict(E[idx])
            tmp = etiquetas.copy()
            for j, l in zip(idx, lab):
                tmp[j] = f"{tema}_{l}"
            ev(tmp, f"tema={tema} k={k}")

    # --- gating con tema predicho (kmeans 3), eligiendo k por coherencia
    print("\n=== gating con tema predicho (k=3) + frames ===")
    for kframes in (4, 5):
        etiquetas = np.empty(len(uniq), dtype=object)
        for g in np.unique(lab_tema):
            idx = np.where(lab_tema == g)[0]
            lab = MiniBatchKMeans(n_clusters=kframes, random_state=0, n_init=10,
                                  batch_size=4096).fit_predict(E[idx])
            for j, l in zip(idx, lab):
                etiquetas[j] = f"g{g}_{l}"
        ev(etiquetas, f"tema predicho + kframes={kframes}")

    # --- publicacion: pureza de tema
    print("\n=== publicaciones vs tema ===")
    pt = d.groupby("publicacion_id").tema.nunique()
    print("publicaciones con un solo tema:", int((pt == 1).sum()), "de", len(pt))
    top = d.groupby("publicacion_id").tema.apply(lambda s: s.value_counts(normalize=True).iloc[0])
    print("pureza media por publicacion:", round(float(top.mean()), 3),
          "| >=90%:", int((top >= 0.9).sum()))


if __name__ == "__main__":
    main()
