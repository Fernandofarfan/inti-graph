"""Diagnostico de la calidad de la particion en macro-temas por publicacion."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import embeddings, ingesta  # noqa: E402


def main():
    pubs = pd.read_csv(RAIZ / "datos" / "practica" / "publicaciones.csv.gz", dtype=str,
                       keep_default_na=False)
    df = ingesta.cargar_comentarios(RAIZ / "datos" / "practica")
    verdad = pd.read_csv(RAIZ / "datos" / "practica" / "verdad" / "comentarios.csv.gz",
                         dtype=str, keep_default_na=False)
    df["_y"] = df["comentario_id"].map(dict(zip(verdad["comentario_id"],
                                                verdad["narrativa_id"]))).fillna("ninguna")
    print("publicaciones sin comentarios:",
          int((~pubs["publicacion_id"].isin(df["publicacion_id"])).sum()))
    tema = df["_y"].str[0].where(df["_y"] != "ninguna")
    comp_pub = (df.assign(tema=tema).groupby(["publicacion_id", "tema"]).size()
                .unstack(fill_value=0))
    print("\n=== tema dominante por publicacion ===")
    dom = comp_pub.idxmax(axis=1)
    print(dom.value_counts().to_string())

    Et = embeddings.embeddings_unicos(pubs["titular"].str.lower().tolist(),
                                      etiqueta="titulares", verbose=False)
    for kp in (2, 3, 4):
        lab = KMeans(kp, n_init=20, random_state=0).fit_predict(Et)
        mapa = dict(zip(pubs["publicacion_id"], lab))
        marco = pd.crosstab(dom.reindex(pubs["publicacion_id"]).to_numpy(), lab)
        print(f"\n=== k_pub={kp}: tema real (filas) x cluster (columnas) ===")
        print(marco.to_string())

    # alternativa: promedio de embeddings de comentarios por publicacion
    print("\n=== embeddings promedio de comentarios ===")
    uniq = df.groupby("norm").size().rename("peso").reset_index()
    E = embeddings.embeddings_unicos(uniq["norm"].tolist(), etiqueta="unicos", verbose=False)
    idx = dict(zip(uniq["norm"], np.arange(len(uniq))))
    pesos = df.groupby(["publicacion_id", "norm"]).size().rename("w").reset_index()
    pub_ids = sorted(pesos["publicacion_id"].unique())
    pidx = {p: i for i, p in enumerate(pub_ids)}
    M = np.zeros((len(pub_ids), E.shape[1]), dtype=np.float32)
    for p, n, w in zip(pesos["publicacion_id"], pesos["norm"], pesos["w"]):
        M[pidx[p]] += E[idx[n]] * w
    M /= np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)
    for kp in (2, 3, 4, 5):
        lab = KMeans(kp, n_init=20, random_state=0).fit_predict(M)
        mapa = dict(zip(pub_ids, lab))
        marco = pd.crosstab(dom.reindex(pub_ids).to_numpy(), lab)
        print(f"\n=== comentarios k_pub={kp} ===")
        print(marco.to_string())


if __name__ == "__main__":
    main()
