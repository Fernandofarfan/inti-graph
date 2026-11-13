"""Diagnostico del origen en la jerarquia tema -> narrativa.

Reproduce la configuracion ganadora del experimento (3 temas, k fijo 4 o reparto
proporcional) y muestra, para cada narrativa emergente, el cluster mayoritario, su
composicion real, la masa diaria y el dia de onset elegido.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import calibracion, embeddings, ingesta, narrativas, ruido as ruido_mod  # noqa: E402


def main():
    art = calibracion.construir(verbose=False)
    df = ingesta.cargar_comentarios(RAIZ / "datos" / "practica")
    verdad = pd.read_csv(RAIZ / "datos" / "practica" / "verdad" / "comentarios.csv.gz",
                         dtype=str, keep_default_na=False)
    nar_verdad = pd.read_csv(RAIZ / "datos" / "practica" / "verdad" / "narrativas.csv",
                             dtype=str, keep_default_na=False)
    df["_y"] = df["comentario_id"].map(dict(zip(verdad["comentario_id"],
                                                verdad["narrativa_id"]))).fillna("ninguna")
    mask = ruido_mod.predecir(df, art["modelo_ruido"])
    uniq = narrativas.textos_unicos(df, ~mask)
    E = embeddings.embeddings_unicos(uniq["norm"].tolist(), etiqueta="unicos", verbose=False)
    X = narrativas.escalar(E, art["w"])
    del E
    pos = dict(zip(uniq["norm"], np.arange(len(uniq))))

    pesos = (df.loc[~mask].groupby(["publicacion_id", "norm"]).size()
             .rename("w").reset_index())
    pesos = pesos[pesos["norm"] != ""]
    pub_ids = sorted(pesos["publicacion_id"].unique())
    pidx = {p: i for i, p in enumerate(pub_ids)}
    M = np.zeros((len(pub_ids), X.shape[1]), dtype=np.float32)
    for p, n, w in zip(pesos["publicacion_id"], pesos["norm"], pesos["w"]):
        M[pidx[p]] += X[pos[n]] * w
    M /= np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)
    tema_pub = pd.Series(KMeans(3, n_init=20, random_state=0).fit_predict(M), index=pub_ids)

    t0 = df["fecha"].min()
    prim = ((df.loc[~mask].groupby("norm")["fecha"].min() - t0).dt.total_seconds() / 86400)
    contenido = df.loc[~mask, ["comentario_id", "fecha", "norm", "publicacion_id", "_y"]].copy()
    contenido = contenido.reset_index(drop=True)
    contenido["tema"] = contenido["publicacion_id"].map(tema_pub).fillna(-1).astype(int)
    w = contenido.groupby(["tema", "norm"]).size().rename("peso").reset_index()
    w["i"] = w["norm"].map(pos)
    masa = w.groupby("tema")["peso"].sum()

    for modo, ks in (("fijo4", {t: 4 for t in masa.index}),
                     ("prop11", {t: max(2, int(round(masa[t] / masa.sum() * 11)))
                                 for t in masa.index})):
        print(f"\n########## {modo} {ks}")
        etiqueta = {}
        for tema, sub in w.groupby("tema"):
            lab_i = narrativas.agrupar(X[sub["i"].to_numpy()], min(ks[tema], len(sub)),
                                       peso=sub["peso"].to_numpy(float))
            for i, l in zip(sub["i"].to_numpy(), lab_i):
                etiqueta[int(i)] = f"t{tema}_{int(l)}"
        lab_u = np.array([etiqueta[int(i)] for i in range(len(uniq))], dtype=object)
        corte = narrativas.corte_cohorte(df)
        lab_u = narrativas.etiquetas_cohorte(uniq, lab_u, prim, corte)
        contenido["lab"] = contenido["norm"].map(dict(zip(uniq["norm"], lab_u)))
        dia = np.floor(contenido["norm"].map(prim).fillna(0.0).to_numpy()).astype(int)
        contenido["dia"] = dia

        for nid in ("L5", "B4", "H3"):
            fila = nar_verdad[nar_verdad["narrativa_id"] == nid].iloc[0]
            sub = contenido[contenido["_y"] == nid]
            lab = sub["lab"].value_counts().index[0]
            s2 = contenido[contenido["lab"] == lab]
            print(f"\n{nid} true={fila['fecha_origen']} lab={lab} n={len(s2)} "
                  f"comp={dict(Counter(s2['_y']).most_common(4))}")
            print("  masa diaria:", s2.groupby("dia").size().to_dict())
            m = s2.groupby("dia").size()
            d0 = int(m[m >= 0.10 * m.max()].index[0])
            print("  onset dia:", d0, "fecha:", s2.loc[s2["dia"] == d0, "fecha"].min(),
                  "| min:", s2["fecha"].min())
            print("  dias de", nid, ":", sub.groupby("dia").size().to_dict())


if __name__ == "__main__":
    main()
