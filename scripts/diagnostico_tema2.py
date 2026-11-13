"""Diagnostico: se separa H3 dentro de su macro-tema al subir k?"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import calibracion, embeddings, ingesta, narrativas, ruido as ruido_mod  # noqa: E402


def main():
    art = calibracion.construir(verbose=False)
    df = ingesta.cargar_comentarios(RAIZ / "datos" / "practica")
    verdad = pd.read_csv(RAIZ / "datos" / "practica" / "verdad" / "comentarios.csv.gz",
                         dtype=str, keep_default_na=False)
    df["_y"] = df["comentario_id"].map(dict(zip(verdad["comentario_id"],
                                                verdad["narrativa_id"]))).fillna("ninguna")
    mask = ruido_mod.predecir(df, art["modelo_ruido"])
    uniq = narrativas.textos_unicos(df, ~mask)
    E = embeddings.embeddings_unicos(uniq["norm"].tolist(), etiqueta="unicos", verbose=False)
    X = narrativas.escalar(E, art["w"])
    del E
    temas, n = narrativas.temas_por_publicacion(df, ~mask, uniq, X)
    print("temas:", n)

    # masa real por tema
    contenido = df.loc[~mask]
    tema_c = contenido["publicacion_id"].map(temas).fillna(-1).astype(int)
    print("composicion real por tema:")
    print(pd.crosstab(tema_c, contenido["_y"].str[0]).to_string())

    mapa = dict(zip(uniq["norm"], np.arange(len(uniq))))
    sub = df.loc[~mask, ["norm", "publicacion_id"]].copy()
    sub["tema"] = sub["publicacion_id"].map(temas).fillna(-1).astype(int)
    tab = sub.groupby(["norm", "tema"]).size().rename("w").reset_index()
    tab["i"] = tab["norm"].map(mapa)
    dom = tab.sort_values("w", ascending=False).drop_duplicates("norm", keep="first")

    for tema in sorted(dom["tema"].unique()):
        s = dom[dom["tema"] == tema]
        # narrativa verdadera dominante de cada texto unico
        y_de = (contenido.groupby("norm")["_y"]
                .agg(lambda x: x.value_counts().index[0]))
        y = s["norm"].map(y_de).fillna("?")
        print(f"\n### tema {tema}: {len(s)} textos, {int(s['w'].sum())} comentarios")
        for k in (3, 4, 5, 6, 7, 8):
            lab = narrativas.agrupar(X[s["i"].to_numpy()], k, peso=s["w"].to_numpy(float))
            comp = {}
            for l in range(k):
                comp[l] = dict(Counter(y[lab == l]).most_common(3))
            print(f"  k={k}: {comp}")


if __name__ == "__main__":
    main()
