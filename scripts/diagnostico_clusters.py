"""Composicion de clusters: cuota de textos 'tardios' y narrativa verdadera dominante.

Sirve para decidir si un cluster es una narrativa emergente (casi todos sus textos
aparecen por primera vez varios dias despues del inicio).
"""
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
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    etiquetas = narrativas.agrupar(X, k, peso=uniq["peso"].to_numpy(float))
    del X

    t0 = df["fecha"].min()
    prim = (df.loc[~mask].groupby("norm")["fecha"].min() - t0).dt.total_seconds() / 86400
    uniq["prim"] = prim.reindex(uniq["norm"]).fillna(0.0).to_numpy()
    uniq["lab"] = etiquetas

    cont = df.loc[~mask, ["comentario_id", "fecha", "norm", "_y"]].copy()
    cont["lab"] = cont["norm"].map(dict(zip(uniq["norm"], etiquetas)))
    cont["prim"] = cont["norm"].map(prim)

    print(f"k={k}")
    for l, s in cont.groupby("lab"):
        u = uniq[uniq.lab == l]
        tardio = float((u["peso"] * (u["prim"] >= 5)).sum() / max(u["peso"].sum(), 1))
        comp = Counter(s["_y"])
        n = len(s)
        top = ", ".join(f"{a}:{b/n:.0%}" for a, b in comp.most_common(3))
        print(f"  n{l:02d} n={n:6d} tardio={tardio:5.1%} min={str(s.fecha.min())[:16]} "
              f"max={str(s.fecha.max())[:16]} | {top}")

    print("\n=== narrativas emergentes: cluster mayoritario ===")
    for nid in ("L5", "B4", "H3"):
        s = cont[cont["_y"] == nid]
        l = s["lab"].value_counts().index[0]
        sub = cont[cont["lab"] == l]
        u = uniq[uniq.lab == l]
        tardio = float((u["peso"] * (u["prim"] >= 5)).sum() / max(u["peso"].sum(), 1))
        comp = Counter(sub["_y"])
        print(f"  {nid}: cluster n{l} tam={len(sub)} tardio={tardio:.1%} "
              f"min={sub.fecha.min()} composicion="
              + ", ".join(f"{a}:{b/len(sub):.0%}" for a, b in comp.most_common(4)))


if __name__ == "__main__":
    main()
