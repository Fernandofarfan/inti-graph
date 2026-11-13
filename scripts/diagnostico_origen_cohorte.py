"""Estadisticas de cohorte por etiqueta para decidir el estimador de origen."""
from __future__ import annotations

import sys
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
    et = narrativas.agrupar_jerarquico(df, ~mask, uniq, X, 12, temas)
    prim = narrativas.primera_aparicion(df, ~mask)
    corte = narrativas.corte_cohorte(df)
    lab_u = narrativas.etiquetas_cohorte(uniq, et, prim, corte)
    contenido = df.loc[~mask].copy()
    contenido["lab"] = contenido["norm"].map(dict(zip(uniq["norm"], lab_u)))
    contenido["prim"] = contenido["norm"].map(prim).fillna(0.0)
    contenido["dia"] = np.floor(contenido["prim"]).astype(int)

    filas = []
    for lab, s in contenido.groupby("lab"):
        tardio = s[s["prim"] >= corte]
        masa = s.groupby("dia").size().sort_index()
        d0 = int(masa[masa >= 0.10 * masa.max()].index[0])
        ml = tardio.groupby("dia").size().sort_index()
        dl = int(ml[ml >= 0.10 * ml.max()].index[0]) if not ml.empty else None
        comp = s["_y"].value_counts()
        filas.append({
            "lab": lab, "n": len(s), "late": round(len(tardio) / len(s), 2),
            "d0": d0, "dL": dl,
            "late_peak": int(ml.max()) if not ml.empty else 0,
            "peak": int(masa.max()),
            "late_peak_share": round(float(ml.max()) / masa.max(), 3) if not ml.empty else 0,
            "dom": f"{comp.index[0]}:{comp.iloc[0] / len(s):.0%}",
        })
    print(pd.DataFrame(filas).sort_values("lab").to_string(index=False))


if __name__ == "__main__":
    main()
