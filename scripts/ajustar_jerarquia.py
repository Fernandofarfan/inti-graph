"""Barrido de k para la jerarquia tema -> narrativa, con la logica exacta del pipeline."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import calibracion, embeddings, ingesta, narrativas, ruido as ruido_mod  # noqa: E402


def bcubed(verdad, pred):
    pares, tam_t, tam_p = Counter(), Counter(), Counter()
    for item, t in verdad.items():
        p = pred.get(item, f"__solo__{item}")
        pares[(p, t)] += 1
        tam_t[t] += 1
        tam_p[p] += 1
    n = len(verdad)
    prec = sum(c * c / tam_p[p] for (p, _), c in pares.items()) / n
    rec = sum(c * c / tam_t[t] for (_, t), c in pares.items()) / n
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def origen_puntos(nar_verdad, marco, fechas):
    salida = {}
    for _, fila in nar_verdad.iterrows():
        if fila["emergente"] != "si":
            continue
        nid = fila["narrativa_id"]
        sub = marco[marco["_y"] == nid]
        lab = sub["lab"].value_counts().index[0] if not sub.empty else None
        t = fechas.get(lab)
        if t is None:
            salida[nid] = 0.0
            continue
        err = abs((t - pd.Timestamp(fila["fecha_origen"])).total_seconds()) / 3600
        salida[nid] = 1.0 if err <= 3 else max(0.0, 1 - (err - 3) / 45)
    return float(np.mean(list(salida.values()))), salida


def main():
    art = calibracion.construir(verbose=False)
    df = ingesta.cargar_comentarios(RAIZ / "datos" / "practica")
    verdad = pd.read_csv(RAIZ / "datos" / "practica" / "verdad" / "comentarios.csv.gz",
                         dtype=str, keep_default_na=False)
    nar_verdad = pd.read_csv(RAIZ / "datos" / "practica" / "verdad" / "narrativas.csv",
                             dtype=str, keep_default_na=False)
    df["_y"] = df["comentario_id"].map(dict(zip(verdad["comentario_id"],
                                                verdad["narrativa_id"]))).fillna("ninguna")
    verdad_nar = {c: y for c, y in zip(verdad["comentario_id"], verdad["narrativa_id"])
                  if y != "ninguna"}
    mask = ruido_mod.predecir(df, art["modelo_ruido"])
    uniq = narrativas.textos_unicos(df, ~mask)
    E = embeddings.embeddings_unicos(uniq["norm"].tolist(), etiqueta="unicos", verbose=False)
    X = narrativas.escalar(E, art["w"])
    del E
    temas, n_temas = narrativas.temas_por_publicacion(df, ~mask, uniq, X)
    prim = narrativas.primera_aparicion(df, ~mask)
    corte = narrativas.corte_cohorte(df)
    contenido = df.loc[~mask]
    resultados = []
    for k in (8, 9, 10, 11, 12, 13, 14, 15, 16, 18):
        et = narrativas.agrupar_jerarquico(df, ~mask, uniq, X, k, temas)
        lab_u = narrativas.etiquetas_cohorte(uniq, et, prim, corte)
        lab = contenido["norm"].map(dict(zip(uniq["norm"], lab_u)))
        p, r, f1 = bcubed(verdad_nar, dict(zip(contenido["comentario_id"], lab)))
        fechas = narrativas.origen_por_etiqueta(contenido, lab.to_numpy(), prim, corte)
        o, det = origen_puntos(nar_verdad, contenido.assign(lab=lab.to_numpy()), fechas)
        total = 30 * (0.5 * f1 + 0.15 + 0.2 * 0.9137 + 0.15 * o)
        resultados.append((k, f1, o, total))
        print(f"k={k:2d}: nlab={lab.nunique()} B3={f1:.4f} P={p:.3f} R={r:.3f} "
              f"origen={o:.3f} -> {total:.2f} pts")
    print("mejor:", max(resultados, key=lambda t: t[3]))


if __name__ == "__main__":
    main()
