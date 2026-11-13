"""Ajuste del origen de narrativas emergentes.

La verdad de practica muestra que las narrativas emergentes (L5, B4, H3) usan textos
cuya primera aparicion en el dataset es varios dias despues del inicio. Este script
prueba separar los clusters por "cohorte temporal" del texto y mide el efecto sobre
B-cubed y sobre el origen de emergentes (metrica identica a la del jurado).
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


def bcubed(verdad: dict, pred: dict) -> float:
    pares, tam_t, tam_p = Counter(), Counter(), Counter()
    for item, t in verdad.items():
        p = pred.get(item, f"__solo__{item}")
        pares[(p, t)] += 1
        tam_t[t] += 1
        tam_p[p] += 1
    n = len(verdad)
    prec = sum(c * c / tam_p[p] for (p, _), c in pares.items()) / n
    rec = sum(c * c / tam_t[t] for (_, t), c in pares.items()) / n
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def origen_puntos(nar_verdad: pd.DataFrame, marco: pd.DataFrame,
                  fechas: dict | None = None) -> tuple[float, dict]:
    """Replica la metrica del jurado: origen declarado del cluster mayoritario."""
    salida = {}
    if fechas is None:
        fechas = marco.groupby("lab")["fecha"].min().to_dict()
    for _, fila in nar_verdad.iterrows():
        if fila["emergente"] != "si":
            continue
        nid = fila["narrativa_id"]
        sub = marco[marco["_y"] == nid]
        if sub.empty:
            salida[nid] = 0.0
            continue
        lab = sub["lab"].value_counts().index[0]
        t_guess = fechas.get(lab)
        if t_guess is None:
            salida[nid] = 0.0
            continue
        t_true = pd.Timestamp(fila["fecha_origen"])
        err = abs((t_guess - t_true).total_seconds()) / 3600
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
    peso = uniq["peso"].to_numpy(float)
    E = embeddings.embeddings_unicos(uniq["norm"].tolist(), etiqueta="unicos", verbose=False)
    X = narrativas.escalar(E, art["w"])
    del E

    t0 = df["fecha"].min()
    prim_s = (df.loc[~mask].groupby("norm")["fecha"].min() - t0).dt.total_seconds() / 86400
    uniq["prim"] = prim_s.reindex(uniq["norm"]).fillna(0.0).to_numpy(dtype=np.float32)

    contenido = df.loc[~mask, ["comentario_id", "fecha", "norm", "_y"]].copy()
    contenido["prim"] = contenido["norm"].map(prim_s)
    C = Counter()

    for k in (13, 15):
        etiquetas = narrativas.agrupar(X, k, peso=peso)
        uniq["lab"] = etiquetas
        # cuota tardia por cluster
        tardia = {}
        for l, s in uniq.groupby("lab"):
            tardia[l] = float((s["peso"] * (s["prim"].to_numpy() >= 5)).sum()
                              / max(s["peso"].sum(), 1))
        contenido["lab_base"] = contenido["norm"].map(dict(zip(uniq["norm"], etiquetas)))
        for dias in (4, 5, 6):
            for T in (0.35, 0.5, 0.65, 1.1):
                lab = contenido["lab_base"].astype(str)
                split = contenido["prim"].to_numpy() >= dias
                cond = contenido["lab_base"].map(lambda l: tardia[l] >= T).to_numpy()
                nueva = np.where(split & cond, lab + "L", lab)
                contenido["lab"] = nueva
                pred = {c: f"n{l}" for c, l, m in
                        zip(contenido["comentario_id"], nueva, ~mask)}
                b3 = bcubed(verdad_nar, pred)
                fechas_min = contenido.groupby("lab")["fecha"].min().to_dict()
                o_min, _ = origen_puntos(nar_verdad, contenido, fechas_min)
                # origen por cohorte modal de primera aparicion
                prim_r = contenido["prim"].round().astype("Int64")
                fechas_mod = {}
                for l, sub in contenido.groupby("lab"):
                    d = prim_r.loc[sub.index].value_counts()
                    if d.empty:
                        continue
                    dmod = int(d.index[0])
                    m = sub[(prim_r.loc[sub.index] >= dmod - 1) & (prim_r.loc[sub.index] <= dmod + 1)]
                    fechas_mod[l] = m["fecha"].min()
                o_mod, det_mod = origen_puntos(nar_verdad, contenido, fechas_mod)
                pts_min = 0.5 * b3 * 30 + 0.15 * o_min * 30
                pts_mod = 0.5 * b3 * 30 + 0.15 * o_mod * 30
                print(f"  k={k} dias={dias} T={T:.2f}: B3={b3:.4f} | min: o={o_min:.3f} "
                      f"-> {pts_min:.2f} | modal: o={o_mod:.3f} -> {pts_mod:.2f} {det_mod}")
                del pred, nueva
        del etiquetas

    # esquema alternativo: cluster separado de textos tardios
    print("\n=== cluster separado de textos tardios ===")
    tardios = uniq["prim"].to_numpy() >= 5
    for k2 in (4, 6, 8, 10):
        et = np.full(len(uniq), -1, dtype=np.int32)
        et[tardios] = narrativas.agrupar(X[tardios], k2, peso=peso[tardios])
        base = dict(zip(uniq["norm"], et))
        lab = contenido["norm"].map(base).fillna(-1).astype(int)
        contenido["lab"] = np.where(lab >= 0, "L" + lab.astype(str), "B")
        pred = {c: f"n{l}" for c, l, m in zip(contenido["comentario_id"],
                                              contenido["lab"], ~mask)}
        b3 = bcubed(verdad_nar, pred)
        o, det = origen_puntos(nar_verdad, contenido)
        print(f"  k2={k2}: B3={b3:.4f} origen={o:.3f} B3+origen={0.5 * b3 * 30 + 0.15 * o * 30:.2f} "
              f"pts {det}")


if __name__ == "__main__":
    main()
