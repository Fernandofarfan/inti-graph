"""Refinamiento del estimador de origen sobre la configuracion elegida.

Fija k=15 y el corte de cohorte en 5 dias (cuota tardia >= 35%) y compara variantes de
estimacion del origen por cluster: min de la cohorte modal, min de la cohorte modal
exacta, y min a partir del dia de emergencia.
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

K = 15
DIAS = 5
T = 0.35


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


def evaluar_origen(nar_verdad, marco, fechas, nombre):
    salida = {}
    for _, fila in nar_verdad.iterrows():
        if fila["emergente"] != "si":
            continue
        nid = fila["narrativa_id"]
        sub = marco[marco["_y"] == nid]
        lab = sub["lab"].value_counts().index[0]
        t_guess = fechas.get(lab)
        if t_guess is None:
            salida[nid] = 0.0
            continue
        err = abs((t_guess - pd.Timestamp(fila["fecha_origen"])).total_seconds()) / 3600
        salida[nid] = 1.0 if err <= 3 else max(0.0, 1 - (err - 3) / 45)
    print(f"  {nombre}: origen={np.mean(list(salida.values())):.3f} {salida}")
    return float(np.mean(list(salida.values())))


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
    uniq["prim"] = prim_s.reindex(uniq["norm"]).fillna(0.0).to_numpy()

    etiquetas = narrativas.agrupar(X, K, peso=peso)
    del X
    uniq["lab"] = etiquetas
    tardia = {l: float((s["peso"] * (s["prim"].to_numpy() >= DIAS)).sum() / s["peso"].sum())
              for l, s in uniq.groupby("lab")}

    contenido = df.loc[~mask, ["comentario_id", "fecha", "norm", "_y"]].copy()
    contenido["prim"] = contenido["norm"].map(prim_s)
    base = contenido["norm"].map(dict(zip(uniq["norm"], etiquetas)))
    split = (contenido["prim"].to_numpy() >= DIAS) & base.map(lambda l: tardia[l] >= T).to_numpy()
    contenido["lab"] = np.where(split, base.astype(str) + "L", base.astype(str))
    pred = {c: f"n{l}" for c, l in zip(contenido["comentario_id"], contenido["lab"])}
    b3 = bcubed(verdad_nar, pred)
    print(f"config k={K} dias={DIAS} T={T}: B3={b3:.4f}")

    pr = contenido["prim"]
    dia = np.floor(pr).astype(int)

    def fechas_min(sub_marco):
        return sub_marco.groupby("lab")["fecha"].min().to_dict()

    # A: min de toda la cohorte
    evaluar_origen(nar_verdad, contenido, fechas_min(contenido), "min global")

    # B: min de la cohorte modal +-1 dia
    f = {}
    for l, sub in contenido.groupby("lab"):
        d = dia.loc[sub.index].value_counts()
        if d.empty:
            continue
        dmod = int(d.index[0])
        m = sub[dia.loc[sub.index].between(dmod - 1, dmod + 1)]
        f[l] = m["fecha"].min()
    evaluar_origen(nar_verdad, contenido, f, "modal +-1")

    # C: min de la cohorte modal exacta
    f = {}
    for l, sub in contenido.groupby("lab"):
        d = dia.loc[sub.index].value_counts()
        if d.empty:
            continue
        dmod = int(d.index[0])
        m = sub[dia.loc[sub.index] == dmod]
        f[l] = m["fecha"].min()
    evaluar_origen(nar_verdad, contenido, f, "modal exacta")

    # D: min desde el dia modal en adelante
    f = {}
    for l, sub in contenido.groupby("lab"):
        d = dia.loc[sub.index].value_counts()
        if d.empty:
            continue
        dmod = int(d.index[0])
        m = sub[dia.loc[sub.index] >= dmod]
        f[l] = m["fecha"].min()
    evaluar_origen(nar_verdad, contenido, f, "desde modal")

    # E: dia modal con mayor salto de masa nueva (burst)
    f = {}
    for l, sub in contenido.groupby("lab"):
        d = dia.loc[sub.index]
        masa = d.value_counts().sort_index()
        if masa.empty:
            continue
        dif = masa.diff().fillna(masa.iloc[0])
        dmod = int(dif.idxmax())
        m = sub[dia.loc[sub.index].between(dmod, dmod + 1)]
        f[l] = m["fecha"].min()
    evaluar_origen(nar_verdad, contenido, f, "burst")

    # F: percentil 1 de la cohorte modal (robusto a outliers)
    f = {}
    for l, sub in contenido.groupby("lab"):
        d = dia.loc[sub.index].value_counts()
        if d.empty:
            continue
        dmod = int(d.index[0])
        m = sub[dia.loc[sub.index].between(dmod - 1, dmod + 1)]
        f[l] = m["fecha"].quantile(0.01)
    evaluar_origen(nar_verdad, contenido, f, "p01 modal +-1")

    # G: primer dia con masa >= frac * maximo diario
    for frac in (0.05, 0.10, 0.20, 0.30):
        f = {}
        for l, sub in contenido.groupby("lab"):
            d = dia.loc[sub.index]
            masa = d.value_counts().sort_index()
            if masa.empty:
                continue
            umbral = frac * masa.max()
            dia_ini = int(masa[masa >= umbral].index[0])
            m = sub[dia.loc[sub.index] == dia_ini]
            f[l] = m["fecha"].min()
        evaluar_origen(nar_verdad, contenido, f, f"primer dia masa>={frac:.2f}max")

    # detalle por cluster de los emergentes
    print("\n=== detalle ===")
    for nid in ("L5", "B4", "H3"):
        sub = contenido[contenido["_y"] == nid]
        lab = sub["lab"].value_counts().index[0]
        s2 = contenido[contenido["lab"] == lab]
        d = dia.loc[s2.index].value_counts()
        print(f"  {nid}: lab={lab} n={len(s2)} dias_modales={d.head(5).to_dict()} "
              f"min_lab={s2.fecha.min()}")


if __name__ == "__main__":
    main()
