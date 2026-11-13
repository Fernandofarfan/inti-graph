"""Ajuste del detector de coordinacion: motivos, umbrales y agrupamiento.

Reutiliza las features por cuenta y evalua los 25 puntos de coordinacion contra la
verdad de practica. Sirve para elegir umbral, reglas y criterio de agrupamiento.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import coordinacion, ingesta  # noqa: E402


def b3_grupos(pred: dict, verdad: dict) -> float:
    """B-cubed sobre cuentas coordinadas verdaderas (las no predichas cuentan solas)."""
    if not pred:
        return 0.0
    precs, recs = [], []
    for i, gi in pred.items():
        pi = {j for j, g in pred.items() if g == gi}
        ti = {j for j, t in verdad.items() if t == verdad[i]}
        precs.append(len(pi & ti) / len(pi))
        recs.append(len(pi & ti) / len(ti))
    return 2 * (np.mean(precs) * np.mean(recs)) / (np.mean(precs) + np.mean(recs))


def evaluar(cuentas: set, grupos: dict, verdad: dict, nombre: str) -> float:
    coord = {i for i, k in verdad.items() if k}
    tp = len(cuentas & coord)
    fp = len(cuentas - coord)
    fn = len(coord - cuentas)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    b3 = b3_grupos(grupos, verdad)
    comb = 0.7 * f1 + 0.3 * b3
    print(f"  {nombre}: marcadas={len(cuentas)} P={p:.3f} R={r:.3f} F1={f1:.3f} "
          f"B3grupos={b3:.3f} -> {25 * comb:.2f} pts "
          f"{dict(Counter(verdad.get(i, '') for i in cuentas))}")
    return 25 * comb


def main():
    verdir = RAIZ / "datos" / "practica" / "verdad"
    df = ingesta.cargar_comentarios(RAIZ / "datos" / "practica")
    cu = ingesta.cargar_cuentas(RAIZ / "datos" / "practica")
    ver_cue = ingesta.leer_csv(verdir / "cuentas.csv.gz")
    verdad = dict(zip(ver_cue["cuenta_id"], ver_cue["campana_id"]))
    coord = {i for i, k in verdad.items() if k}

    feats = coordinacion.features_cuentas(df, cu)
    feats.to_csv(RAIZ / "salida" / "features_cuentas.csv", index=False, encoding="utf-8")
    clf = coordinacion.entrenar_modelo(feats, pd.Series(verdad))
    feats["prob"] = coordinacion.puntuar(feats, clf)

    print("=== distribucion de prob por campana ===")
    feats["campana"] = feats["cuenta_id"].map(verdad).fillna("")
    print(feats.groupby("campana").prob.describe()[["count", "mean", "50%", "min", "max"]]
          .round(3).to_string())

    for th in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        marcadas = set(feats.loc[feats.prob >= th, "cuenta_id"])
        evaluar(marcadas, {}, verdad, f"solo modelo th={th}")

    print("\n=== reglas individuales sobre las cuentas ===")
    reglas = {
        "antiguedad<7": feats.antiguedad < 7,
        "antiguedad<10": feats.antiguedad < 10,
        "noches>=0.35": feats.noches >= 0.35,
        "ratio_dup>=0.25": feats.ratio_dup >= 0.25,
        "ratio_dup>=0.15": feats.ratio_dup >= 0.15,
        "co_respuestas>=2": feats.co_respuestas >= 2,
        "co_respuestas>=3": feats.co_respuestas >= 3,
        "plantilla_rep>=3": feats.plantilla_rep >= 3,
        "score_plantilla<=20": feats.score_plantilla <= 20,
        "pubs>=12": feats.pubs >= 12,
        "siguiendo>400": feats.siguiendo > 400,
        "seguidores<30": feats.seguidores < 30,
    }
    for nombre, m in reglas.items():
        ids = set(feats.loc[m, "cuenta_id"])
        print(f"  {nombre}: {len(ids)} marcadas "
              f"{dict(Counter(verdad.get(i, '') for i in ids))}")

    print("\n=== combinaciones modelo + reglas ===")
    nuevos = set(feats.loc[feats.antiguedad < 7, "cuenta_id"])
    for th in (0.2, 0.3, 0.5):
        base = set(feats.loc[feats.prob >= th, "cuenta_id"])
        for extra, nombre in [(set(), "ninguna"),
                              (nuevos - base, "nuevas"),
                              (set(feats.loc[feats.co_respuestas >= 2, "cuenta_id"]) - base, "coresp")]:
            ids = base | extra
            # agrupamiento: C2 = modelo, C1/C3 = nuevas partidas por noche
            grupos = {}
            for i in ids:
                fila = feats.set_index("cuenta_id").loc[i]
                if i in nuevos:
                    grupos[i] = "C3" if fila["noches"] >= 0.35 else "C1"
                else:
                    grupos[i] = "C2"
            evaluar(ids, grupos, verdad, f"th={th}+{nombre}")

    print("\n=== umbral optimo solo modelo (barrido fino) ===")
    mejor = (0, -1)
    for th in np.arange(0.05, 0.95, 0.05):
        ids = set(feats.loc[feats.prob >= th, "cuenta_id"])
        grupos = {}
        for i in ids:
            if i in nuevos:
                grupos[i] = "C3" if feats.set_index("cuenta_id").loc[i, "noches"] >= 0.35 else "C1"
            else:
                grupos[i] = "C2"
        tp = len(ids & coord)
        fp = len(ids - coord)
        fn = len(coord - ids)
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        b3 = b3_grupos(grupos, verdad)
        pts = 25 * (0.7 * f1 + 0.3 * b3)
        if pts > mejor[1]:
            mejor = (th, pts)
        print(f"  th={th:.2f}: marcadas={len(ids)} P={p:.3f} R={r:.3f} F1={f1:.3f} "
              f"B3={b3:.3f} -> {pts:.2f} pts")
    print("mejor:", mejor)


if __name__ == "__main__":
    main()
