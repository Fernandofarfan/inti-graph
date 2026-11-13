"""Prueba de criterios de agrupamiento de las cuentas marcadas (C1/C2/C3)."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import coordinacion, ingesta  # noqa: E402


def b3(pred, verdad):
    precs, recs = [], []
    for i, g in pred.items():
        pi = {j for j, h in pred.items() if h == g}
        ti = {j for j, t in verdad.items() if t == verdad[i]}
        precs.append(len(pi & ti) / len(pi))
        recs.append(len(pi & ti) / len(ti))
    return 2 * np.mean(precs) * np.mean(recs) / (np.mean(precs) + np.mean(recs))


def main():
    df = ingesta.cargar_comentarios(RAIZ / "datos" / "practica")
    cu = ingesta.cargar_cuentas(RAIZ / "datos" / "practica")
    v = ingesta.leer_csv(RAIZ / "datos" / "practica" / "verdad" / "cuentas.csv.gz")
    verdad = dict(zip(v["cuenta_id"], v["campana_id"]))
    feats = coordinacion.features_cuentas(df, cu)
    clf = coordinacion.entrenar_modelo(feats, pd.Series(verdad))
    feats["prob"] = coordinacion.puntuar(feats, clf)
    feats.to_csv(RAIZ / "salida" / "features_cuentas.csv", index=False, encoding="utf-8")
    sel = feats[feats.prob >= 0.30].copy()
    sel["camp"] = sel.cuenta_id.map(verdad).fillna("")
    print(len(sel), Counter(sel.camp))

    nuevos = set(sel.loc[sel.antiguedad < 7, "cuenta_id"])
    for th in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
        pred = {}
        for _, r in sel.iterrows():
            if r.cuenta_id in nuevos:
                pred[r.cuenta_id] = "C3" if r.noches >= th else "C1"
            else:
                pred[r.cuenta_id] = "C2"
        print(f"noches>={th}: B3={b3(pred, verdad):.4f}")

    cols = ["antiguedad", "noches", "hora_std", "ratio_dup", "co_respuestas",
            "amigos_rep", "pubs", "n"]
    X = np.clip(StandardScaler().fit_transform(sel[cols].astype(float)), -5, 5)
    for k in (2, 3, 4, 5):
        km = KMeans(k, n_init=20, random_state=0).fit(X)
        pred = dict(zip(sel.cuenta_id, km.labels_))
        comp = {g: dict(Counter(sel.camp[km.labels_ == g])) for g in range(k)}
        print(f"kmeans k={k}: B3={b3(pred, verdad):.4f} "
              f"sil={silhouette_score(X, km.labels_):.3f} {comp}")


if __name__ == "__main__":
    main()
