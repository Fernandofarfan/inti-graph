"""Herencia de narrativa y postura por respuesta_a en respuestas cortas.

El desafio lo dice explicitamente: "Tal cual", "Exacto" o "Mentira" heredan la
narrativa del comentario al que responden (y la postura, invirtiendola si contradicen).
Aqui se mide el efecto sobre B-cubed y macro-F1 de postura.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import calibracion, ingesta, narrativas, pipeline, ruido as ruido_mod  # noqa: E402


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


def macro_postura(y, pred):
    f1s = []
    for cls in ("apoya", "rechaza"):
        tp = int(((y == cls) & (pred == cls)).sum())
        fp = int(((y != cls) & (pred == cls)).sum())
        fn = int(((y == cls) & (pred != cls)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def main():
    art = calibracion.construir(verbose=False)
    df = ingesta.cargar_comentarios(RAIZ / "datos" / "practica")
    verdad = pd.read_csv(RAIZ / "datos" / "practica" / "verdad" / "comentarios.csv.gz",
                         dtype=str, keep_default_na=False)
    y_de = dict(zip(verdad["comentario_id"], verdad["narrativa_id"]))
    pos_de = dict(zip(verdad["comentario_id"], verdad["postura"]))
    mask = ruido_mod.predecir(df, art["modelo_ruido"])
    narr = pipeline.predecir_narrativas(df, mask, art, verbose=False)

    lab = narr["etiquetas"].to_numpy(object)
    pos = narr["postura"].to_numpy(object)
    nt = df["nt"].to_numpy()
    pid = df["comentario_id"].to_numpy()
    resp = df["respuesta_a"].to_numpy()
    texto = df["norm"].to_numpy(object)
    idx = {c: i for i, c in enumerate(pid)}

    verdad_nar = {c: y for c, y in y_de.items() if y != "ninguna"}
    y_post = np.array([pos_de.get(c, "") for c in pid])
    m_post = y_post != ""

    def evaluar(lab, pos, nombre):
        pred = {pid[i]: lab[i] for i in range(len(lab)) if not mask[i] and isinstance(lab[i], str)}
        p, r, f1 = bcubed(verdad_nar, pred)
        macro = macro_postura(y_post[m_post], pos[m_post])
        print(f"{nombre}: B3={f1:.4f} (P={p:.3f} R={r:.3f}) postura={macro:.4f}")
        return f1, macro

    evaluar(lab, pos, "base")

    for umbral_nt in (3, 5, 7, 10):
        for pasadas in (1, 2):
            lab2 = lab.copy()
            pos2 = pos.copy()
            for _ in range(pasadas):
                cambios = 0
                for i in np.where(~mask)[0]:
                    if nt[i] > umbral_nt or not resp[i]:
                        continue
                    j = idx.get(resp[i])
                    if j is None or mask[j] or not isinstance(lab2[j], str):
                        continue
                    if lab2[i] == lab2[j] and pos2[i] == pos2[j]:
                        continue
                    lab2[i] = lab2[j]
                    if any(m in texto[i] for m in narrativas.MARCADORES_RECHAZO):
                        pos2[i] = "apoya" if pos2[j] == "rechaza" else "rechaza"
                    else:
                        pos2[i] = pos2[j]
                    cambios += 1
                if cambios == 0:
                    break
            f1, macro = evaluar(lab2, pos2, f"nt<={umbral_nt} pasadas={pasadas}")
            pts = 30 * (0.5 * f1 + 0.15 + 0.2 * macro + 0.15 * 0.896)
            print(f"   -> {pts:.2f} pts de narrativas (sin ruido/origen exactos)")


if __name__ == "__main__":
    main()
