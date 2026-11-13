"""Ajuste de narrativas: k, feature temporal de primera aparicion y origen.

Carga una sola vez los embeddings cacheados y evalua varias configuraciones con la
misma metrica del jurado (B-cubed sobre contenido, postura macro-F1, origen de
emergentes). Sirve para elegir k y el peso de la senal temporal.

Uso:
    python scripts/ajustar_narrativas.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import calibracion, embeddings, ingesta, narrativas, ruido as ruido_mod  # noqa: E402


def bcubed(verdad: dict, pred: dict) -> tuple[float, float, float]:
    pares = Counter()
    tam_t = Counter()
    tam_p = Counter()
    for item, t in verdad.items():
        p = pred.get(item, f"__solo__{item}")
        pares[(p, t)] += 1
        tam_t[t] += 1
        tam_p[p] += 1
    n = len(verdad)
    prec = sum(c * c / tam_p[p] for (p, _), c in pares.items()) / n
    rec = sum(c * c / tam_t[t] for (_, t), c in pares.items()) / n
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def f1_bin(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def macro_postura(y: np.ndarray, pred: np.ndarray) -> float:
    f1s = []
    for cls in ("apoya", "rechaza"):
        tp = int(((y == cls) & (pred == cls)).sum())
        fp = int(((y != cls) & (pred == cls)).sum())
        fn = int(((y == cls) & (pred != cls)).sum())
        f1s.append(f1_bin(tp, fp, fn))
    return float(np.mean(f1s))


def puntaje_origen(nar_verdad: pd.DataFrame, marco: pd.DataFrame,
                   fechas_cluster: dict) -> tuple[float, dict]:
    """Replica la metrica del jurado: mayoria de cada emergente -> fecha del cluster."""
    salida = {}
    for _, fila in nar_verdad.iterrows():
        if fila["emergente"] != "si":
            continue
        nid = fila["narrativa_id"]
        sub = marco[marco["_y"] == nid]
        if sub.empty:
            salida[nid] = 0.0
            continue
        lab = sub["lab"].value_counts().index[0]
        t_guess = fechas_cluster.get(lab)
        t_true = pd.Timestamp(fila["fecha_origen"])
        err = abs((t_guess - t_true).total_seconds()) / 3600
        salida[nid] = 1.0 if err <= 3 else max(0.0, 1 - (err - 3) / 45)
    return float(np.mean(list(salida.values()))) if salida else 0.0, salida


def main():
    art = calibracion.construir(verbose=False)
    df = ingesta.cargar_comentarios(RAIZ / "datos" / "practica")
    verdad = pd.read_csv(RAIZ / "datos" / "practica" / "verdad" / "comentarios.csv.gz",
                         dtype=str, keep_default_na=False)
    nar_verdad = pd.read_csv(RAIZ / "datos" / "practica" / "verdad" / "narrativas.csv",
                             dtype=str, keep_default_na=False)
    df["_y"] = df["comentario_id"].map(dict(zip(verdad["comentario_id"],
                                                verdad["narrativa_id"]))).fillna("ninguna")
    df["_pos"] = df["comentario_id"].map(dict(zip(verdad["comentario_id"],
                                                  verdad["postura"]))).fillna("")

    mask = ruido_mod.predecir(df, art["modelo_ruido"])
    es_ruido = (df["_y"] == "ninguna").to_numpy()
    tp = int((mask & es_ruido).sum())
    fp = int((mask & ~es_ruido).sum())
    fn = int((~mask & es_ruido).sum())
    print(f"ruido predicho={int(mask.sum())} verdad={int(es_ruido.sum())} "
          f"F1={f1_bin(tp, fp, fn):.5f} (fp={fp}, fn={fn})")

    uniq = narrativas.textos_unicos(df, ~mask)
    print("textos unicos:", len(uniq))
    E = embeddings.embeddings_unicos(uniq["norm"].tolist(), etiqueta="unicos", verbose=False)
    Xbase = narrativas.escalar(E, art["w"])
    del E

    # --- primera aparicion de cada texto (dias desde el inicio del dataset)
    t0 = df["fecha"].min()
    prim = (df.loc[~mask].groupby("norm")["fecha"].min() - t0).dt.total_seconds() / 86400
    dias = prim.reindex(uniq["norm"]).fillna(0.0).to_numpy(dtype=np.float32)
    escala = max(float(np.percentile(dias, 99)), 1e-6)
    f_temp = (dias / escala)[:, None].astype(np.float32)

    verdad_nar = {c: y for c, y in zip(verdad["comentario_id"], verdad["narrativa_id"])
                  if y != "ninguna"}
    pos_verdad = {c: p for c, p in zip(verdad["comentario_id"], verdad["postura"]) if p}
    peso = uniq["peso"].to_numpy(float)

    postura_u = np.array(narrativas.puntuar_postura(uniq["norm"].tolist(), art["lexico"],
                                                    art["umbral_postura"]))
    postura_de = dict(zip(uniq["norm"], postura_u))
    postura = df["norm"].map(postura_de)
    y_post = np.array([pos_verdad.get(c, "") for c in df["comentario_id"]])
    m_post = y_post != ""
    macro = macro_postura(y_post[m_post], postura[m_post].to_numpy())
    print(f"postura macro-F1: {macro:.4f}")

    contenido = df.loc[~mask, ["comentario_id", "fecha", "_y"]].copy()
    contenido["_c"] = contenido.index.to_numpy()
    resultados = []
    for alpha in (0.0, 2.0, 8.0):
        for k in (13, 19, 22):
            if alpha == 0:
                X = Xbase
            else:
                X = np.hstack([Xbase, alpha * f_temp]).astype(np.float32)
                nrm = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.clip(nrm, 1e-9, None)
            etiquetas_u = narrativas.agrupar(X, k, peso=peso)
            idx = dict(zip(uniq["norm"], etiquetas_u))
            lab = df["norm"].map(idx)
            pred = {c: f"n{l}" for c, l in zip(df["comentario_id"], lab) if not pd.isna(l)}
            p, r, f1 = bcubed(verdad_nar, pred)
            contenido["lab"] = lab.reindex(contenido["_c"]).to_numpy()
            fechas = {l: s["fecha"].min() for l, s in contenido.groupby("lab")}
            o, det = puntaje_origen(nar_verdad, contenido, fechas)
            total = 0.5 * f1 * 30 + 0.15 * 1.0 * 30 + 0.2 * macro * 30 + 0.15 * o * 30
            resultados.append({"alpha": alpha, "k": k, "b3": f1, "origen": o, "total": total})
            print(f"  alpha={alpha:4.1f} k={k:2d} B3={f1:.4f} (P={p:.3f} R={r:.3f}) "
                  f"origen={o:.3f} -> {total:.2f} pts  "
                  + " ".join(f"{n}:{v:.2f}" for n, v in det.items()))
            del X, lab, pred, fechas, etiquetas_u, idx
    contenido.drop(columns=["lab"], inplace=True, errors="ignore")

    mejor = max(resultados, key=lambda t: t["total"])
    print(f"\nmejor: alpha={mejor['alpha']} k={mejor['k']} total={mejor['total']:.2f} "
          f"B3={mejor['b3']:.4f} origen={mejor['origen']:.3f}")
    (RAIZ / "salida" / "ajuste_narrativas.json").write_text(
        json.dumps(resultados, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
