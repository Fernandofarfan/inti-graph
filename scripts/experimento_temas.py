"""Experimento: separar primero por macro-tema (publicacion) y agrupar despues.

La verdad de practica tiene 3 temas y 13 narrativas. Un k-means global mezcla temas
y castiga B-cubed aunque cada narrativa quede bien. Este script prueba una jerarquia
de dos niveles:
  1. agrupar las publicaciones por el embedding de su titular;
  2. dentro de cada tema, agrupar los textos de contenido en k_i narrativas.

Mide B-cubed, origen de emergentes y el total automatico de narrativas.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import calibracion, embeddings, ingesta, narrativas, ruido as ruido_mod  # noqa: E402


def bcubed(verdad: dict, pred: dict) -> tuple[float, float, float]:
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


def origen_puntos(nar_verdad: pd.DataFrame, marco: pd.DataFrame, fechas: dict) -> tuple[float, dict]:
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
        t_guess = fechas.get(lab)
        if t_guess is None:
            salida[nid] = 0.0
            continue
        err = abs((t_guess - pd.Timestamp(fila["fecha_origen"])).total_seconds()) / 3600
        salida[nid] = 1.0 if err <= 3 else max(0.0, 1 - (err - 3) / 45)
    return float(np.mean(list(salida.values()))), salida


def fechas_onset(marco: pd.DataFrame, prim: pd.Series, frac: float = 0.10) -> dict:
    dia = np.floor(marco["norm"].map(prim).fillna(0.0).to_numpy()).astype(int)
    salida = {}
    for lab, s in marco.groupby("lab"):
        d = pd.Series(dia[s.index.to_numpy()])
        masa = d.value_counts().sort_index()
        d0 = int(masa[masa >= frac * masa.max()].index[0])
        salida[lab] = pd.Timestamp(s.loc[(d == d0).to_numpy(), "fecha"].min())
    return salida


def main():
    art = calibracion.construir(verbose=False)
    df = ingesta.cargar_comentarios(RAIZ / "datos" / "practica")
    pubs = pd.read_csv(RAIZ / "datos" / "practica" / "publicaciones.csv.gz", dtype=str,
                       keep_default_na=False)
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
    t0 = df["fecha"].min()
    prim = ((df.loc[~mask].groupby("norm")["fecha"].min() - t0).dt.total_seconds() / 86400)
    uniq["prim"] = prim.reindex(uniq["norm"]).fillna(0.0).to_numpy()

    # --- tema por publicacion
    Et = embeddings.embeddings_unicos(pubs["titular"].str.lower().tolist(),
                                      etiqueta="titulares", verbose=False)
    tema_pub = {}
    for kp in (3, 4, 5, 6, 7):
        lab = KMeans(kp, n_init=20, random_state=0).fit_predict(Et)
        sil = silhouette_score(Et, lab)
        tema_pub[kp] = (lab, sil)
        comp = Counter(lab)
        print(f"k_pub={kp}: sil={sil:.3f} tamanos={dict(sorted(comp.items()))}")
    del Et

    contenido = df.loc[~mask, ["comentario_id", "fecha", "norm", "publicacion_id", "_y"]].copy()
    contenido = contenido.reset_index(drop=True)
    contenido["peso"] = 1
    for kp, (lab, _) in tema_pub.items():
        mapa_pub = dict(zip(pubs["publicacion_id"], lab))
        tema_c = contenido["publicacion_id"].map(mapa_pub).fillna(-1).astype(int)
        for k_total in (13, 15, 18):
            # unicos por (tema, norm)
            marco = contenido[["norm", "publicacion_id"]].copy()
            marco["tema"] = tema_c
            w = marco.groupby(["tema", "norm"]).size().rename("peso").reset_index()
            pos_de = dict(zip(uniq["norm"], np.arange(len(uniq))))
            w["i"] = w["norm"].map(pos_de)
            k_temas = w.groupby("tema")["peso"].sum()
            cuota = k_temas / k_temas.sum()
            etiqueta = {}
            for tema, sub in w.groupby("tema"):
                if tema == -1:
                    k_i = max(1, int(round(cuota.get(tema, 0) * k_total)))
                else:
                    k_i = max(2, int(round(cuota.loc[tema] * k_total)))
                idx = sub["i"].to_numpy()
                lab_i = narrativas.agrupar(X[idx], min(k_i, len(idx)),
                                           peso=sub["peso"].to_numpy(float))
                for i, l in zip(idx, lab_i):
                    etiqueta[int(i)] = f"t{tema}_{int(l)}"
            base = contenido["norm"].map(dict(zip(uniq["norm"],
                                                  [etiqueta.get(int(i), "?")
                                                   for i in range(len(uniq))])))
            contenido["lab"] = base
            pred = dict(zip(contenido["comentario_id"], base))
            p, r, f1 = bcubed(verdad_nar, pred)
            fechas = fechas_onset(contenido, prim)
            o, det = origen_puntos(nar_verdad, contenido, fechas)
            pose = 0.9137
            total = 30 * (0.5 * f1 + 0.15 * 1.0 + 0.2 * pose + 0.15 * o)
            print(f"  kp={kp} k={k_total}: B3={f1:.4f} (P={p:.3f} R={r:.3f}) "
                  f"origen={o:.3f} -> {total:.2f} pts {det}")


if __name__ == "__main__":
    main()
