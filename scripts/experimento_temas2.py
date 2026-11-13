"""Jerarquia tema -> narrativa usando el promedio de embeddings de cada publicacion.

Prueba la particion en macro-temas con el promedio de embeddings de los comentarios
(separa mejor que el titular), elige k_pub por silueta y evalua distintas formas de
repartir k entre temas. Incluye la separacion por cohorte temporal y la estimacion de
origen por onset, igual que el pipeline.
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


def origen_puntos(nar_verdad, marco, fechas):
    salida = {}
    for _, fila in nar_verdad.iterrows():
        if fila["emergente"] != "si":
            continue
        nid = fila["narrativa_id"]
        sub = marco[marco["_y"] == nid]
        lab = sub["lab"].value_counts().index[0] if not sub.empty else None
        t_guess = fechas.get(lab)
        if t_guess is None:
            salida[nid] = 0.0
            continue
        err = abs((t_guess - pd.Timestamp(fila["fecha_origen"])).total_seconds()) / 3600
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
    pos = dict(zip(uniq["norm"], np.arange(len(uniq))))

    # --- tema por publicacion (promedio de embeddings de comentarios)
    pesos = (df.loc[~mask].groupby(["publicacion_id", "norm"]).size()
             .rename("w").reset_index())
    pesos = pesos[pesos["norm"] != ""]
    pub_ids = sorted(pesos["publicacion_id"].unique())
    pidx = {p: i for i, p in enumerate(pub_ids)}
    M = np.zeros((len(pub_ids), X.shape[1]), dtype=np.float32)
    for p, n, w in zip(pesos["publicacion_id"], pesos["norm"], pesos["w"]):
        M[pidx[p]] += X[pos[n]] * w
    M /= np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)

    mejor_kp, mejor_sil = 2, -2
    for kp in (2, 3, 4, 5, 6):
        lab = KMeans(kp, n_init=20, random_state=0).fit_predict(M)
        s = float(silhouette_score(M, lab))
        print(f"k_pub={kp}: sil={s:.3f} tamanos={dict(sorted(Counter(lab).items()))}")
        if s > mejor_sil:
            mejor_kp, mejor_sil = kp, s
    print("kp por silueta:", mejor_kp)

    t0 = df["fecha"].min()
    prim = ((df.loc[~mask].groupby("norm")["fecha"].min() - t0).dt.total_seconds() / 86400)
    contenido = df.loc[~mask, ["comentario_id", "fecha", "norm", "publicacion_id", "_y"]].copy()
    contenido = contenido.reset_index(drop=True)

    for kp in (3,):
        tema_pub = pd.Series(KMeans(kp, n_init=20, random_state=0).fit_predict(M), index=pub_ids)
        tema_c = contenido["publicacion_id"].map(tema_pub).fillna(-1).astype(int)
        contenido["tema"] = tema_c
        w = (contenido.groupby(["tema", "norm"]).size().rename("peso").reset_index())
        w["i"] = w["norm"].map(pos)
        masa = w.groupby("tema")["peso"].sum()
        print("masa por tema:", masa.round(0).to_dict())

        asignaciones = {}
        for k_total in (11, 12, 13, 14, 15, 16, 17, 18):
            cuota = masa / masa.sum()
            asignaciones[f"prop k={k_total}"] = {t: max(2, int(round(cuota[t] * k_total)))
                                                 for t in masa.index}
        for kf in (3, 4, 5, 6):
            asignaciones[f"fijo k={kf}"] = {t: kf for t in masa.index}

        for nombre, ks in asignaciones.items():
            etiqueta = {}
            for tema, sub in w.groupby("tema"):
                k_i = min(ks[tema], len(sub))
                lab_i = narrativas.agrupar(X[sub["i"].to_numpy()], k_i,
                                           peso=sub["peso"].to_numpy(float))
                for i, l in zip(sub["i"].to_numpy(), lab_i):
                    etiqueta[int(i)] = f"t{tema}_{int(l)}"
            lab_u = np.array([etiqueta[int(i)] for i in range(len(uniq))], dtype=object)

            # cohorte temporal (misma logica del pipeline)
            corte = narrativas.corte_cohorte(df)
            lab_u = narrativas.etiquetas_cohorte(uniq, lab_u, prim, corte)
            base = contenido["norm"].map(dict(zip(uniq["norm"], lab_u)))
            contenido["lab"] = base
            pred = dict(zip(contenido["comentario_id"], base))
            p, r, f1 = bcubed(verdad_nar, pred)
            # origen por onset
            dia = np.floor(contenido["norm"].map(prim).fillna(0.0).to_numpy()).astype(int)
            fechas = {}
            for lab, s in contenido.groupby("lab"):
                d = pd.Series(dia[s.index.to_numpy()])
                m = d.value_counts().sort_index()
                d0 = int(m[m >= 0.10 * m.max()].index[0])
                fechas[lab] = pd.Timestamp(s.loc[(d == d0).to_numpy(), "fecha"].min())
            o, det = origen_puntos(nar_verdad, contenido, fechas)
            total = 30 * (0.5 * f1 + 0.15 * 1.0 + 0.2 * 0.9137 + 0.15 * o)
            print(f"  {nombre}: nlab={len(fechas)} B3={f1:.4f} (P={p:.3f} R={r:.3f}) "
                  f"origen={o:.3f} -> {total:.2f} pts {det}")


if __name__ == "__main__":
    main()
