"""Seleccion de k por tema y origen hibrido (cohorte tardia) en la jerarquia.

Configuración: 3 macro-temas por promedio de embeddings de publicacion, k por tema
(silueta / proporcional / fijo), corte de cohorte y dos estimadores de origen:
  - onset sobre todo el cluster;
  - onset sobre la cohorte de textos tardios cuando esa cohorte pesa >= 10%.
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


def onset(s: pd.DataFrame, frac=0.10):
    m = s.groupby("dia").size().sort_index()
    if m.empty:
        return None
    d0 = int(m[m >= frac * m.max()].index[0])
    return pd.Timestamp(s.loc[s["dia"] == d0, "fecha"].min())


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
    pos = dict(zip(uniq["norm"], np.arange(len(uniq))))

    pesos = (df.loc[~mask].groupby(["publicacion_id", "norm"]).size()
             .rename("w").reset_index())
    pesos = pesos[pesos["norm"] != ""]
    pub_ids = sorted(pesos["publicacion_id"].unique())
    pidx = {p: i for i, p in enumerate(pub_ids)}
    M = np.zeros((len(pub_ids), X.shape[1]), dtype=np.float32)
    for p, n, w in zip(pesos["publicacion_id"], pesos["norm"], pesos["w"]):
        M[pidx[p]] += X[pos[n]] * w
    M /= np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)
    kp = max((silhouette_score(M, KMeans(k, n_init=20, random_state=0).fit_predict(M)), k)
             for k in (2, 3, 4, 5, 6))[1]
    tema_pub = pd.Series(KMeans(kp, n_init=20, random_state=0).fit_predict(M), index=pub_ids)
    print("kp:", kp, "| tamanos:", dict(sorted(Counter(tema_pub).items())))

    t0 = df["fecha"].min()
    prim = ((df.loc[~mask].groupby("norm")["fecha"].min() - t0).dt.total_seconds() / 86400)
    contenido = df.loc[~mask, ["comentario_id", "fecha", "norm", "publicacion_id", "_y"]].copy()
    contenido = contenido.reset_index(drop=True)
    contenido["tema"] = contenido["publicacion_id"].map(tema_pub).fillna(-1).astype(int)
    contenido["prim"] = contenido["norm"].map(prim).fillna(0.0)
    contenido["dia"] = np.floor(contenido["prim"].to_numpy()).astype(int)
    w = contenido.groupby(["tema", "norm"]).size().rename("peso").reset_index()
    w["i"] = w["norm"].map(pos)
    masa = w.groupby("tema")["peso"].sum()
    corte = narrativas.corte_cohorte(df)
    print("corte cohorte:", round(corte, 2), "dias")

    def k_silueta(sub, rango=(2, 3, 4, 5, 6), muestra=6000):
        idx = sub["i"].to_numpy()
        Xs = X[idx]
        if len(idx) > muestra:
            rng = np.random.default_rng(0)
            sel = rng.choice(len(idx), muestra, replace=False)
            Xs = Xs[sel]
        mejor, mejor_k = -2.0, 2
        for k in rango:
            if k >= len(Xs):
                break
            lab = narrativas.agrupar(Xs, k)
            if len(set(lab.tolist())) < 2:
                continue
            s = float(silhouette_score(Xs, lab))
            if s > mejor:
                mejor, mejor_k = s, k
        return mejor_k

    modos = {
        "silueta": lambda: {t: k_silueta(sub) for t, sub in w.groupby("tema")},
        "prop12": lambda: {t: max(2, int(round(masa[t] / masa.sum() * 12))) for t in masa.index},
        "prop11": lambda: {t: max(2, int(round(masa[t] / masa.sum() * 11))) for t in masa.index},
        "fijo4": lambda: {t: 4 for t in masa.index},
    }

    for nombre, fn in modos.items():
        ks = fn()
        etiqueta = {}
        for tema, sub in w.groupby("tema"):
            lab_i = narrativas.agrupar(X[sub["i"].to_numpy()], min(ks[tema], len(sub)),
                                       peso=sub["peso"].to_numpy(float))
            for i, l in zip(sub["i"].to_numpy(), lab_i):
                etiqueta[int(i)] = f"t{tema}_{int(l)}"
        lab_u = np.array([etiqueta[int(i)] for i in range(len(uniq))], dtype=object)
        lab_u = narrativas.etiquetas_cohorte(uniq, lab_u, prim, corte)
        contenido["lab"] = contenido["norm"].map(dict(zip(uniq["norm"], lab_u)))
        p, r, f1 = bcubed(verdad_nar, dict(zip(contenido["comentario_id"], contenido["lab"])))

        # origen: onset global y onset sobre cohorte tardia si pesa >= 10%
        f_all, f_hib = {}, {}
        for lab, s in contenido.groupby("lab"):
            f_all[lab] = onset(s)
            tardio = s[s["prim"] >= corte]
            cuota = len(tardio) / len(s)
            f_hib[lab] = onset(tardio) if (cuota >= 0.10 and not tardio.empty) else f_all[lab]
        o_all, _ = origen_puntos(nar_verdad, contenido, f_all)
        o_hib, det = origen_puntos(nar_verdad, contenido, f_hib)
        base = 30 * (0.5 * f1 + 0.15 * 1.0 + 0.2 * 0.9137)
        print(f"{nombre}: ks={ks} nlab={contenido['lab'].nunique()} B3={f1:.4f} "
              f"(P={p:.3f} R={r:.3f}) origen_all={o_all:.3f} origen_hib={o_hib:.3f} "
              f"-> {base + 30 * 0.15 * o_hib:.2f} pts {det}")


if __name__ == "__main__":
    main()
