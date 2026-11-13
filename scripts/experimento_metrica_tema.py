"""Metrica supervisada restringida a pares del mismo macro-tema.

La metrica global aprende sobre todo a separar temas (pares faciles). Aqui los
negativos se muestrean solo dentro del mismo tema, para afinar la separacion entre
narrativas que comparten vocabulario. Se evalua B-cubed y el origen con la logica
del pipeline.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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


def metrica(E, y, tema, peso, n_pares=60000, semilla=0):
    rng = np.random.default_rng(semilla)
    idx = {c: np.where(y == c)[0] for c in np.unique(y)}
    clases = [c for c in idx if len(idx[c]) >= 2]
    tema_de = {c: Counter(tema[idx[c]]).most_common(1)[0][0] for c in clases}
    por_tema = {}
    for c in clases:
        por_tema.setdefault(tema_de[c], []).append(c)
    pos, neg = [], []
    for _ in range(n_pares):
        c = clases[rng.integers(len(clases))]
        a, b = rng.choice(idx[c], 2, replace=False)
        pos.append((a, b))
    for _ in range(n_pares):
        t = list(por_tema)[rng.integers(len(por_tema))]
        cs = por_tema[t]
        if len(cs) < 2:
            continue
        c1, c2 = rng.choice(cs, 2, replace=False)
        neg.append((rng.choice(idx[c1]), rng.choice(idx[c2])))
    pares = np.array(pos + neg)
    etiqueta = np.array([1] * len(pos) + [0] * len(neg))
    perm = rng.permutation(len(pares))
    pares, etiqueta = pares[perm], etiqueta[perm]
    a, b = pares[:, 0], pares[:, 1]
    cos = (E[a] * E[b]).sum(1, keepdims=True)
    F = np.hstack([E[a] * E[b], cos]).astype(np.float32)
    clf = LogisticRegression(max_iter=300, C=1.0).fit(F, etiqueta)
    w = np.clip(clf.coef_[0][:E.shape[1]], 0.0, None)
    return (w / w.mean()).astype(np.float32)


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
    X0 = narrativas.escalar(E, art["w"])
    temas, n_temas = narrativas.temas_por_publicacion(df, ~mask, uniq, X0)
    prim = narrativas.primera_aparicion(df, ~mask)
    corte = narrativas.corte_cohorte(df)
    contenido = df.loc[~mask]

    # etiqueta y tema por texto unico
    y_de = contenido.groupby("norm")["_y"].agg(lambda s: s.value_counts().index[0])
    tema_de = contenido.assign(tema=contenido["publicacion_id"].map(temas).fillna(-1))
    tema_txt = tema_de.groupby("norm")["tema"].agg(lambda s: s.value_counts().index[0])
    y = y_de.reindex(uniq["norm"]).fillna("ninguna").to_numpy(object)
    tema = tema_txt.reindex(uniq["norm"]).fillna(-1).to_numpy(int)

    w2 = metrica(E, y, tema, uniq["peso"].to_numpy(float))
    np.save(RAIZ / "salida" / "w_metrica_tema.npy", w2)
    print(f"metrica intra-tema: dims activas={int((w2 > 0.5).sum())} max={w2.max():.2f}")

    for nombre, w in (("global", art["w"]), ("intra-tema", w2)):
        X = narrativas.escalar(E, w)
        et = narrativas.agrupar_jerarquico(df, ~mask, uniq, X, 12, temas)
        lab_u = narrativas.etiquetas_cohorte(uniq, et, prim, corte)
        lab = contenido["norm"].map(dict(zip(uniq["norm"], lab_u)))
        p, r, f1 = bcubed(verdad_nar, dict(zip(contenido["comentario_id"], lab)))
        fechas = narrativas.origen_por_etiqueta(contenido, lab.to_numpy(), prim, corte)
        o, det = origen_puntos(nar_verdad, contenido.assign(lab=lab.to_numpy()), fechas)
        cols = contenido.copy()
        cols["lab"] = lab.to_numpy()
        comp = {}
        for nid in ("L5", "B4", "H3"):
            s = cols[cols["_y"] == nid]
            l = s["lab"].value_counts().index[0]
            ss = cols[cols["lab"] == l]
            comp[nid] = (l, round(float((ss["_y"] == nid).mean()), 2),
                         round(float((ss["norm"].map(prim) >= corte).mean()), 2))
        print(f"{nombre}: B3={f1:.4f} P={p:.3f} R={r:.3f} origen={o:.3f} "
              f"-> {30 * (0.5 * f1 + 0.15 + 0.2 * 0.9137 + 0.15 * o):.2f} pts {det} {comp}")


if __name__ == "__main__":
    main()
