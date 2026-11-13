"""Reparto de k entre macro-temas (minimo por tema) y efecto sobre B-cubed."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

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


def etiquetar(df, mask, uniq, X, temas, ks):
    mapa = dict(zip(uniq["norm"], np.arange(len(uniq))))
    sub = df.loc[mask, ["norm", "publicacion_id"]].copy()
    sub["tema"] = sub["publicacion_id"].map(temas).fillna(-1).astype(int)
    tab = sub.groupby(["norm", "tema"]).size().rename("w").reset_index()
    tab["i"] = tab["norm"].map(mapa)
    dom = tab.sort_values("w", ascending=False).drop_duplicates("norm", keep="first")
    et = np.empty(len(uniq), dtype=object)
    et[:] = ""
    for tema, s in dom.groupby("tema"):
        k = min(ks[tema], len(s))
        lab = narrativas.agrupar(X[s["i"].to_numpy()], k, peso=s["w"].to_numpy(float))
        for i, l in zip(s["i"].to_numpy(), lab):
            et[i] = f"t{tema}_{int(l)}"
    return et


def main():
    art = calibracion.construir(verbose=False)
    df = ingesta.cargar_comentarios(RAIZ / "datos" / "practica")
    verdad = pd.read_csv(RAIZ / "datos" / "practica" / "verdad" / "comentarios.csv.gz",
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
    temas, n = narrativas.temas_por_publicacion(df, ~mask, uniq, X)
    prim = narrativas.primera_aparicion(df, ~mask)
    corte = narrativas.corte_cohorte(df)
    contenido = df.loc[~mask]

    masa = (contenido.assign(tema=contenido["publicacion_id"].map(temas).fillna(-1))
            .groupby("tema").size())
    print("masa:", masa.to_dict())

    variantes = {}
    for total in (11, 12, 13, 14):
        variantes[f"prop{total}"] = {t: max(2, int(round(masa[t] / masa.sum() * total)))
                                     for t in masa.index}
        variantes[f"prop{total}_min4"] = {t: max(4, int(round(masa[t] / masa.sum() * total)))
                                          for t in masa.index}
    for kf in (3, 4, 5):
        variantes[f"fijo{kf}"] = {t: kf for t in masa.index}

    for nombre, ks in variantes.items():
        et = etiquetar(df, ~mask, uniq, X, temas, ks)
        lab_u = narrativas.etiquetas_cohorte(uniq, et, prim, corte)
        ets = pd.Series(lab_u, index=uniq["norm"]).reindex(df["norm"]).to_numpy(object)
        pos = pd.Series(np.array(["apoya"] * len(df), dtype=object))
        ets, pos = narrativas.heredar_respuestas(df, mask, ets, pos.to_numpy(object))
        pred = {c: l for c, l, m in zip(df["comentario_id"], ets, mask)
                if not m and isinstance(l, str)}
        p, r, f1 = bcubed(verdad_nar, pred)
        print(f"{nombre}: ks={ks} B3={f1:.4f} (P={p:.3f} R={r:.3f})")


if __name__ == "__main__":
    main()
