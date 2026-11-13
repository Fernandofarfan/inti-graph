"""Deteccion de coordinacion: features por cuenta, scoring supervisado y agrupamiento.

Combina tres familias de senales descritas por el desafio:
  - cuentas nuevas creadas para la operacion (edad < 7 dias);
  - copy-paste / plantillas repetidas (ratio de duplicacion + plantilla compartida);
  - actividad nocturna en rafaga y reciprocidad de respuestas.
La verdad de practica calibra el modelo, pero las features son estructurales:
no dependen del tema ni del vocabulario de la campana.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .comun import normalizar

FEATURES = [
    "n", "pubs", "nt", "noches", "hora_std", "rep_ratio", "seguidores", "siguiendo",
    "antiguedad", "ratio_dup", "usuario_len", "sufijo_digitos", "amigos_rep",
    "cob_amigos", "score_plantilla", "nt_std", "co_respuestas", "plantilla_rep",
]


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def features_cuentas(df: pd.DataFrame, cu: pd.DataFrame) -> pd.DataFrame:
    c = df
    ini = c["fecha"].min()
    c["hora"] = c["fecha"].dt.hour
    c["es_noche"] = c["hora"].between(1, 5)
    c["es_respuesta"] = c["respuesta_a"] != ""

    g = c.groupby("cuenta_id")
    f = pd.DataFrame({
        "n": g.size(),
        "pubs": g["publicacion_id"].nunique(),
        "nt": g["nt"].mean(),
        "nt_std": g["nt"].std().fillna(0.0),
        "noches": g["es_noche"].mean(),
        "hora_std": g["hora"].std().fillna(0.0),
        "rep_ratio": g["es_respuesta"].mean(),
        "unicos": g["norm"].nunique(),
    })
    f["ratio_dup"] = 1.0 - f["unicos"] / f["n"]

    cu_idx = cu.set_index("cuenta_id")
    f["seguidores"] = cu_idx["seguidores"].reindex(f.index).fillna(0).astype(float)
    f["siguiendo"] = cu_idx["siguiendo"].reindex(f.index).fillna(0).astype(float)
    f["antiguedad"] = ((ini - cu_idx["fecha_creacion"]).dt.total_seconds() / 86400).reindex(f.index)
    f["usuario_len"] = cu_idx["usuario"].str.len().reindex(f.index).fillna(0).astype(float)
    f["sufijo_digitos"] = (cu_idx["usuario"].str.extract(r"(\d+)$")[0]
                           .fillna("0").str.len().reindex(f.index).fillna(0).astype(float))

    # --- grafo de respuestas
    mapa = dict(zip(c["comentario_id"], c["cuenta_id"]))
    rep = c[c["respuesta_a"] != ""][["cuenta_id", "respuesta_a"]].copy()
    rep["padre"] = rep["respuesta_a"].map(mapa)
    rep = rep[rep["padre"].notna() & (rep["padre"] != rep["cuenta_id"])]
    amigos = defaultdict(set)
    for a, b in zip(rep["cuenta_id"], rep["padre"]):
        amigos[a].add(b)
        amigos[b].add(a)
    f["amigos_rep"] = pd.Series({i: len(v) for i, v in amigos.items()}).reindex(f.index).fillna(0)

    pub_de = c.groupby("cuenta_id")["publicacion_id"].apply(set).to_dict()
    cob = {}
    for u, vs in amigos.items():
        vals = [len(pub_de[u] & pub_de[x]) / max(1, min(len(pub_de[u]), len(pub_de[x])))
                for x in vs]
        cob[u] = float(np.mean(vals)) if vals else 0.0
    f["cob_amigos"] = pd.Series(cob).reindex(f.index).fillna(0)

    # --- repeticion de plantillas entre cuentas
    cuenta_por_plantilla = c.groupby("plantilla")["cuenta_id"].nunique()
    f["score_plantilla"] = c.groupby("cuenta_id")["plantilla"].apply(
        lambda s: float(np.mean([cuenta_por_plantilla[t] for t in s])))
    rep_plantilla = c.groupby(["cuenta_id", "plantilla"]).size()
    f["plantilla_rep"] = rep_plantilla.groupby("cuenta_id").max().reindex(f.index).fillna(0)

    # --- co-respuesta: padres contestados por >=2 cuentas distintas
    por_padre = rep.groupby("respuesta_a")["cuenta_id"].nunique()
    padres_multi = set(por_padre[por_padre >= 2].index)
    participantes = rep[rep["respuesta_a"].isin(padres_multi)]["cuenta_id"]
    f["co_respuestas"] = participantes.value_counts().reindex(f.index).fillna(0).astype(float)

    f = f.reset_index()
    return f


# --------------------------------------------------------------------------- #
# Modelo y reglas
# --------------------------------------------------------------------------- #
def entrenar_modelo(feats: pd.DataFrame, campana: pd.Series) -> RandomForestClassifier:
    y = campana.reindex(feats["cuenta_id"]).fillna("").ne("").astype(int).values
    X = feats[FEATURES].to_numpy(dtype=np.float32)
    clf = RandomForestClassifier(600, random_state=0, n_jobs=1, min_samples_leaf=2,
                                 class_weight="balanced_subsample")
    clf.fit(X, y)
    return clf


def puntuar(feats: pd.DataFrame, clf: RandomForestClassifier) -> np.ndarray:
    return clf.predict_proba(feats[FEATURES].to_numpy(dtype=np.float32))[:, 1]


def detectar(feats: pd.DataFrame, prob: np.ndarray, umbral: float = 0.30,
             reglas: bool = True) -> dict:
    """Devuelve cuenta -> motivo (string) para las cuentas marcadas.

    El modelo aporta el ranking; las reglas cubren casos estructurales que el modelo
    puede dejar pasar en un set nuevo (cuentas creadas para la operacion).
    """
    motivos = {}
    for cid, p in zip(feats["cuenta_id"], prob):
        if p >= umbral:
            motivos[cid] = f"modelo:{p:.2f}"

    if reglas:
        for r in feats.itertuples(index=False):
            cid = r.cuenta_id
            if cid in motivos:
                continue
            # cuenta recien creada con actividad de plantilla o nocturna
            if (r.antiguedad < 7 and r.n >= 3
                    and (r.ratio_dup >= 0.10 or r.noches >= 0.30)):
                motivos[cid] = "cuenta_nueva"
    return motivos


# --------------------------------------------------------------------------- #
# Agrupamiento en operaciones
# --------------------------------------------------------------------------- #
FEATURES_GRUPO = ["antiguedad", "noches", "hora_std", "ratio_dup", "co_respuestas",
                  "amigos_rep", "pubs", "n"]


def agrupar(motivos: dict | set, feats: pd.DataFrame, min_grupo: int = 5) -> dict:
    """Asigna una operacion (grupo) a cada cuenta marcada.

    Las campanas del desafio tienen dos regimenes claros: cuentas creadas para la
    operacion (varias, con ritmos distintos) y cuentas antiguas infiltradas. Se
    separan primero por antiguedad y, dentro de las nuevas, por comportamiento
    (nocturno / diurno, plantilla, reciprocidad) con k-means k=2.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    ids = list(motivos)
    if not ids:
        return {}
    sub = feats.set_index("cuenta_id").reindex(ids)
    nuevas = sub[sub["antiguedad"] < 30]
    veteranas = sub[sub["antiguedad"] >= 30]
    grupos: dict[str, str] = {}
    n = 0

    def nuevo_grupo() -> str:
        nonlocal n
        n += 1
        return f"OP{n:02d}"

    def asignar(miembros) -> bool:
        if len(miembros) < min_grupo:
            return False
        e = nuevo_grupo()
        for cid in miembros:
            grupos[cid] = e
        return True

    if len(nuevas) >= 2 * min_grupo:
        X = nuevas[FEATURES_GRUPO].to_numpy(dtype=np.float64)
        X = np.clip(StandardScaler().fit_transform(X), -5, 5)
        km = KMeans(2, n_init=20, random_state=0).fit(X)
        partes = [nuevas.index[km.labels_ == g] for g in range(2)]
        if not all(len(p) >= min_grupo for p in partes):
            partes = [nuevas.index]
    else:
        partes = [nuevas.index] if len(nuevas) else []

    for p in partes:
        asignar(list(p))
    if len(veteranas):
        asignar(list(veteranas.index))

    # cualquier cuenta que haya quedado sin grupo: operacion propia si es numerosa
    resto = [c for c in ids if c not in grupos]
    if len(resto) >= min_grupo:
        asignar(resto)
    return grupos
