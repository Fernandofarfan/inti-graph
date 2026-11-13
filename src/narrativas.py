"""Descubrimiento de narrativas: embeddings + metrica supervisada de pares + k-means.

La verdad del set de practica se usa solo para:
  1. aprender que dimensiones del embedding separan "misma narrativa" de "distinta";
  2. calibrar el lexico de postura.
No se usan titulos, temas ni etiquetas de la practica para predecir el set final.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize as l2

from .comun import VACIAS, MARCADORES_RECHAZO, normalizar


# Pictogramas y banderas: el titulo se muestra al jurado, no al evaluador automatico.
EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\uFE0F\u2190-\u21FF\u2B00-\u2BFF]"
)


# --------------------------------------------------------------------------- #
# Representacion
# --------------------------------------------------------------------------- #
def textos_unicos(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """Un registro por texto normalizado distinto, con su frecuencia."""
    sub = df.loc[mask]
    uniq = (sub.groupby("norm", as_index=False)
            .agg(plantilla=("plantilla", "first"), peso=("comentario_id", "size"),
                 ejemplo=("texto", "first")))
    uniq = uniq.sort_values("peso", ascending=False).reset_index(drop=True)
    return uniq


# --------------------------------------------------------------------------- #
# Metrica supervisada de pares
# --------------------------------------------------------------------------- #
def aprender_metrica(E: np.ndarray, y: np.ndarray, peso: np.ndarray,
                     n_pares: int = 60000, semilla: int = 0) -> np.ndarray:
    """Pesos por dimension que maximizan la separacion de pares de la misma narrativa."""
    rng = np.random.default_rng(semilla)
    idx = {c: np.where(y == c)[0] for c in np.unique(y)}
    clases = list(idx)
    pos, neg = [], []
    for _ in range(n_pares):
        c = clases[rng.integers(len(clases))]
        if len(idx[c]) < 2:
            continue
        a, b = rng.choice(idx[c], 2, replace=False)
        pos.append((a, b))
    for _ in range(n_pares):
        c1, c2 = rng.choice(len(clases), 2, replace=False)
        neg.append((rng.choice(idx[clases[c1]]), rng.choice(idx[clases[c2]])))
    pares = np.array(pos + neg)
    etiqueta = np.array([1] * len(pos) + [0] * len(neg))
    perm = rng.permutation(len(pares))
    pares, etiqueta = pares[perm], etiqueta[perm]

    a, b = pares[:, 0], pares[:, 1]
    cos = (E[a] * E[b]).sum(1, keepdims=True)
    F = np.hstack([E[a] * E[b], cos]).astype(np.float32)
    del a, b
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(F, etiqueta)
    w = np.clip(clf.coef_[0][:E.shape[1]], 0.0, None)
    if w.sum() <= 0:
        w = np.ones(E.shape[1], dtype=np.float32)
    return (w / w.mean()).astype(np.float32)


def escalar(E: np.ndarray, w: np.ndarray) -> np.ndarray:
    return l2(E * np.sqrt(w + 1e-6)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Eleccion de k
# --------------------------------------------------------------------------- #
def elegir_k(X: np.ndarray, rango=range(9, 25), muestra: int = 6000, semilla: int = 0) -> int:
    """k por silueta sobre una submuestra (con fallback a la cota inferior)."""
    rng = np.random.default_rng(semilla)
    n = X.shape[0]
    sel = rng.choice(n, size=min(muestra, n), replace=False)
    Xs = X[sel]
    mejor_k, mejor = rango.start, -2.0
    for k in rango:
        if k >= len(Xs):
            break
        lab = MiniBatchKMeans(n_clusters=k, random_state=semilla, n_init=5,
                              batch_size=4096, max_iter=300).fit_predict(Xs)
        if len(set(lab.tolist())) < 2:
            continue
        try:
            s = float(silhouette_score(Xs, lab))
        except ValueError:
            continue
        if s > mejor:
            mejor, mejor_k = s, k
    return mejor_k


def agrupar(X: np.ndarray, k: int, peso: np.ndarray | None = None,
            semilla: int = 0) -> np.ndarray:
    km = MiniBatchKMeans(n_clusters=k, random_state=semilla, n_init=10,
                         batch_size=8192, max_iter=500)
    return km.fit_predict(X, sample_weight=peso).astype(np.int32)


# --------------------------------------------------------------------------- #
# Jerarquia tema -> narrativa
# --------------------------------------------------------------------------- #
def temas_por_publicacion(df: pd.DataFrame, mask: np.ndarray, uniq: pd.DataFrame,
                          X: np.ndarray, rango=range(2, 7),
                          semilla: int = 0) -> tuple[pd.Series, int]:
    """Agrupa las publicaciones por el promedio de embeddings de sus comentarios.

    Cada tema del dataset gira alrededor de un conjunto de publicaciones; agrupar
    primero por publicacion evita que dos temas distintos terminen en el mismo
    cluster de narrativas (lo que castiga B-cubed). El numero de temas se elige por
    silueta sobre los centroides de publicacion.
    """
    from scipy import sparse
    from sklearn.cluster import KMeans

    sub = df.loc[mask, ["publicacion_id", "norm"]]
    sub = sub[sub["norm"] != ""]
    if sub.empty:
        return pd.Series(dtype=int), 1
    pos = pd.Series(np.arange(len(uniq)), index=uniq["norm"].to_numpy())
    conteo = sub.groupby(["publicacion_id", "norm"]).size().reset_index(name="w")
    conteo["i"] = conteo["norm"].map(pos)
    conteo = conteo[conteo["i"].notna()]
    pubs = sorted(conteo["publicacion_id"].unique())
    if len(pubs) < 3:
        return pd.Series(0, index=pubs), 1

    pidx = {p: i for i, p in enumerate(pubs)}
    A = sparse.csr_matrix(
        (conteo["w"].to_numpy(float),
         (conteo["publicacion_id"].map(pidx).to_numpy(), conteo["i"].to_numpy(int))),
        shape=(len(pubs), len(uniq)))
    M = np.asarray(A @ X, dtype=np.float32)
    M /= np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)
    del A

    mejor_k, mejor = 1, -2.0
    for k in rango:
        if k >= len(pubs):
            break
        lab = KMeans(k, n_init=20, random_state=semilla).fit_predict(M)
        if len(set(lab.tolist())) < 2:
            continue
        s = float(silhouette_score(M, lab))
        if s > mejor:
            mejor, mejor_k = s, k
    if mejor_k == 1:
        return pd.Series(0, index=pubs), 1
    lab = KMeans(mejor_k, n_init=20, random_state=semilla).fit_predict(M)
    return pd.Series(lab, index=pubs), mejor_k


def agrupar_jerarquico(df: pd.DataFrame, mask: np.ndarray, uniq: pd.DataFrame,
                       X: np.ndarray, k_total: int, temas: pd.Series,
                       semilla: int = 0) -> np.ndarray:
    """Agrupa los textos de contenido dentro de cada macro-tema.

    Reparte `k_total` entre temas en proporcion a su masa de comentarios, con un
    minimo de 2 narrativas por tema. Devuelve etiquetas de texto (str) alineadas a
    `uniq`.
    """
    mapa = dict(zip(uniq["norm"], np.arange(len(uniq))))
    sub = df.loc[mask, ["norm", "publicacion_id"]].copy()
    sub = sub[sub["norm"] != ""]
    sub["tema"] = sub["publicacion_id"].map(temas).fillna(-1).astype(int)
    tab = sub.groupby(["norm", "tema"]).size().rename("w").reset_index()
    tab["i"] = tab["norm"].map(mapa)
    tab = tab[tab["i"].notna()]

    # cada texto se asigna al tema donde mas se usa
    dominante = (tab.sort_values("w", ascending=False)
                 .drop_duplicates("norm", keep="first").copy())
    masa = dominante.groupby("tema")["w"].sum()
    n_temas = max(len(masa), 1)
    etiqueta = np.empty(len(uniq), dtype=object)
    etiqueta[:] = ""
    for tema, s in dominante.groupby("tema"):
        k_i = k_total if n_temas == 1 else max(2, int(round(masa[tema] / masa.sum() * k_total)))
        k_i = max(1, min(k_i, len(s)))
        idx = s["i"].to_numpy(int)
        lab = agrupar(X[idx], k_i, peso=s["w"].to_numpy(float), semilla=semilla)
        for i, l in zip(idx, lab):
            etiqueta[i] = f"t{tema}_{int(l)}"
    faltan = np.where(etiqueta == "")[0]
    for n, i in enumerate(faltan):
        etiqueta[i] = f"solo_{n}"
    return etiqueta


def heredar_respuestas(df: pd.DataFrame, mask: np.ndarray, etiquetas: np.ndarray,
                       postura: np.ndarray, max_tokens: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Propaga narrativa y postura a las respuestas cortas desde su comentario padre.

    El desafio lo pide explicitamente: "Tal cual" o "Exacto" acompanan al padre;
    "mentira" o "nada que ver" heredan la narrativa y invierten la postura. Solo se
    aplica a respuestas de hasta `max_tokens` palabras, que no tienen contenido
    propio suficiente para asignarles narrativa por embeddings.
    """
    et = np.array(etiquetas, dtype=object, copy=True)
    po = np.array(postura, dtype=object, copy=True)
    indice = {c: i for i, c in enumerate(df["comentario_id"])}
    nt = df["nt"].to_numpy()
    resp = df["respuesta_a"].to_numpy()
    norm = df["norm"].to_numpy(object)
    for i in np.where(~mask)[0]:
        if nt[i] > max_tokens or not resp[i]:
            continue
        j = indice.get(resp[i])
        if j is None or mask[j] or not isinstance(et[j], str):
            continue
        et[i] = et[j]
        if any(m in norm[i] for m in MARCADORES_RECHAZO):
            po[i] = "apoya" if po[j] == "rechaza" else "rechaza"
        else:
            po[i] = po[j]
    return et, po


# --------------------------------------------------------------------------- #
# Postura
# --------------------------------------------------------------------------- #
def aprender_lexico_postura(textos: list[str], posturas: np.ndarray,
                            max_terminos: int = 400) -> tuple[dict, float]:
    """Log-odds de tokens para 'rechaza' vs 'apoya', recortado a los mas estables."""
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(ngram_range=(1, 2), min_df=10, binary=True,
                         stop_words=list(VACIAS), max_features=200000)
    X = cv.fit_transform(textos)
    terminos = cv.get_feature_names_out()
    m = posturas == "rechaza"
    p = np.asarray(X[m].mean(0)).ravel() + 1e-6
    q = np.asarray(X[~m].mean(0)).ravel() + 1e-6
    log_odds = np.log(p / q)
    orden = np.argsort(-np.abs(log_odds))[:max_terminos]
    lexico = {str(terminos[i]): float(log_odds[i]) for i in orden}
    # umbral: cuantil que equilibra la decision sobre el propio corpus de calibracion
    score = np.array([sum(lexico.get(t, 0.0) for t in txt.split()) for txt in textos])
    umbral = float(np.quantile(score, float(np.mean(~m))))
    return lexico, umbral


def puntuar_postura(textos: list[str], lexico: dict | None, umbral: float) -> list[str]:
    salida = []
    for txt in textos:
        if any(m in txt for m in MARCADORES_RECHAZO):
            salida.append("rechaza")
            continue
        if lexico:
            s = 0.0
            for tok in txt.split():
                s += lexico.get(tok, 0.0)
            salida.append("rechaza" if s > umbral else "apoya")
        else:
            salida.append("apoya")
    return salida


# --------------------------------------------------------------------------- #
# Origen y descripcion
# --------------------------------------------------------------------------- #
# Total de narrativas del reparto jerarquico (se reparte entre macro-temas en
# proporcion a su masa de comentarios; practica: 5 + 4 + 3).
K_NARRATIVAS = 12

# Corte de cohorte temporal, en fraccion de la duracion del dataset.
FRACCION_COHORTE = 0.35
# Cuota minima de comentarios "tardios" para que un cluster se parta en dos cohortes.
CUOTA_TARDIA = 0.35
# Fraccion del pico diario que define el inicio de una narrativa.
FRACCION_ONSET = 0.10
# Peso minimo de la cohorte tardia para estimar el origen solo con ella.
CUOTA_ONSET_TARDIO = 0.10


def primera_aparicion(df: pd.DataFrame, mask: np.ndarray) -> pd.Series:
    """Dias desde el primer comentario del dataset hasta la primera aparicion de cada texto."""
    t0 = df["fecha"].min()
    return ((df.loc[mask].groupby("norm")["fecha"].min() - t0).dt.total_seconds() / 86400)


def corte_cohorte(df: pd.DataFrame) -> float:
    span = (df["fecha"].max() - df["fecha"].min()).total_seconds() / 86400
    return max(3.0, FRACCION_COHORTE * span)


def etiquetas_cohorte(uniq: pd.DataFrame, etiquetas: np.ndarray,
                      prim: pd.Series, corte: float,
                      cuota: float = CUOTA_TARDIA) -> np.ndarray:
    """Parte en dos cohortes los clusters dominados por textos que aparecen tarde.

    Las narrativas emergentes se reconocen porque casi todos sus textos aparecen por
    primera vez varios dias despues del inicio. La parte tardia recibe el sufijo 'L'.
    """
    prim_u = prim.reindex(uniq["norm"]).fillna(0.0).to_numpy()
    tardio = prim_u >= corte
    peso = uniq["peso"].to_numpy(float)
    salida = np.empty(len(uniq), dtype=object)
    for cl in np.unique(etiquetas):
        m = etiquetas == cl
        cuota_cl = float((peso[m] * tardio[m]).sum() / max(peso[m].sum(), 1e-9))
        parte = tardio & m
        salida[m] = np.where(parte[m] & (cuota_cl >= cuota), f"{cl}L", f"{cl}")
    return salida


def _onset(s: pd.DataFrame, frac: float = FRACCION_ONSET) -> pd.Timestamp | None:
    """Primer comentario del primer dia que alcanza `frac` del pico diario."""
    masa = s.groupby("dia").size().sort_index()
    if masa.empty:
        return None
    d0 = int(masa[masa >= frac * masa.max()].index[0])
    return pd.Timestamp(s.loc[s["dia"] == d0, "fecha"].min())


def origen_por_etiqueta(df: pd.DataFrame, etiquetas: np.ndarray,
                        prim: pd.Series, corte: float | None = None) -> dict:
    """Origen = primer comentario del primer dia que alcanza 10% del pico diario.

    Es una deteccion de onset: no el minimo absoluto (que arrastra comentarios
    aislados previos), sino el arranque de la ola principal de la narrativa.

    Si el cluster contiene una cohorte tardia con peso suficiente, el onset se estima
    solo sobre esa cohorte: es la redaccion propia de la narrativa emergente y evita
    que textos compartidos con narrativas viejas corran el origen al dia 1.
    """
    prim_c = prim.reindex(df["norm"]).fillna(0.0)
    marco = pd.DataFrame({"lab": etiquetas, "fecha": df["fecha"].to_numpy(),
                          "prim": prim_c.to_numpy(), "dia": np.floor(prim_c).astype(int)})
    salida = {}
    for lab, s in marco.groupby("lab"):
        t = None
        if corte is not None:
            tardio = s[s["prim"] >= corte]
            if not tardio.empty and len(tardio) / len(s) >= CUOTA_ONSET_TARDIO:
                t = _onset(tardio)
        salida[lab] = t if t is not None else _onset(s)
    return salida


def _limpiar_titulo(texto: str) -> str:
    """Colapsa espacios, quita emojis y deja una frase presentable para el jurado."""
    limpio = EMOJI.sub(" ", " ".join(str(texto).split()))
    limpio = " ".join(limpio.split()).strip(" ,;:-")
    if limpio:
        limpio = limpio[0].upper() + limpio[1:]
    return limpio


def describir_etiquetas(uniq: pd.DataFrame, etiquetas: np.ndarray,
                        max_terminos: int = 6) -> dict:
    """Titulo legible y frases clave por etiqueta.

    El titulo es el comentario original mas repetido del grupo, priorizando los que
    tienen al menos 5 palabras: un "tal cual" repetido no describe una narrativa.
    Se conservan mayusculas y acentos del texto original, porque este campo lo lee
    el jurado, no el evaluador automatico.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    out = {}
    ejemplos = uniq["ejemplo"] if "ejemplo" in uniq else uniq["norm"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.6, sublinear_tf=True,
                          stop_words=list(VACIAS), max_features=60000)
    # se vectoriza el texto original en minusculas: conserva acentos y ñ en las frases
    # clave, que es lo que lee el jurado ("año", no "ano").
    X = vec.fit_transform(ejemplos.astype(str).str.lower())
    terminos = np.array(vec.get_feature_names_out())
    for cl in pd.unique(etiquetas):
        m = etiquetas == cl
        sub = uniq.loc[m]
        textos = ejemplos.loc[m].astype(str)
        n_palabras = textos.str.split().str.len().to_numpy()
        peso = sub["peso"].to_numpy()
        # preferir el texto mas repetido con cuerpo suficiente para resumir la idea
        aptos = np.where(n_palabras >= 5)[0]
        idx = int(aptos[peso[aptos].argmax()]) if len(aptos) else int(peso.argmax())
        titulo = _limpiar_titulo(textos.iloc[idx])
        media = np.asarray(X[m].mean(0)).ravel()
        top = terminos[np.argsort(-media)[:max_terminos]]
        out[str(cl)] = {
            "titulo": titulo[:120],
            "frases_clave": "|".join(str(t) for t in top),
            "comentarios": int(sub["peso"].sum()),
        }
    return out
