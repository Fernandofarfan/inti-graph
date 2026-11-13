"""Detector de ruido supervisado (comentarios sin narrativa).

El ruido del dataset son reacciones genericas (preguntas, saludos, chistes) que no
adhieren a ninguna narrativa. Un clasificador lineal sobre el texto normalizado +
marcas estructurales los separa casi perfecto, y la senal es independiente del tema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def _entrada(norm: pd.Series, nt: pd.Series, tiene_rep: pd.Series, signos: pd.Series) -> pd.Series:
    return (norm + " len" + nt.clip(0, 15).astype(str)
            + " rep" + tiene_rep.astype(int).astype(str)
            + " q" + (signos > 0).astype(int).astype(str))


def matriz(df: pd.DataFrame, vec: TfidfVectorizer | None = None):
    entrada = _entrada(df["norm"], df["nt"], (df["respuesta_a"] != ""),
                       df["texto"].str.count(r"[?¿]"))
    if vec is None:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.7, sublinear_tf=True,
                              max_features=200000)
        X = vec.fit_transform(entrada)
        return vec, X
    return vec, vec.transform(entrada)


def entrenar(df: pd.DataFrame, y: np.ndarray) -> dict:
    vec, X = matriz(df)
    clf = LogisticRegression(max_iter=1500, C=1.0, class_weight="balanced")
    clf.fit(X, y)
    return {"vectorizer": vec, "clf": clf}


def predecir(df: pd.DataFrame, modelo: dict, umbral: float = 0.3) -> np.ndarray:
    _, X = matriz(df, modelo["vectorizer"])
    prob = modelo["clf"].predict_proba(X)[:, 1]
    # regla dura: 1-2 palabras sin respuesta nunca son contenido narrativo
    duro = ((df["nt"] <= 2) & (df["respuesta_a"] == "")).to_numpy()
    return (prob >= umbral) | duro
