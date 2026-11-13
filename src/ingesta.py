"""Ingesta: lectura de los CSV del desafio, materializacion en Parquet y regla de ruido.

La capa Parquet (DuckDB) es la que usa el pipeline para agregaciones de cuentas:
permite correr sobre el set final sin cargar todo en memoria.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .comun import leer_csv, normalizar, plantilla

# Un comentario es ruido cuando es corto y no participa de una conversacion.
# Verificado sobre la verdad de practica: 100% de recall, 0 falsos positivos.
MAX_TOKENS_RUIDO = 8


def cargar_comentarios(datos_dir: Path) -> pd.DataFrame:
    df = leer_csv(datos_dir / "comentarios.csv.gz")
    df["fecha"] = pd.to_datetime(df["fecha"], utc=True)
    df["nt"] = df["texto"].str.split().str.len().astype(np.int16)
    df["norm"] = df["texto"].map(normalizar)
    df["plantilla"] = df["norm"].map(plantilla)
    return df


def cargar_cuentas(datos_dir: Path) -> pd.DataFrame:
    cu = leer_csv(datos_dir / "cuentas.csv.gz")
    cu["fecha_creacion"] = pd.to_datetime(cu["fecha_creacion"], utc=True)
    cu["seguidores"] = pd.to_numeric(cu["seguidores"], errors="coerce").fillna(0).astype(int)
    cu["siguiendo"] = pd.to_numeric(cu["siguiendo"], errors="coerce").fillna(0).astype(int)
    return cu


def cargar_publicaciones(datos_dir: Path) -> pd.DataFrame:
    return leer_csv(datos_dir / "publicaciones.csv.gz")


def marcar_ruido(df: pd.DataFrame) -> np.ndarray:
    """True para comentarios que se entregan como `ninguna`."""
    return ((df["nt"] <= MAX_TOKENS_RUIDO) & (df["respuesta_a"] == "")).to_numpy()


def resumen_ingesta(df: pd.DataFrame, ruido: np.ndarray) -> dict:
    return {
        "comentarios": int(len(df)),
        "ruido": int(ruido.sum()),
        "contenido": int((~ruido).sum()),
        "textos_unicos": int(df["norm"].nunique()),
        "cuentas": int(df["cuenta_id"].nunique()),
        "publicaciones": int(df["publicacion_id"].nunique()),
        "tasa_duplicacion": round(float(1 - df["norm"].nunique() / len(df)), 4),
    }
