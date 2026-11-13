"""Utilidades compartidas del pipeline INTI-GRAPH.

Convenciones:
- Todos los textos se normalizan con `normalizar` antes de vectorizar.
- Las fechas del dataset vienen en ISO con offset -03:00; se conserva el offset
  al serializar para que `fecha_origen` quede en el formato que espera el jurado.
"""
from __future__ import annotations

import gzip
import re
import unicodedata
from datetime import timedelta, timezone
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "datos"
SALIDA = RAIZ / "salida"
ENTREGA = RAIZ / "entrega"
MODELOS = RAIZ / "modelos"

AR = timezone(timedelta(hours=-3))

ABREVIATURAS = {
    "q": "que", "xq": "porque", "x": "por", "tmb": "tambien", "pa": "para",
    "tdo": "todo", "d": "de", "xfa": "por favor", "naide": "nadie", "etsan": "estan",
    "estbaa": "estaba", "comuindades": "comunidades", "aunq": "aunque", "pq": "porque",
}

VACIAS = set("""a al algo con como de del el ella en es esa ese eso esta este esto la las le lo los
mas me mi muy no nos o para pero por que se si sin su sus te todo todos un una uno y ya ahi aca hay
""".split())

# Marcadores lexicos de refutacion: el comentario niega la afirmacion de la narrativa.
MARCADORES_RECHAZO = [
    "mentira", "verso", "falso", "fake", "nada que ver", "no es cierto", "trucho",
    "desmintio", "desmintieron", "sin pruebas", "inventan", "dejen de", "no hubo",
    "nadie va a", "no es verdad", "no se robo", "no robaron", "en realidad",
    "lo que pasa es", "abran los ojos", "lean bien", "informense", "hay que informarse",
    "mucho antes", "no tiene nada que ver", "otra vez con", "eso es viejo", "sin fundamento",
    "no es culpa", "no son", "no estaba", "no es la mina", "no es el hospital",
    "no es el boleto", "no es la obra", "no se freno", "no se paro", "jamas",
]

# Expresiones que abren un comentario y no aportan contenido: se recortan para
# que la plantilla quede al descubierto (copy-paste con muletillas distintas).
PREFIJOS = [
    ["te", "digo", "algo"], ["te", "digo"], ["la", "verdad", "que"], ["la", "verdad"],
    ["y", "bueno"], ["como", "siempre"], ["a", "ver"], ["para", "mi"], ["ojo"],
    ["jaja"], ["jeje"], ["che"], ["escuchen"], ["mira"], ["sinceramente"], ["posta"],
    ["en", "serio"], ["bueno"], ["dale"], ["la", "posta"], ["posta"], ["eh"],
]
SUFIJOS = {"asi", "estamos", "bronca", "despierten", "verguenza", "lamentable", "triste",
           "serio", "nada", "juro", "ojo", "eh", "posta", "punto", "tristeza"}


def normalizar(texto: str) -> str:
    """Minusculas, sin acentos, sin URLs/arrobas/hashtags, sin caracteres raros."""
    t = unicodedata.normalize("NFD", texto.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[@#]\w+", " ", t)
    t = re.sub(r"[^a-z0-9\u00f1\s]", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1", t)
    return " ".join(ABREVIATURAS.get(w, w) for w in t.split())


def plantilla(texto_normalizado: str) -> str:
    """Quita muletillas de apertura/cierre para comparar plantillas de copy-paste."""
    palabras = texto_normalizado.split()
    cambio = True
    while cambio:
        cambio = False
        for pref in PREFIJOS:
            if len(palabras) > len(pref) and palabras[:len(pref)] == pref:
                palabras = palabras[len(pref):]
                cambio = True
                break
    while palabras and palabras[-1] in SUFIJOS:
        palabras = palabras[:-1]
    return " ".join(palabras)


def tiene_rechazo(texto_normalizado: str) -> bool:
    return any(m in texto_normalizado for m in MARCADORES_RECHAZO)


def leer_csv(ruta: Path, **kw) -> pd.DataFrame:
    """Lee CSV o CSV.GZ con todos los campos como texto y sin NaN."""
    return pd.read_csv(ruta, dtype=str, keep_default_na=False, **kw)


def a_iso_ar(fecha) -> str:
    """Serializa un timestamp a ISO con offset -03:00."""
    ts = pd.Timestamp(fecha)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(AR).strftime("%Y-%m-%dT%H:%M:%S") + "-03:00"


def asegurar_dirs() -> None:
    for d in (SALIDA, ENTREGA, MODELOS):
        d.mkdir(parents=True, exist_ok=True)


def fmt_num(x: float, dec: int = 3) -> str:
    return f"{x:.{dec}f}"
