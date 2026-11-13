"""Calibracion: artefactos aprendidos de la verdad del set de practica.

Genera una sola vez (y cachea) tres artefactos que el pipeline reutiliza:
  - `w_metrica.npy`: pesos por dimension del embedding (metrica de misma narrativa);
  - `lexico_postura.json`: lexico de refutacion + umbral;
  - `modelo_ruido.joblib`: detector de comentarios sin narrativa.

Nada de esto memoriza temas: son transformaciones y lexicos de estilo.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import embeddings, ingesta, narrativas, ruido
from .comun import RAIZ, SALIDA, leer_csv, normalizar

DIR = SALIDA / "calibracion"


def _rutas() -> dict:
    return {
        "metrica": DIR / "w_metrica.npy",
        "postura": DIR / "lexico_postura.json",
        "ruido": DIR / "modelo_ruido.joblib",
        "cuentas": DIR / "modelo_cuentas.joblib",
        "meta": DIR / "meta.json",
    }


def existen() -> bool:
    return all(p.exists() for p in _rutas().values())


def construir(datos_dir: Path | None = None, verdad_dir: Path | None = None,
              forzar: bool = False, verbose: bool = True) -> dict:
    datos_dir = datos_dir or (RAIZ / "datos" / "practica")
    verdad_dir = verdad_dir or (RAIZ / "datos" / "practica" / "verdad")
    rutas = _rutas()
    if existen() and not forzar:
        if verbose:
            print("[calibracion] artefactos existentes; se reutilizan")
        return cargar()

    DIR.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("[calibracion] construyendo artefactos desde", datos_dir)

    df = ingesta.cargar_comentarios(datos_dir)
    ver = leer_csv(verdad_dir / "comentarios.csv.gz")
    nar_de = dict(zip(ver["comentario_id"], ver["narrativa_id"]))
    pos_de = dict(zip(ver["comentario_id"], ver["postura"]))
    df["_y"] = df["comentario_id"].map(nar_de).fillna("ninguna")
    df["_pos"] = df["comentario_id"].map(pos_de).fillna("")

    # 1) detector de ruido
    y_ruido = (df["_y"] == "ninguna").to_numpy()
    modelo_ruido = ruido.entrenar(df, y_ruido.astype(int))
    joblib.dump(modelo_ruido, rutas["ruido"])
    if verbose:
        print(f"[calibracion] detector de ruido entrenado ({y_ruido.mean():.1%} ruido)")

    # 2) metrica de embeddings sobre textos unicos de contenido
    uniq = narrativas.textos_unicos(df, ~y_ruido)
    y_uniq = df.loc[~y_ruido].groupby("norm")["_y"].agg(lambda s: s.value_counts().index[0])
    y = y_uniq.reindex(uniq["norm"]).fillna("ninguna").to_numpy(dtype=object)
    E = embeddings.embeddings_unicos(uniq["norm"].tolist(), etiqueta="unicos", verbose=verbose)
    w = narrativas.aprender_metrica(E, y, uniq["peso"].to_numpy(float))
    np.save(rutas["metrica"], w)
    del E
    if verbose:
        print(f"[calibracion] metrica aprendida (max={w.max():.2f}, dims activas={int((w > 0.5).sum())})")

    # 3) lexico de postura
    cont = df[(df["_y"] != "ninguna") & (df["_pos"] != "")]
    lexico, umbral = narrativas.aprender_lexico_postura(
        cont["texto"].map(normalizar).tolist(), cont["_pos"].to_numpy(dtype=object)
    )
    rutas["postura"].write_text(json.dumps({"lexico": lexico, "umbral": umbral},
                                           ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"[calibracion] lexico de postura: {len(lexico)} terminos, umbral={umbral:.2f}")

    # 4) detector de coordinacion (modelo por cuenta)
    from . import coordinacion

    cu = ingesta.cargar_cuentas(datos_dir)
    ver_cue = leer_csv(verdad_dir / "cuentas.csv.gz")
    campanas = dict(zip(ver_cue["cuenta_id"], ver_cue["campana_id"]))
    feats = coordinacion.features_cuentas(df, cu)
    clf = coordinacion.entrenar_modelo(feats, pd.Series(campanas))
    joblib.dump(clf, rutas["cuentas"])
    positivas = int(sum(1 for v in campanas.values() if v))
    if verbose:
        print(f"[calibracion] detector de cuentas entrenado ({positivas} cuentas de campana)")

    meta = {
        "datos": str(datos_dir),
        "comentarios": int(len(df)),
        "textos_unicos": int(len(uniq)),
        "ruido": int(y_ruido.sum()),
        "narrativas": int(df.loc[df["_y"] != "ninguna", "_y"].nunique()),
        "cuentas_coordinadas": positivas,
        "dims_metrica_activas": int((w > 0.5).sum()),
        "k_narrativas": int(narrativas.K_NARRATIVAS),
    }
    rutas["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return cargar()


def cargar() -> dict:
    rutas = _rutas()
    postura = json.loads(rutas["postura"].read_text(encoding="utf-8"))
    meta = json.loads(rutas["meta"].read_text(encoding="utf-8")) if rutas["meta"].exists() else {}
    return {
        "w": np.load(rutas["metrica"]),
        "lexico": postura["lexico"],
        "umbral_postura": float(postura["umbral"]),
        "modelo_ruido": joblib.load(rutas["ruido"]),
        "modelo_cuentas": joblib.load(rutas["cuentas"]),
        "meta": meta,
    }


if __name__ == "__main__":
    print(json.dumps(construir(forzar=True)["meta"], ensure_ascii=False, indent=2))
