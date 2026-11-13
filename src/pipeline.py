"""Orquestador del pipeline INTI-GRAPH.

Uso:
    python -m src.pipeline --datos datos/practica --entrega entrega \
        --calibracion datos/practica/verdad

Los artefactos de calibracion (metrica de embeddings, lexico de postura, detector
de ruido) se construyen una vez desde la verdad del set de practica y se cachean en
`salida/calibracion/`. El set objetivo nunca aporta etiquetas.
"""
from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from . import (almacen, calibracion, coordinacion, embeddings, ingesta, narrativas,
               ruido as ruido_mod)
from .comun import ENTREGA, RAIZ, SALIDA, a_iso_ar, asegurar_dirs


def predecir_narrativas(df: pd.DataFrame, mascara_ruido: np.ndarray, art: dict,
                        k: int | None = None, verbose: bool = True) -> dict:
    k = int(k or narrativas.K_NARRATIVAS)
    uniq = narrativas.textos_unicos(df, ~mascara_ruido)
    if verbose:
        print(f"[narrativas] {len(uniq)} textos unicos sobre {int((~mascara_ruido).sum())} comentarios")
    E = embeddings.embeddings_unicos(uniq["norm"].tolist(), etiqueta="unicos", verbose=verbose)
    X = narrativas.escalar(E, art["w"])
    del E

    # Jerarquia tema -> narrativa: primero se separan los macro-temas por publicacion
    # y despues se agrupan los textos dentro de cada tema. Evita mezclar temas.
    temas, n_temas = narrativas.temas_por_publicacion(df, ~mascara_ruido, uniq, X)
    etiquetas_u = narrativas.agrupar_jerarquico(df, ~mascara_ruido, uniq, X, k, temas)
    del X
    if verbose:
        print(f"[narrativas] {n_temas} macro-temas, k total={k}")

    # cohorte temporal: separa narrativas emergentes (textos que aparecen tarde)
    prim = narrativas.primera_aparicion(df, ~mascara_ruido)
    corte = narrativas.corte_cohorte(df)
    lab_u = narrativas.etiquetas_cohorte(uniq, etiquetas_u, prim, corte)
    if verbose:
        print(f"[narrativas] corte de cohorte={corte:.1f} dias, "
              f"{len(pd.unique(lab_u))} etiquetas tras la cohorte")

    idx_de = dict(zip(uniq["norm"], lab_u))
    etiquetas = df["norm"].map(idx_de)

    postura_u = narrativas.puntuar_postura(uniq["norm"].tolist(), art["lexico"],
                                           art["umbral_postura"])
    postura_de = dict(zip(uniq["norm"], postura_u))
    postura = df["norm"].map(postura_de)

    # respuestas cortas: heredan narrativa y postura del comentario padre
    etiquetas_np, postura_np = narrativas.heredar_respuestas(
        df, mascara_ruido, etiquetas.to_numpy(object), postura.to_numpy(object))
    etiquetas = pd.Series(etiquetas_np, index=df.index)
    postura = pd.Series(postura_np, index=df.index)

    cont = df.loc[~mascara_ruido]
    etiquetas_cont = etiquetas[~mascara_ruido].to_numpy()
    origen = narrativas.origen_por_etiqueta(cont, etiquetas_cont, prim, corte)
    desc = narrativas.describir_etiquetas(uniq, lab_u)

    orden = sorted(desc, key=lambda l: -desc[l]["comentarios"])
    mapa = {lab: f"n{i + 1:02d}" for i, lab in enumerate(orden)}

    return {
        "etiquetas": etiquetas,
        "postura": postura,
        "mapa": mapa,
        "desc": {mapa[c]: v for c, v in desc.items()},
        "origen": {mapa[c]: t for c, t in origen.items()},
        "k": k,
        "corte_cohorte": corte,
        "n_temas": n_temas,
        "textos_unicos": len(uniq),
    }


def predecir_coordinacion(df: pd.DataFrame, cu: pd.DataFrame, art: dict,
                          umbral: float = 0.30, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = coordinacion.features_cuentas(df, cu)
    prob = coordinacion.puntuar(feats, art["modelo_cuentas"])
    feats["prob"] = prob
    motivos = coordinacion.detectar(feats, prob, umbral=umbral)
    grupos = coordinacion.agrupar(set(motivos), feats)
    if verbose:
        print(f"[coordinacion] {len(motivos)} cuentas marcadas, "
              f"{len(grupos)} agrupadas en {len(set(grupos.values()))} operaciones")
    return pd.DataFrame({
        "cuenta_id": list(motivos),
        "grupo": [grupos.get(c, "") for c in motivos],
    }).sort_values("cuenta_id"), feats


def escribir_entrega(df: pd.DataFrame, mascara_ruido: np.ndarray, narr: dict,
                     cuentas: pd.DataFrame, cu: pd.DataFrame, entrega_dir: Path,
                     costo: dict) -> None:
    entrega_dir.mkdir(parents=True, exist_ok=True)
    mapa = narr["mapa"]
    filas = []
    for cid, cl, pos, es_ruido in zip(df["comentario_id"], narr["etiquetas"],
                                      narr["postura"], mascara_ruido):
        if es_ruido or pd.isna(cl):
            filas.append((cid, "ninguna", ""))
        else:
            filas.append((cid, mapa[cl], pos if isinstance(pos, str) else ""))
    pd.DataFrame(filas, columns=["comentario_id", "narrativa", "postura"]).to_csv(
        entrega_dir / "comentarios.csv", index=False, encoding="utf-8")

    filas_n = []
    for lab in sorted(narr["desc"]):
        d = narr["desc"][lab]
        fecha = narr["origen"].get(lab)
        filas_n.append({
            "narrativa": lab,
            "titulo": d["titulo"],
            "frases_clave": d["frases_clave"],
            "fecha_origen": a_iso_ar(fecha) if fecha is not None else "",
        })
    pd.DataFrame(filas_n, columns=["narrativa", "titulo", "frases_clave", "fecha_origen"]).to_csv(
        entrega_dir / "narrativas.csv", index=False, encoding="utf-8")

    # una fila por cuenta: grupo vacio si es genuina (formato del desafio)
    grupo_de = dict(zip(cuentas["cuenta_id"], cuentas["grupo"]))
    filas_c = [(cid, grupo_de.get(cid, "")) for cid in cu["cuenta_id"]]
    pd.DataFrame(filas_c, columns=["cuenta_id", "grupo"]).to_csv(
        entrega_dir / "cuentas.csv", index=False, encoding="utf-8")
    (entrega_dir / "costo.json").write_text(
        json.dumps(costo, ensure_ascii=False, indent=2), encoding="utf-8")


def ejecutar(datos_dir: Path, entrega_dir: Path, calib_dir: Path,
             k: int | None = None, umbral_cuentas: float = 0.30,
             verbose: bool = True) -> dict:
    asegurar_dirs()
    t0 = time.time()
    tracemalloc.start()

    art = calibracion.construir(datos_dir=calib_dir.parent, verdad_dir=calib_dir,
                               verbose=verbose)

    if verbose:
        print(f"[ingesta] {datos_dir}")
    try:
        almacen_stats = almacen.construir(datos_dir, verbose=verbose)
    except Exception as exc:  # el warehouse no debe romper el pipeline
        almacen_stats = {"error": str(exc)}
        if verbose:
            print(f"[almacen] aviso: {exc}")
    df = ingesta.cargar_comentarios(datos_dir)
    cu = ingesta.cargar_cuentas(datos_dir)

    mascara_ruido = ruido_mod.predecir(df, art["modelo_ruido"])
    stats = ingesta.resumen_ingesta(df, mascara_ruido)
    if verbose:
        print(f"[ingesta] {stats}")

    t_narr = time.time()
    narr = predecir_narrativas(df, mascara_ruido, art, k=k, verbose=verbose)
    if verbose:
        print(f"[narrativas] k={narr['k']} en {time.time() - t_narr:.1f}s")

    cuentas, feats = predecir_coordinacion(df, cu, art, umbral=umbral_cuentas,
                                           verbose=verbose)
    feats.to_parquet(SALIDA / "features_cuentas.parquet", index=False)
    segundos = time.time() - t0
    actual, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    costo = {
        "comentarios_procesados": int(stats["comentarios"]),
        "segundos": round(segundos, 1),
        "tokens_ia": 0,
        "usd_ia": 0.0,
        "usd_por_mil_comentarios": 0.0,
        "hardware": "notebook CPU, 8 GB RAM, sin GPU",
        "almacenamiento": ("DuckDB + Parquet ZSTD en salida/warehouse "
                           f"({almacen_stats.get('mb', 0)} MB); pandas/NumPy en RAM; "
                           "embeddings en cache .npz por hash de texto"),
        "notas": (f"Sin API ni GPU: costo IA 0. {stats['comentarios'] / max(segundos, 1e-9):.0f} "
                  "comentarios/s de punta a punta, un solo comando. "
                  f"Pico de memoria del proceso Python: {pico / 1024 / 1024:.0f} MB. "
                  f"{narr['k']} narrativas base x {narr.get('n_temas', '?')} temas; "
                  f"{len(cuentas)} cuentas marcadas."),
    }
    escribir_entrega(df, mascara_ruido, narr, cuentas, cu, entrega_dir, costo)

    reporte = {
        "datos": str(datos_dir),
        "entrega": str(entrega_dir),
        "k": narr["k"],
        "temas": narr.get("n_temas"),
        "ingesta": stats,
        "almacen": almacen_stats,
        "calibracion": art["meta"],
        "coordinacion": {
            "cuentas_marcadas": int(len(cuentas)),
            "cuentas_agrupadas": int(cuentas["grupo"].ne("").sum()),
            "operaciones": int(cuentas["grupo"].nunique()),
        },
        "costo": costo,
    }
    (SALIDA / "reporte.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    return reporte


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=RAIZ / "datos" / "practica")
    ap.add_argument("--entrega", type=Path, default=ENTREGA)
    ap.add_argument("--calibracion", type=Path, default=RAIZ / "datos" / "practica" / "verdad")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--umbral-cuentas", type=float, default=0.30)
    args = ap.parse_args()
    rep = ejecutar(args.datos, args.entrega, args.calibracion, k=args.k,
                   umbral_cuentas=args.umbral_cuentas)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
