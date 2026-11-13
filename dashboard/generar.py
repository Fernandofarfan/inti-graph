"""Genera un dashboard HTML autocontenido a partir de la entrega y del warehouse.

Sin dependencias de red ni servidores: lee los CSV de la entrega, la fecha original
de cada comentario (Parquet del warehouse o CSV.gz del set) y las features por cuenta,
y escribe un unico `index.html` con los datos embebidos.

Uso:
    python dashboard/generar.py --datos datos/practica --entrega entrega

Abrir `dashboard/index.html` en el navegador (doble clic) o servirlo con
    python -m http.server 8000 --directory dashboard
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.comun import leer_csv  # noqa: E402


def cargar_fechas(datos_dir: Path) -> pd.DataFrame:
    parquet = RAIZ / "salida" / "warehouse" / "comentarios.parquet"
    if parquet.exists():
        cols = pd.read_parquet(parquet, columns=["comentario_id", "fecha"])
        cols["fecha"] = pd.to_datetime(cols["fecha"], utc=True)
        return cols
    c = leer_csv(datos_dir / "comentarios.csv.gz", usecols=["comentario_id", "fecha"])
    c["fecha"] = pd.to_datetime(c["fecha"], utc=True)
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=RAIZ / "datos" / "practica")
    ap.add_argument("--entrega", type=Path, default=RAIZ / "entrega")
    ap.add_argument("--salida", type=Path, default=RAIZ / "dashboard" / "index.html")
    args = ap.parse_args()

    ent = args.entrega
    com = pd.read_csv(ent / "comentarios.csv", dtype=str, keep_default_na=False)
    nar = pd.read_csv(ent / "narrativas.csv", dtype=str, keep_default_na=False)
    cue = pd.read_csv(ent / "cuentas.csv", dtype=str, keep_default_na=False)
    costo = json.loads((ent / "costo.json").read_text(encoding="utf-8"))

    fechas = cargar_fechas(args.datos)
    com = com.merge(fechas, on="comentario_id", how="left")
    com["dia"] = com["fecha"].dt.strftime("%Y-%m-%d")
    com["es_ruido"] = com["narrativa"].str.lower().isin(["", "ninguna", "none", "null"])

    dias = sorted(com["dia"].dropna().unique())
    contenido = com[~com["es_ruido"]]
    total = len(com)
    ruido = int(com["es_ruido"].sum())
    span = (fechas["fecha"].max() - fechas["fecha"].min()).total_seconds() / 86400

    # --- narrativas: serie diaria y postura
    corte = fechas["fecha"].min() + pd.Timedelta(days=0.35 * span)
    narrativas = []
    for _, fila in nar.iterrows():
        nid = fila["narrativa"]
        sub = contenido[contenido["narrativa"] == nid]
        serie = sub.groupby("dia").size().reindex(dias, fill_value=0).astype(int).tolist()
        origen = fila.get("fecha_origen", "")
        primer = sub["fecha"].min()
        temprano = float((sub["fecha"] < corte).mean()) if len(sub) else 1.0
        narrativas.append({
            "id": nid,
            "titulo": fila.get("titulo", ""),
            "frases": fila.get("frases_clave", ""),
            "origen": origen,
            "primer": primer.isoformat() if pd.notna(primer) else "",
            "total": int(len(sub)),
            "apoya": int((sub["postura"] == "apoya").sum()),
            "rechaza": int((sub["postura"] == "rechaza").sum()),
            "serie": serie,
            "tardia": bool(len(sub) and temprano < 0.10),
        })
    narrativas.sort(key=lambda n: -n["total"])

    # --- operaciones coordinadas
    features = RAIZ / "salida" / "features_cuentas.parquet"
    f = pd.read_parquet(features) if features.exists() else None
    operaciones = []
    for grupo, sub in cue[cue["grupo"] != ""].groupby("grupo"):
        fila = {"id": grupo, "cuentas": int(len(sub))}
        if f is not None:
            ff = f[f["cuenta_id"].isin(sub["cuenta_id"])]
            if not ff.empty:
                fila.update({
                    "antiguedad": round(float(ff["antiguedad"].median()), 1),
                    "noches": round(float(ff["noches"].mean()) * 100, 0),
                    "ratio_dup": round(float(ff["ratio_dup"].mean()) * 100, 0),
                    "pubs": round(float(ff["pubs"].mean()), 1),
                    "prob": round(float(ff["prob"].mean()), 2),
                })
        operaciones.append(fila)
    operaciones.sort(key=lambda o: o["id"])

    kpis = {
        "comentarios": total,
        "ruido_pct": round(100 * ruido / max(total, 1), 1),
        "narrativas": len(narrativas),
        "emergentes": int(sum(1 for n in narrativas if n["tardia"])),
        "cuentas_marcadas": int((cue["grupo"] != "").sum()),
        "operaciones": len(operaciones),
        "segundos": costo.get("segundos"),
        "usd": costo.get("usd_ia", 0),
    }
    datos = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(args.datos),
        "dias": dias,
        "kpis": kpis,
        "narrativas": narrativas,
        "operaciones": operaciones,
        "costo": costo,
    }

    plantilla = (Path(__file__).parent / "plantilla.html").read_text(encoding="utf-8")
    html = plantilla.replace("__DATOS__", json.dumps(datos, ensure_ascii=False))
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(html, encoding="utf-8")
    print(f"[dashboard] escrito {args.salida} "
          f"({total} comentarios, {len(narrativas)} narrativas, {len(operaciones)} operaciones)")


if __name__ == "__main__":
    main()
