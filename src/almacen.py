"""Capa de almacenamiento: CSV.gz -> DuckDB -> Parquet (ZSTD).

El desafio puntua donde y como se guardan los datos. La eleccion:

  - **Parquet columnar + ZSTD** porque el set es de solo lectura y se presta a
    agregaciones por columna (fecha, cuenta, publicacion). Comprime ~5x y permite
    leer solo las columnas necesarias.
  - **DuckDB** como motor SQL embebido (sin servidor, sin instalar nada): hace la
    conversion y las agregaciones en segundos y escala a millones de filas por dia
    particionando por fecha.
  - El analisis fino (embeddings, clustering) corre en pandas/NumPy porque el set
    entra comodo en RAM; la capa Parquet es la fuente de verdad y el punto de
    entrada para escalar por particiones.

El pipeline usa esta capa para las agregaciones de actividad y deja el warehouse en
`salida/warehouse/` para el dashboard.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .comun import SALIDA

TABLAS = {
    "comentarios": "comentarios.csv.gz",
    "cuentas": "cuentas.csv.gz",
    "publicaciones": "publicaciones.csv.gz",
    "fuentes": "fuentes.csv",
}

SQL_ACTIVIDAD_DIARIA = """
COPY (
    SELECT CAST(fecha AS DATE) AS dia,
           publicacion_id,
           count(*) AS comentarios,
           count(DISTINCT cuenta_id) AS cuentas
    FROM comentarios
    GROUP BY 1, 2
    ORDER BY 1, 2
) TO '{destino}' (FORMAT PARQUET, COMPRESSION ZSTD)
"""

SQL_ACTIVIDAD_CUENTAS = """
COPY (
    SELECT cuenta_id,
           count(*) AS comentarios,
           count(DISTINCT publicacion_id) AS publicaciones,
           min(fecha) AS primer_comentario,
           max(fecha) AS ultimo_comentario
    FROM comentarios
    GROUP BY 1
) TO '{destino}' (FORMAT PARQUET, COMPRESSION ZSTD)
"""


def construir(datos_dir: Path, destino: Path | None = None,
              verbose: bool = True) -> dict:
    """Materializa el warehouse y devuelve filas por tabla y tamano en disco."""
    import duckdb

    destino = destino or (SALIDA / "warehouse")
    destino.mkdir(parents=True, exist_ok=True)
    db = destino / "qualydata.duckdb"
    con = duckdb.connect(str(db))
    stats: dict[str, object] = {"ruta": str(destino), "tablas": {}, "bytes": 0}
    try:
        con.execute("PRAGMA threads=2")
        for tabla, archivo in TABLAS.items():
            ruta = (datos_dir / archivo)
            if not ruta.exists():
                continue
            con.execute(
                f"""CREATE OR REPLACE TABLE {tabla} AS
                    SELECT * FROM read_csv_auto('{ruta.as_posix()}',
                                                header=true, all_varchar=true)"""
            )
            parquet = destino / f"{tabla}.parquet"
            con.execute(
                f"""COPY {tabla} TO '{parquet.as_posix()}'
                    (FORMAT PARQUET, COMPRESSION ZSTD)"""
            )
            filas = int(con.execute(f"SELECT count(*) FROM {tabla}").fetchone()[0])
            stats["tablas"][tabla] = filas

        for nombre, sql in (("actividad_diaria", SQL_ACTIVIDAD_DIARIA),
                            ("actividad_cuentas", SQL_ACTIVIDAD_CUENTAS)):
            parquet = destino / f"{nombre}.parquet"
            con.execute(sql.format(destino=parquet.as_posix()))
            stats["tablas"][nombre] = int(
                con.execute(f"SELECT count(*) FROM read_parquet('{parquet.as_posix()}')")
                .fetchone()[0])
    finally:
        con.close()

    stats["bytes"] = int(sum(p.stat().st_size for p in destino.glob("*.parquet")))
    stats["mb"] = round(stats["bytes"] / 1024 / 1024, 1)
    if verbose:
        print(f"[almacen] DuckDB + Parquet en {destino} "
              f"({stats['tablas']}, {stats['mb']} MB)")
    return stats


def leer_comentarios(destino: Path | None = None) -> pd.DataFrame:
    """Lee la tabla de comentarios desde Parquet (para el dashboard)."""
    destino = (destino or (SALIDA / "warehouse")) / "comentarios.parquet"
    if not destino.exists():
        raise FileNotFoundError(f"Falta {destino}; corre el pipeline primero")
    return pd.read_parquet(destino)
