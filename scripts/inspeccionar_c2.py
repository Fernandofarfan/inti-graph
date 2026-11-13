"""Inspeccionar el patron temporal de C2 en una publicacion concreta."""
import pathlib
from collections import Counter

import pandas as pd

RAIZ = pathlib.Path(__file__).resolve().parent.parent


def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/cuentas.csv.gz", dtype=str, keep_default_na=False)
    c["fecha"] = pd.to_datetime(c.fecha, utc=True)
    c = c.merge(v, on="cuenta_id", how="left")
    c["campana_id"] = c.campana_id.fillna("")

    por_pub = c[c.campana_id == "C2"].groupby("publicacion_id").size().sort_values(ascending=False)
    print("publicaciones con mas C2:", por_pub.head(12).to_dict())

    pub = por_pub.index[0]
    sub = c[c.publicacion_id == pub].sort_values("fecha")
    print(f"\n=== publicacion {pub}: {len(sub)} comentarios ({sub.campana_id.ne('').sum()} de campana) ===")
    t0 = sub.fecha.min()
    for _, r in sub.head(60).iterrows():
        delta = (r.fecha - t0).total_seconds()
        print(f"  +{delta:7.0f}s {r.cuenta_id} [{r.campana_id or '   '}] nt={len(r.texto.split()):2d} "
              f"rep={'si' if r.respuesta_a else '  '} {r.texto[:70]}")

    print("\n=== C2: sus comentarios en esa publicacion ===")
    s2 = sub[sub.campana_id == "C2"]
    print(s2[["cuenta_id", "fecha", "texto", "respuesta_a"]].to_string(max_colwidth=80))


if __name__ == "__main__":
    main()
