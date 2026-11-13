"""Verificar si fecha_origen == min(fecha) por narrativa, y medir postura con reglas."""
import pathlib
from collections import Counter

import pandas as pd

RAIZ = pathlib.Path(__file__).resolve().parent.parent


def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/comentarios.csv.gz", dtype=str, keep_default_na=False)
    nar = pd.read_csv(RAIZ / "datos/practica/verdad/narrativas.csv", dtype=str, keep_default_na=False)
    c["fecha"] = pd.to_datetime(c.fecha, utc=True)
    d = c.merge(v, on="comentario_id")

    print("=== min(fecha) por narrativa vs fecha_origen ===")
    for _, r in nar.iterrows():
        nid = r.narrativa_id
        if r.emergente != "si":
            continue
        sub = d[(d.narrativa_id == nid) & (d.postura != "")] if "postura" in d else d[d.narrativa_id == nid]
        mn = sub.fecha.min()
        fo = pd.to_datetime(r.fecha_origen, utc=True)
        err = abs((mn - fo).total_seconds()) / 3600
        print(f"  {nid} emergente: min={mn} fecha_origen={fo} err={err:.2f}h n={len(sub)}")

    print("\n=== min(fecha) todas las narrativas ===")
    for nid in sorted(nar.narrativa_id):
        sub = d[d.narrativa_id == nid]
        mn = sub.fecha.min()
        fo = pd.to_datetime(nar[nar.narrativa_id == nid].fecha_origen.iloc[0], utc=True)
        err = abs((mn - fo).total_seconds()) / 3600
        print(f"  {nid}: min={mn} origen={fo} err={err:.2f}h")

    print("\n=== distribucion de postura ===")
    print(d.postura.value_counts().to_string())
    print("\npor narrativa:")
    print(pd.crosstab(d.narrativa_id, d.postura).to_string())

    print("\n=== marcadores de rechazo/apoyo ===")
    d["nt"] = d.texto.str.split().str.len()
    cont = d[(d.narrativa_id != "ninguna")]
    rech = cont[cont.postura == "rechaza"]
    apo = cont[cont.postura == "apoya"]
    print("rechaza nt medio:", rech.nt.mean(), "apoya:", apo.nt.mean())
    print("\ntextos de rechazo:")
    for t in rech.texto.sample(15, random_state=3):
        print("  -", t[:100])
    print("\ntextos de apoyo:")
    for t in apo.texto.sample(10, random_state=3):
        print("  -", t[:100])


if __name__ == "__main__":
    main()
