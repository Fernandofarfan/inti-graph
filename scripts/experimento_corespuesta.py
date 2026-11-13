"""Co-respuesta: varias cuentas de la misma campana respondiendo al mismo comentario."""
import pathlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

RAIZ = pathlib.Path(__file__).resolve().parent.parent


def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/cuentas.csv.gz", dtype=str, keep_default_na=False)
    c["fecha"] = pd.to_datetime(c.fecha, utc=True)
    c = c.merge(v, on="cuenta_id", how="left")
    c["campana_id"] = c.campana_id.fillna("")
    cam = dict(zip(c.cuenta_id, c.campana_id))
    c["nt"] = c.texto.str.split().str.len()

    rep = c[c.respuesta_a != ""].copy()
    print("respuestas totales:", len(rep))

    # grupos por padre
    g = rep.groupby("respuesta_a")
    print("padres con >=2 respuestas:", int((g.size() >= 2).sum()))
    print("distribucion de #respuestas por padre:", g.size().value_counts().sort_index().head(10).to_dict())

    # por padre: cuantas campanas distintas responden
    def resumen(s):
        camps = [x for x in s.campana_id if x]
        return len(camps), len(set(camps)), (max(Counter(camps).values()) if camps else 0)

    mult = g.apply(lambda s: pd.Series(resumen(s), index=["n_camp", "camps_dist", "max_igual"]),
                   include_groups=False)
    print("\npadres respondidos por >=2 cuentas de campana:", int((mult.n_camp >= 2).sum()))
    print("padres con 2+ de la MISMA campana:", int((mult.max_igual >= 2).sum()))
    print(mult[mult.max_igual >= 2].sort_values("max_igual", ascending=False).head(10).to_string())

    # pares co-respuesta
    print("\n=== pares que comparten padre ===")
    pares = Counter()
    for padre, s in rep.groupby("respuesta_a"):
        ids = list(dict.fromkeys(s.cuenta_id))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pares[tuple(sorted((ids[i], ids[j])))] += 1
    print("pares con >=1 padre comun:", len(pares))
    print("pares con >=2 padres comunes:", sum(1 for n in pares.values() if n >= 2))
    coord = {i for i, k in cam.items() if k}
    for k in (1, 2, 3):
        sel = {p for p, n in pares.items() if n >= k}
        intra = Counter()
        for (a, b) in sel:
            if cam[a] and cam[a] == cam[b]:
                intra[cam[a]] += 1
        cuentas = {i for p in sel for i in p}
        print(f"  >= {k} padres comunes: {len(sel)} pares, intra-campana={dict(intra)}, "
              f"cuentas={len(cuentas)}, cuentas_de_campana={len(cuentas & coord)}")

    # pads: cuantas respuestas del mismo autor en segundos
    print("\n=== coincidencia temporal de respuestas al mismo padre ===")
    sub = rep.sort_values("fecha")
    filas = []
    for padre, s in sub.groupby("respuesta_a"):
        if len(s) < 2:
            continue
        t = s.fecha.values.astype("datetime64[s]").astype(np.int64)
        dif = np.diff(t)
        filas.append((padre, len(s), float(np.min(dif)), float(np.median(dif))))
    df = pd.DataFrame(filas, columns=["padre", "n", "min_diff", "med_diff"])
    print("padres con >=3 respuestas y min_diff<=10s:", int(((df.n >= 3) & (df.min_diff <= 10)).sum()))
    print("padres con min_diff<=5s:", int((df.min_diff <= 5).sum()))

    # esas respuestas simultaneas, de que campana son
    rapidos = set(df[df.min_diff <= 10].padre)
    s = rep[rep.respuesta_a.isin(rapidos)]
    print("respuestas involucradas:", len(s), "campanas:",
          dict(Counter(s.campana_id)))
    # cuentas
    print("cuentas involucradas:", len(set(s.cuenta_id)),
          "de campana:", len(set(s.cuenta_id) & coord),
          "dist:", dict(Counter(cam[i] for i in set(s.cuenta_id) & coord)))

    # analisis: respuestas cortas de acuerdo
    print("\n=== respuestas cortas (<=4 palabras) por campana ===")
    cortas = rep[rep.nt <= 4]
    print("total cortas:", len(cortas))
    print("por campana del autor:", dict(Counter(cortas.campana_id)))
    print("cuentas distintas con >=2 cortas:", len(cortas.groupby("cuenta_id").filter(lambda s: len(s) >= 2).cuenta_id.unique()))
    print("  de campana:", len(set(cortas.groupby("cuenta_id").filter(lambda s: len(s) >= 2).cuenta_id) & coord))


if __name__ == "__main__":
    main()
