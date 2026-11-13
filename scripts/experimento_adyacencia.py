"""Adyacencia repetida: pares de cuentas consecutivas en la misma publicacion, repetido en varias publicaciones."""
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
    coord = {i for i, k in cam.items() if k}

    ady = Counter()          # pares consecutivos
    ady_pub = defaultdict(set)  # pares -> publicaciones
    ady_cerca = Counter()    # consecutivos con <=60s
    for pub, s in c.sort_values("fecha").groupby("publicacion_id"):
        u = s.cuenta_id.values
        t = s.fecha.values.astype("datetime64[s]").astype(np.int64)
        for k in range(len(u) - 1):
            if u[k] == u[k + 1]:
                continue
            par = tuple(sorted((u[k], u[k + 1])))
            ady[par] += 1
            ady_pub[par].add(pub)
            if t[k + 1] - t[k] <= 60:
                ady_cerca[par] += 1

    print("pares adyacentes:", len(ady))
    for k in (1, 2, 3, 5):
        sel = {p for p, n in ady.items() if n >= k}
        intra = Counter(cam[a] for a, b in sel if cam[a] and cam[a] == cam[b])
        cuentas = {i for p in sel for i in p}
        print(f"  ady>= {k}: {len(sel)} pares, intra={dict(intra)}, cuentas={len(cuentas)}, "
              f"de_campana={len(cuentas & coord)}")

    print("\n=== adyacencia cercana (<=60s) ===")
    for k in (1, 2, 3):
        sel = {p for p, n in ady_cerca.items() if n >= k}
        intra = Counter(cam[a] for a, b in sel if cam[a] and cam[a] == cam[b])
        cuentas = {i for p in sel for i in p}
        print(f"  ady60>= {k}: {len(sel)} pares, intra={dict(intra)}, cuentas={len(cuentas)}, "
              f"de_campana={len(cuentas & coord)}")

    print("\n=== co-publicacion ademas de adyacencia ===")
    pub_de = c.groupby("cuenta_id").publicacion_id.apply(set).to_dict()
    for k_ady in (1, 2):
        for k_pub in (2, 3):
            sel = [p for p, n in ady.items() if n >= k_ady and len(pub_de[p[0]] & pub_de[p[1]]) >= k_pub]
            intra = Counter(cam[a] for a, b in sel if cam[a] and cam[a] == cam[b])
            cuentas = {i for p in sel for i in p}
            print(f"  ady>={k_ady} & pubcom>={k_pub}: {len(sel)} pares intra={dict(intra)} "
                  f"cuentas={len(cuentas)} de_camp={len(cuentas & coord)}")

    # ---- grafo combinado
    print("\n=== grafo combinado: ady>=2 o (ady>=1 y pubcom>=3) ===")
    edges = {p for p, n in ady.items() if n >= 2}
    edges |= {p for p, n in ady.items() if len(pub_de[p[0]] & pub_de[p[1]]) >= 3 and n >= 1}
    padre = {}

    def find(x):
        padre.setdefault(x, x)
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            padre[rb] = ra
    grupos = defaultdict(set)
    for x in list(padre):
        grupos[find(x)].add(x)
    comps = sorted(grupos.values(), key=len, reverse=True)
    print("componentes:", len(comps), "top:", [len(x) for x in comps[:15]])
    for comp in comps[:8]:
        print(f"  tam={len(comp)} campanas={dict(Counter(cam[i] for i in comp))}")


if __name__ == "__main__":
    main()
