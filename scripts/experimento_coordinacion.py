"""Experimento: grafo de co-publicacion + respuestas para detectar C2, y comunidades."""
import itertools
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent


def evaluar(marcadas, grupos, cam_de, nombre):
    """F1 sobre cuentas marcadas y B3 sobre grupos predichos."""
    coord = {i for i, k in cam_de.items() if k}
    tp = len(marcadas & coord)
    fp = len(marcadas - coord)
    fn = len(coord - marcadas)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    # B3 de grupos (solo sobre cuentas coordinadas verdaderas que fueron marcadas)
    from collections import Counter as C
    verdad = {i: cam_de[i] for i in coord}
    pred = {i: grupos[i] for i in coord if i in grupos}
    if not pred:
        b3 = 0.0
    else:
        precs, recs = [], []
        for i in pred:
            pi = {j for j in pred if pred[j] == pred[i]}
            ti = {j for j in verdad if verdad[j] == verdad[i]}
            precs.append(len(pi & ti) / len(pi))
            recs.append(len(pi & ti) / len(ti))
        b3 = 2 * (np.mean(precs) * np.mean(recs)) / (np.mean(precs) + np.mean(recs))
    comb = 0.7 * f1 + 0.3 * b3
    print(f"  {nombre}: marcadas={len(marcadas)} P={p:.3f} R={r:.3f} F1={f1:.3f} "
          f"B3grupos={b3:.3f} comb={comb:.3f} ({25*comb:.1f} pts)")
    return f1, b3


def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    cu = pd.read_csv(RAIZ / "datos/practica/cuentas.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/cuentas.csv.gz", dtype=str, keep_default_na=False)
    c["fecha"] = pd.to_datetime(c.fecha, utc=True)
    cu["fecha_creacion"] = pd.to_datetime(cu.fecha_creacion, utc=True)
    ini = c.fecha.min()
    c = c.merge(cu[["cuenta_id", "fecha_creacion"]], on="cuenta_id")
    c = c.merge(v, on="cuenta_id", how="left")
    c["campana_id"] = c.campana_id.fillna("")
    cam_de = dict(zip(c.cuenta_id, c.campana_id))
    edad = ((ini - cu.set_index("cuenta_id").fecha_creacion).dt.total_seconds() / 86400)
    pub_de = c.groupby("cuenta_id").publicacion_id.apply(set).to_dict()

    # ---------- grafo de publicaciones compartidas (solo cuentas con 4..25 publicaciones)
    candidatas = [i for i, s in pub_de.items() if 3 <= len(s) <= 30]
    print("candidatas:", len(candidatas), "de", len(pub_de))
    idx_de_pub = defaultdict(list)
    for i in candidatas:
        for p in pub_de[i]:
            idx_de_pub[p].append(i)

    comp = Counter()
    for p, ids in idx_de_pub.items():
        for a, b in itertools.combinations(sorted(ids), 2):
            comp[(a, b)] += 1

    print("pares con >=2 publicaciones compartidas:", sum(1 for x in comp.values() if x >= 2))
    for k in (2, 3, 4, 5):
        pares = {p: n for p, n in comp.items() if n >= k}
        cuentas = {i for p in pares for i in p}
        print(f"\n=== >= {k} publicaciones compartidas: {len(pares)} pares, {len(cuentas)} cuentas ===")
        # pureza de pares
        tp = sum(1 for (a, b) in pares if cam_de[a] and cam_de[a] == cam_de[b])
        print(f"  pares dentro de la misma campana: {tp}")

    # ---------- respuestas
    mapa = dict(zip(c.comentario_id, c.cuenta_id))
    rep = c[c.respuesta_a != ""].copy()
    rep["padre"] = rep.respuesta_a.map(mapa)
    rep = rep[rep.padre.notna() & (rep.padre != rep.cuenta_id)]
    resp = Counter()
    for a, b in zip(rep.cuenta_id, rep.padre):
        resp[(min(a, b), max(a, b))] += 1

    # ---------- puntaje de coordinacion por par
    def componentes(edges):
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
        return list(grupos.values())

    print("\n=== componentes conexas con >=3 publicaciones compartidas ===")
    for k in (3, 4, 5):
        pares = [p for p, n in comp.items() if n >= k]
        comps = componentes(pares)
        comps.sort(key=len, reverse=True)
        print(f"  k={k}: {len(comps)} componentes, tamanos top:", [len(x) for x in comps[:12]])
        for tam in comps[:3]:
            mezcla = Counter(cam_de[i] for i in tam)
            print(f"    tam={len(tam)} campanas={dict(mezcla)}")

    print("\n=== co-publicacion + respuestas ===")
    for kpub in (2, 3):
        for krep in (0, 1, 2):
            pares = [p for p, n in comp.items() if n >= kpub and resp.get(p, 0) >= krep]
            comps = componentes(pares)
            comps.sort(key=len, reverse=True)
            marcadas = {i for x in comps for i in x}
            grupos = {i: f"g{n}" for n, x in enumerate(comps) for i in x}
            evaluar(marcadas, grupos, cam_de, f"pub>={kpub} resp>={krep} comps={len(comps)}")

    # ---------- solo C2: cuentas con >=2 publicaciones y respuestas mutuas
    print("\n=== analisis C2 ===")
    c2 = {i for i, k in cam_de.items() if k == "C2"}
    for i in sorted(c2)[:5]:
        pares_i = {p: n for p, n in comp.items() if i in p}
        amigos = [(b if a == i else a) for (a, b) in pares_i]
        print(f"  {i}: pubs={len(pub_de[i])} amigos_pub>=2={len(amigos)} "
              f"campanas_amigos={Counter(cam_de[x] for x in amigos).most_common(3)} "
              f"respuestas={sum(n for (a, b), n in resp.items() if i in (a, b))}")


if __name__ == "__main__":
    main()
