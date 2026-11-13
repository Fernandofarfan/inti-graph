"""Buscar la firma exacta de C2 (cuentas antiguas sincronizadas que se responden)."""
import itertools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent


def b3(prec, rec):
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/cuentas.csv.gz", dtype=str, keep_default_na=False)
    c["fecha"] = pd.to_datetime(c.fecha, utc=True)
    c = c.merge(v, on="cuenta_id", how="left")
    c["campana_id"] = c.campana_id.fillna("")
    cam_de = dict(zip(c.cuenta_id, c.campana_id))
    pub_de = c.groupby("cuenta_id").publicacion_id.apply(set).to_dict()
    n_de = c.groupby("cuenta_id").size().to_dict()

    # replies
    mapa = dict(zip(c.comentario_id, c.cuenta_id))
    rep = c[c.respuesta_a != ""].copy()
    rep["padre"] = rep.respuesta_a.map(mapa)
    rep = rep[rep.padre.notna() & (rep.padre != rep.cuenta_id)]
    dir_rep = Counter((a, b) for a, b in zip(rep.cuenta_id, rep.padre))
    mutuo = set()
    for (a, b) in dir_rep:
        if (b, a) in dir_rep:
            mutuo.add((min(a, b), max(a, b)))
    print("pares de respuesta (dirigidos):", len(dir_rep), "| pares mutuos:", len(mutuo))

    def cobertura(a, b):
        A, B = pub_de[a], pub_de[b]
        return len(A & B) / min(len(A), len(B))

    # ==== 1. perfil por cuenta: con cuantas cuentas comparte publicaciones y se responde
    print("\n=== features de reciprocidad ===")
    amigos_rep = defaultdict(set)
    for (a, b) in dir_rep:
        amigos_rep[a].add(b)
        amigos_rep[b].add(a)
    amigos_mut = defaultdict(set)
    for (a, b) in mutuo:
        amigos_mut[a].add(b)
        amigos_mut[b].add(a)

    feats = []
    for i in pub_de:
        ar = amigos_rep.get(i, set())
        am = amigos_mut.get(i, set())
        cob = [cobertura(i, x) for x in ar]
        feats.append({
            "cuenta": i,
            "campana": cam_de.get(i, ""),
            "n": n_de[i],
            "pubs": len(pub_de[i]),
            "amigos_rep": len(ar),
            "amigos_mut": len(am),
            "cob_media_rep": float(np.mean(cob)) if cob else 0.0,
            "cob_max_rep": float(np.max(cob)) if cob else 0.0,
            "amigos_rep_cob50": sum(1 for x in cob if x >= 0.5),
        })
    f = pd.DataFrame(feats)
    print(f.groupby("campana")[["pubs", "amigos_rep", "amigos_mut", "cob_media_rep",
                                "cob_max_rep", "amigos_rep_cob50"]].median().round(2).to_string())
    nor = f[f.campana == ""]
    print("\ncuantiles normales:")
    for col in ["amigos_rep", "amigos_mut", "cob_media_rep", "cob_max_rep", "amigos_rep_cob50"]:
        print(f"  {col}: {nor[col].quantile([.5, .9, .99, .999]).round(2).to_dict()}")

    # ==== 2. reglas simples de marcado
    print("\n=== reglas sobre cuentas ===")
    coord = {i for i, k in cam_de.items() if k}
    for nombre, m in [
        ("2+ amigos respuesta", f.amigos_rep >= 2),
        ("3+ amigos respuesta", f.amigos_rep >= 3),
        ("2+ amigos mutuos", f.amigos_mut >= 2),
        ("cob_max_rep>=0.4", f.cob_max_rep >= 0.4),
        ("cob_max_rep>=0.4 y amigos>=2", (f.cob_max_rep >= 0.4) & (f.amigos_rep >= 2)),
        ("cob_media_rep>=0.2 y amigos>=2", (f.cob_media_rep >= 0.2) & (f.amigos_rep >= 2)),
        ("amigos_rep_cob50>=1", f.amigos_rep_cob50 >= 1),
        ("amigos_rep_cob50>=2", f.amigos_rep_cob50 >= 2),
        ("amigos_rep_cob50>=2 | amigos_mut>=3", (f.amigos_rep_cob50 >= 2) | (f.amigos_mut >= 3)),
    ]:
        ids = set(f[nombre and m].cuenta)
        tp = len(ids & coord)
        fp = len(ids - coord)
        fn = len(coord - ids)
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / len(coord)
        print(f"  {nombre}: marcadas={len(ids)} P={p:.3f} R={r:.3f} "
              f"| por campana marcadas={dict(Counter(cam_de[i] for i in ids))}")

    # ==== 3. componente conexa del grafo de respuestas mutuas con cobertura
    print("\n=== componentes de grafo de respuestas (mutuas) ===")
    for umbral in (0.3, 0.4, 0.5, 0.6):
        edges = [(a, b) for (a, b) in mutuo if cobertura(a, b) >= umbral]
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
        ids = set().union(*comps) if comps else set()
        tp = len(ids & coord)
        print(f"  cob>={umbral}: {len(comps)} comps, cuentas={len(ids)} "
              f"campanas={dict(Counter(cam_de[i] for i in ids if cam_de[i]))} "
              f"tamanos={[len(x) for x in comps[:8]]}")

    # ==== 4. pares C2: como se conectan
    print("\n=== detalle pares C2 ===")
    c2 = [i for i in cam_de if cam_de[i] == "C2"]
    for i in c2[:6]:
        amigos = [(x, round(cobertura(i, x), 2)) for x in amigos_rep.get(i, set())]
        amigos.sort(key=lambda t: -t[1])
        print(f"  {i}: amigos={len(amigos)} top={amigos[:6]}")

    # ==== 5. timing: misma publicacion, diferencia temporal
    print("\n=== sincronia temporal en la misma publicacion ===")
    for cam in ("C1", "C2", "C3", ""):
        ids = {i for i, k in cam_de.items() if k == cam}
        if cam == "":
            ids = set(list(ids)[:2000])
        sub = c[c.cuenta_id.isin(ids)]
        difs = []
        for pub, s in sub.groupby("publicacion_id"):
            t = sorted(s.fecha.values.astype("datetime64[s]").astype(np.int64))
            difs.extend(np.diff(t).tolist())
        difs = np.array(difs)
        if len(difs) == 0:
            continue
        print(f"  {cam or 'normal'}: mediana diff={np.median(difs):.0f}s "
              f"P(<120s)={np.mean(difs < 120):.3f} P(<600s)={np.mean(difs < 600):.3f}")


if __name__ == "__main__":
    main()
