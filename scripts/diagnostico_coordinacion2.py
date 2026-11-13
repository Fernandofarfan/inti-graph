"""Diagnostico fino de coordinacion: solapamiento de publicaciones, respuestas mutuas, cuentas nuevas."""
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent


def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    cu = pd.read_csv(RAIZ / "datos/practica/cuentas.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/cuentas.csv.gz", dtype=str, keep_default_na=False)
    c["fecha"] = pd.to_datetime(c.fecha, utc=True)
    c["hora"] = c.fecha.dt.hour
    cu["fecha_creacion"] = pd.to_datetime(cu.fecha_creacion, utc=True)
    c = c.merge(cu[["cuenta_id", "fecha_creacion", "seguidores", "siguiendo"]], on="cuenta_id")
    c = c.merge(v, on="cuenta_id", how="left")
    c["campana_id"] = c.campana_id.fillna("")
    ini = c.fecha.min()
    cam_de = dict(zip(c.cuenta_id, c.campana_id))
    pub_de = c.groupby("cuenta_id").publicacion_id.apply(set).to_dict()
    fec_de = c.groupby("cuenta_id").fecha.apply(list).to_dict()
    antig = ((ini - cu.set_index("cuenta_id").fecha_creacion).dt.total_seconds() / 86400)

    # ---------- 1. antiguedad
    print("=== antiguedad (dias) por grupo ===")
    for cam in ("C1", "C2", "C3", ""):
        s = antig[[i for i in antig.index if cam_de.get(i, "") == cam]]
        print(f"  {cam or 'normal'}: n={len(s)} p1={s.quantile(.01):.1f} p5={s.quantile(.05):.1f} "
              f"p50={s.median():.1f} p95={s.quantile(.95):.1f} max={s.max():.0f}")
    for umbral in (7, 15, 30, 60, 90):
        nuevas = set(antig[antig < umbral].index)
        print(f"  antiguedad < {umbral}d: {len(nuevas)} cuentas | "
              + " ".join(f"{cam}={len([i for i in nuevas if cam_de.get(i)=='{cam}'])}" for cam in ("C1", "C2", "C3")))

    # ---------- 2. solapamiento de publicaciones
    print("\n=== solapamiento de publicaciones entre pares de la misma campana ===")
    for cam in ("C1", "C2", "C3"):
        ids = [i for i in pub_de if cam_de.get(i) == cam]
        jac, inter, cont = [], [], []
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                A, B = pub_de[ids[a]], pub_de[ids[b]]
                jac.append(len(A & B) / len(A | B))
                inter.append(len(A & B))
        print(f"  {cam}: pares={len(jac)} jaccard medio={np.mean(jac):.2f} "
              f"| pares con >=3 pub en comun={np.mean(np.array(inter)>=3):.2f}")

    # baseline: pares aleatorios de normales
    rng = np.random.default_rng(0)
    normales = [i for i in pub_de if cam_de.get(i) == ""]
    for n_pub in (4, 5, 6, 10, 19):
        sub = [i for i in normales if len(pub_de[i]) == n_pub]
        if len(sub) < 50:
            continue
        js, it = [], []
        for _ in range(2000):
            a, b = rng.choice(len(sub), 2, replace=False)
            A, B = pub_de[sub[a]], pub_de[sub[b]]
            js.append(len(A & B) / len(A | B))
            it.append(len(A & B))
        print(f"  normal n_pub={n_pub} ({len(sub)} cuentas): jaccard medio={np.mean(js):.3f} "
              f"| P(>=3 pub comun)={np.mean(np.array(it)>=3):.4f}")

    # ---------- 3. respuestas mutuas
    print("\n=== respuestas entre cuentas ===")
    mapa = dict(zip(c.comentario_id, c.cuenta_id))
    rep = c[c.respuesta_a != ""].copy()
    rep["padre"] = rep.respuesta_a.map(mapa)
    rep = rep[rep.padre.notna() & (rep.padre != rep.cuenta_id)]
    cuenta_rep = Counter()
    for a, b in zip(rep.cuenta_id, rep.padre):
        cuenta_rep[a] += 1
        cuenta_rep[b] += 1
    s = pd.Series(cuenta_rep)
    for cam in ("C1", "C2", "C3", ""):
        ids = [i for i in s.index if cam_de.get(i, "") == cam]
        print(f"  {cam or 'normal'}: cuentas con respuestas={len(ids)} "
              f"mediana={s[ids].median() if ids else 0:.0f} p90={s[ids].quantile(.9) if ids else 0:.0f}")
    # pares mutuos
    print("\n  pares de la misma campana que se responden:")
    pares = Counter(tuple(sorted((a, b))) for a, b in zip(rep.cuenta_id, rep.padre))
    for cam in ("C1", "C2", "C3"):
        ids = {i for i in pub_de if cam_de.get(i) == cam}
        dentro = sum(1 for (a, b) in pares if a in ids and b in ids)
        print(f"    {cam}: {len(ids)} cuentas, pares internos que se responden={dentro} "
              f"de {len(ids)*(len(ids)-1)//2}")

    # ---------- 4. textos exactos compartidos
    print("\n=== repeticion de texto entre cuentas de la campana ===")
    c["norm"] = c.texto.str.lower().str.replace(r"[^a-z0-9\u00f1 ]", " ", regex=True)
    for cam in ("C1", "C2", "C3"):
        ids = {i for i in pub_de if cam_de.get(i) == cam}
        sub = c[c.cuenta_id.isin(ids)]
        cnt = sub.norm.value_counts()
        rep_txt = cnt[cnt >= 3]
        docs = int(rep_txt.sum())
        print(f"  {cam}: comentarios={len(sub)} textos con >=3 usos={len(rep_txt)} docs={docs} "
              f"({docs/len(sub):.1%})")
    for cam in ("C1", "C3"):
        ids = {i for i in pub_de if cam_de.get(i) == cam}
        sub = c[c.cuenta_id.isin(ids)]
        print(f"\n  top textos {cam}:")
        for t, n in sub.norm.value_counts().head(5).items():
            print(f"    {n:3d} {t[:90]}")

    # ---------- 5. coincidencia temporal intra-publicacion (< 3 min)
    print("\n=== rafagas: comentarios de la misma campana en la misma publicacion a <3 min ===")
    for cam in ("C1", "C2", "C3"):
        ids = {i for i in pub_de if cam_de.get(i) == cam}
        sub = c[c.cuenta_id.isin(ids)].sort_values("fecha")
        n_cerca = 0
        for pub, s2 in sub.groupby("publicacion_id"):
            if len(s2) < 2:
                continue
            t = s2.fecha.values.astype("datetime64[s]").astype(np.int64)
            for i in range(len(t) - 1):
                if t[i + 1] - t[i] <= 180:
                    n_cerca += 1
        print(f"  {cam}: {n_cerca} pares consecutivos <3min (de {len(sub)} comentarios)")


if __name__ == "__main__":
    main()
