"""Detectar la comunidad plantada C2 en el grafo de respuestas (k-core, clustering local, Louvain)."""
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
    pub_de = c.groupby("cuenta_id").publicacion_id.apply(set).to_dict()
    n_de = c.groupby("cuenta_id").size().to_dict()

    m = dict(zip(c.comentario_id, c.cuenta_id))
    rep = c[c.respuesta_a != ""].copy()
    rep["padre"] = rep.respuesta_a.map(m)
    rep = rep[rep.padre.notna() & (rep.padre != rep.cuenta_id)]

    W = Counter()
    for a, b in zip(rep.cuenta_id, rep.padre):
        W[(min(a, b), max(a, b))] += 1

    # grafo no dirigido
    adj = defaultdict(set)
    for (a, b) in W:
        adj[a].add(b)
        adj[b].add(a)

    # ---- k-core
    def kcore(adj):
        grado = {u: len(vs) for u, vs in adj.items()}
        vivo = set(adj)
        k = 0
        while True:
            quitar = [u for u in vivo if grado[u] <= k]
            if not quitar:
                k += 1
                continue
            if len(vivo) - len(quitar) < k + 1:
                return k, vivo
            for u in quitar:
                vivo.discard(u)
                for w in adj[u]:
                    if w in vivo:
                        grado[w] -= 1
        return k, vivo

    k, nucleo = kcore(adj)
    print(f"k-core maximo={k} con {len(nucleo)} cuentas")
    print("  campanas:", dict(Counter(cam[i] for i in nucleo)))
    for kk in (2, 3, 4, 5):
        # extraer subgrafo por poda iterativa
        vivo = set(adj)
        grado = {u: len(vs) for u, vs in adj.items()}
        cambio = True
        while cambio:
            cambio = False
            for u in list(vivo):
                if grado[u] < kk:
                    vivo.discard(u)
                    for w in adj[u]:
                        if w in vivo:
                            grado[w] -= 1
                    cambio = True
        comps = []
        visto = set()
        for u in vivo:
            if u in visto:
                continue
            pila = [u]
            comp = set()
            while pila:
                x = pila.pop()
                if x in comp:
                    continue
                comp.add(x)
                visto.add(x)
                pila.extend(adj[x] & vivo - comp)
            comps.append(comp)
        comps.sort(key=len, reverse=True)
        print(f"\nk-core k={kk}: {len(vivo)} cuentas, {len(comps)} componentes, "
              f"top={[len(x) for x in comps[:10]]}")
        print("  campanas:", dict(Counter(cam[i] for i in vivo if cam[i])))

    # ---- clustering local sobre el grafo completo de respuestas
    print("\n=== coeficiente de clustering local (top) ===")
    filas = []
    for u in adj:
        vs = list(adj[u])
        if len(vs) < 5:
            continue
        e = sum(1 for i in range(len(vs)) for j in range(i + 1, len(vs)) if vs[j] in adj[vs[i]])
        pos = len(vs) * (len(vs) - 1) / 2
        filas.append((u, len(vs), e / pos if pos else 0.0, cam.get(u, "")))
    f = pd.DataFrame(filas, columns=["cuenta", "grado", "clustering", "campana"])
    print(f.groupby("campana")[["grado", "clustering"]].agg(["median", "max"]).round(3).to_string())

    print("\n  top 30 clustering (grado>=7):")
    print(f[f.grado >= 7].sort_values("clustering", ascending=False).head(30).to_string())

    # ---- componente conexa del subgrafo inducido por cuentas con clustering alto
    print("\n=== subgrafo de cuentas con grado>=6 y clustering>=0.25 ===")
    sel = set(f[(f.grado >= 6) & (f.clustering >= 0.25)].cuenta)
    print("  seleccionadas:", len(sel), "campanas:", dict(Counter(cam[i] for i in sel if cam[i])))
    print("  campanas totales presentes:", dict(Counter(cam[i] for i in sel)))


if __name__ == "__main__":
    main()
