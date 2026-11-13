"""Ranking de percentiles: que features separan a las cuentas de cada campana."""
import pathlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

RAIZ = pathlib.Path(__file__).resolve().parent.parent


def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    cu = pd.read_csv(RAIZ / "datos/practica/cuentas.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/cuentas.csv.gz", dtype=str, keep_default_na=False)
    c["fecha"] = pd.to_datetime(c.fecha, utc=True)
    cu["fecha_creacion"] = pd.to_datetime(cu.fecha_creacion, utc=True)
    ini = c.fecha.min()
    c = c.merge(cu[["cuenta_id", "fecha_creacion", "seguidores", "siguiendo"]], on="cuenta_id")
    c = c.merge(v, on="cuenta_id", how="left")
    c["campana_id"] = c.campana_id.fillna("")
    cam = dict(zip(c.cuenta_id, c.campana_id))

    c["hora"] = c.fecha.dt.hour
    c["nt"] = c.texto.str.split().str.len()
    c["norm"] = (c.texto.str.lower().str.replace(r"http\S+", " ", regex=True)
                 .str.replace(r"[^a-z0-9\u00f1 ]", " ", regex=True)
                 .str.replace(r"\s+", " ", regex=True).str.strip())
    pub_de = c.groupby("cuenta_id").publicacion_id.apply(set).to_dict()
    m = dict(zip(c.comentario_id, c.cuenta_id))
    rep = c[c.respuesta_a != ""].copy()
    rep["padre"] = rep.respuesta_a.map(m)
    rep = rep[rep.padre.notna() & (rep.padre != rep.cuenta_id)]
    amigos = defaultdict(set)
    for a, b in zip(rep.cuenta_id, rep.padre):
        amigos[a].add(b)
        amigos[b].add(a)

    g = c.groupby("cuenta_id")
    f = pd.DataFrame({
        "n": g.size(),
        "pubs": g.publicacion_id.nunique(),
        "noches": g.apply(lambda s: float((s.hora.between(1, 5)).mean()), include_groups=False),
        "hora_std": g.apply(lambda s: float(s.hora.std()), include_groups=False),
        "nt": g.nt.mean(),
        "seguidores": g.seguidores.first().astype(float),
        "siguiendo": g.siguiendo.first().astype(float),
        "unicos": g.norm.nunique(),
        "amigos_rep": pd.Series({i: len(a) for i, a in amigos.items()}),
    })
    f["ratio_dup"] = 1 - f.unicos / f.n
    f["antiguedad"] = ((ini - cu.set_index("cuenta_id").fecha_creacion).dt.total_seconds() / 86400)
    f["rep_ratio"] = g.apply(lambda s: float((s.respuesta_a != "").mean()), include_groups=False)
    f["tasa"] = f.n / f.pubs
    f["seguidores"] = f.seguidores.fillna(0)
    f["siguiendo"] = f.siguiendo.fillna(0)
    f["amigos_rep"] = f.amigos_rep.fillna(0)
    # co-publicacion media con amigos de respuesta
    cob = {}
    for u, vs in amigos.items():
        vals = [len(pub_de[u] & pub_de[x]) / min(len(pub_de[u]), len(pub_de[x])) for x in vs]
        cob[u] = float(np.mean(vals)) if vals else 0.0
    f["cob_amigos"] = pd.Series(cob).reindex(f.index).fillna(0)
    # repeticion de texto con otras cuentas
    txt_cuentas = c.groupby("norm").cuenta_id.nunique()
    tmp = c.groupby("cuenta_id").norm.apply(lambda s: float(np.mean([txt_cuentas[t] for t in s])))
    f["score_plantilla"] = tmp

    f["campana"] = [cam[i] for i in f.index]
    f.to_csv(RAIZ / "salida" / "features_cuentas.csv", encoding="utf-8")

    cols = ["n", "pubs", "noches", "hora_std", "nt", "seguidores", "siguiendo", "ratio_dup",
            "antiguedad", "rep_ratio", "tasa", "amigos_rep", "cob_amigos", "score_plantilla"]
    print("=== mediana por campana ===")
    print(f.groupby("campana")[cols].median().round(2).to_string())

    print("\n=== percentil medio de cada campana (0-100) ===")
    for camp in ("C1", "C2", "C3"):
        idx = f.campana == camp
        pct = {col: round(float(f[col].rank(pct=True)[idx].mean() * 100), 1) for col in cols}
        print(f"  {camp}: {pct}")

    print("\n=== cuentas normales con pubs >= 15 ===")
    nor = f[f.campana == ""]
    print("  normales pubs>=15:", int((nor.pubs >= 15).sum()), "de", len(nor))
    print("  normales pubs>=12:", int((nor.pubs >= 12).sum()))
    print("  normales ratio_dup>=0.15:", int((nor.ratio_dup >= 0.15).sum()))
    print("  normales noches>=0.4:", int((nor.noches >= 0.4).sum()))
    print("  normales antiguedad<10:", int((nor.antiguedad < 10).sum()))

    print("\n=== C2 vs normales con pubs>=15: amigos_rep y cob_amigos ===")
    print(f[f.pubs >= 15].groupby("campana")[["amigos_rep", "cob_amigos", "noches", "seguidores"]]
          .median().round(2).to_string())

    # C2: rank percentil individual
    print("\n=== C2 individual: percentiles ===")
    for col in ("pubs", "amigos_rep", "cob_amigos", "n"):
        s = f[col].rank(pct=True)
        print(f"  {col}: " + " ".join(f"{int(s[i]*100):3d}" for i in f[f.campana == 'C2'].index[:30]))


if __name__ == "__main__":
    main()
