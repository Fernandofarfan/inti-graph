"""Rafagas: densidad de comentarios por publicacion en ventanas cortas (firma C2)."""
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
    c = c.sort_values("fecha").reset_index(drop=True)
    c["nt"] = c.texto.str.split().str.len()

    # numero de comentarios en la misma publicacion a +/- 30 min
    vecinos = np.zeros(len(c), dtype=np.int32)
    vecinos_15 = np.zeros(len(c), dtype=np.int32)
    for pub, sub in c.groupby("publicacion_id"):
        idx = sub.index.values
        t = sub.fecha.values.astype("datetime64[s]").astype(np.int64)
        j0 = 0
        j1 = 0
        j0b = 0
        for k in range(len(t)):
            if j0 < k:
                j0 = k
            while j0 < len(t) and t[j0] < t[k] - 1800:
                j0 += 1
            if j1 < k:
                j1 = k
            while j1 < len(t) and t[j1] <= t[k] + 1800:
                j1 += 1
            if j0b < k:
                j0b = k
            while j0b < len(t) and t[j0b] < t[k] - 900:
                j0b += 1
            # ventana +/-30 (simetrica aproximada usando j0..j1))
            vecinos[idx[k]] = j1 - j0
        # ventana 15 min simetrica
        j0 = 0
        j1 = 0
        for k in range(len(t)):
            if j0 < k:
                j0 = k
            while j0 < len(t) and t[j0] < t[k] - 900:
                j0 += 1
            if j1 < k:
                j1 = k
            while j1 < len(t) and t[j1] <= t[k] + 900:
                j1 += 1
            vecinos_15[idx[k]] = j1 - j0
    c["v30"] = vecinos
    c["v15"] = vecinos_15

    print("=== densidad de rafaga por campana ===")
    print(c.groupby("campana_id")[["v30", "v15"]].describe().round(1).to_string())
    print("\npercentiles normales v30:", np.percentile(c[c.campana_id == ""].v30, [50, 90, 99, 99.9]).round(1))

    # por cuenta: maximo y media de v30
    g = c.groupby("cuenta_id")
    f = pd.DataFrame({
        "v30_max": g.v30.max(),
        "v30_media": g.v30.mean(),
        "v15_max": g.v15.max(),
        "p25_v30": g.v30.quantile(.25),
        "p25_v15": g.v15.quantile(.25),
        "n": g.size(),
        "nt_medio": g.nt.mean(),
        "campana": g.campana_id.first(),
    })
    print("\n=== por cuenta ===")
    print(f.groupby("campana")[["v30_max", "v30_media", "v15_max", "p25_v30", "p25_v15", "n"]]
          .median().round(1).to_string())
    nor = f[f.campana == ""]
    for col in ("v30_max", "v30_media", "p25_v30", "p25_v15"):
        print(f"  normal {col}: {nor[col].quantile([.5, .9, .99, .999]).round(1).to_dict()}")
    for camp in ("C1", "C2", "C3"):
        s = f[f.campana == camp]
        print(f"  {camp}: " + " ".join(f"{col}={s[col].median():.0f}" for col in ("v30_max", "v30_media", "p25_v30")))

    # cuentas normales con p25_v15 alto
    print("\nnormales p25_v15>=6:", int((nor.p25_v15 >= 6).sum()), "| >=10:", int((nor.p25_v15 >= 10).sum()))
    print("normales v30_max>=20:", int((nor.v30_max >= 20).sum()))

    # grupos de rafaga: por publicacion, ventana 30 min, >=6 cuentas distintas
    print("\n=== grupos de rafaga (pub, ventana 30min, >=6 cuentas) ===")
    grupos = []
    for pub, sub in c.groupby("publicacion_id"):
        sub = sub.sort_values("fecha")
        t = sub.fecha.values.astype("datetime64[s]").astype(np.int64)
        u = sub.cuenta_id.values
        j = 0
        for i in range(len(t)):
            if j < i:
                j = i
            while j + 1 < len(t) and t[j + 1] - t[i] <= 1800:
                j += 1
            if j - i + 1 >= 6:
                ids = list(dict.fromkeys(u[i:j + 1]))
                if len(ids) >= 6:
                    grupos.append((pub, t[i], ids))
    # deduplicar por (pub, conjunto)
    vistos = set()
    unicos = []
    for pub, t0, ids in grupos:
        clave = (pub, frozenset(ids))
        if clave in vistos:
            continue
        vistos.add(clave)
        unicos.append((pub, t0, ids))
    print("grupos candidatos:", len(unicos))
    # cuentas que aparecen en el mejor grupo de cada publicacion
    cuentas = Counter()
    for pub, t0, ids in unicos:
        for i in ids:
            cuentas[i] += 1
    coord = {i for i, k in cam.items() if k}
    rank = pd.Series(cuentas).sort_values(ascending=False)
    print("\ntop cuentas por participacion en rafagas:")
    top = pd.DataFrame({"rafagas": rank.head(50)})
    top["campana"] = [cam.get(i, "") for i in top.index]
    print(top.to_string())
    print("\ncomposicion de campanas en top 200:", Counter(cam.get(i, "") for i in rank.head(200).index))


if __name__ == "__main__":
    main()
