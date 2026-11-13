"""Diagnostico de senales de coordinacion por cuenta (C1 copy-paste, C2 antiguas sincronizadas, C3 nocturnas)."""
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent


def normalize(t):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[^a-z0-9\u00f1\s]", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1", t)
    return " ".join(t.split())


def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    cu = pd.read_csv(RAIZ / "datos/practica/cuentas.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/cuentas.csv.gz", dtype=str, keep_default_na=False)
    print("comentarios", len(c), "| cuentas en catalogo", len(cu), "| cuentas con actividad", c.cuenta_id.nunique())
    print("verdad campanas:", v[v.campana_id != ""].campana_id.value_counts().to_dict())

    c["fecha"] = pd.to_datetime(c.fecha, utc=True)
    c["hora"] = c.fecha.dt.hour
    c["nt"] = c.texto.str.split().str.len()
    c["norm"] = c.texto.map(normalize)
    cu["fecha_creacion"] = pd.to_datetime(cu.fecha_creacion, utc=True)
    c = c.merge(cu[["cuenta_id", "fecha_creacion", "seguidores", "siguiendo"]], on="cuenta_id", how="left")
    c = c.merge(v, on="cuenta_id", how="left")
    c["campana_id"] = c.campana_id.fillna("")

    ini = c.fecha.min()
    print("periodo:", ini, "->", c.fecha.max())

    # --- features por cuenta
    g = c.groupby("cuenta_id")
    f = pd.DataFrame({
        "n": g.size(),
        "nt_medio": g.nt.mean(),
        "unicos": g.norm.nunique(),
        "publicaciones": g.publicacion_id.nunique(),
        "noches": g.apply(lambda s: float((s.hora.between(1, 5)).mean()), include_groups=False),
        "seguidores": g.seguidores.first().astype(float),
        "antiguedad_dias": (ini - g.fecha_creacion.first()).dt.total_seconds() / 86400,
    })
    f["ratio_dup"] = 1 - f.unicos / f.n
    f["tasa_comentario"] = f.n / f.publicaciones
    f["campana"] = c.groupby("cuenta_id").campana_id.first()
    f = f.reset_index()

    print("\n=== features por campana ===")
    print(f.groupby("campana")[["n", "nt_medio", "unicos", "ratio_dup", "publicaciones",
                                "tasa_comentario", "noches", "antiguedad_dias", "seguidores"]]
          .median().round(2).to_string())
    print("\n=== cuantiles de las cuentas normales ===")
    nor = f[f.campana == ""]
    for col in ["n", "ratio_dup", "tasa_comentario", "noches", "antiguedad_dias"]:
        q = nor[col].quantile([0.5, 0.9, 0.99, 0.999]).round(2).to_dict()
        print(f"  {col}: {q}")

    print("\n=== distribucion de ratio_dup por campana ===")
    for cam in ("C1", "C2", "C3", ""):
        s = f[f.campana == cam].ratio_dup
        print(f"  {cam or 'normal'}: n={len(s)} median={s.median():.2f} p10={s.quantile(.1):.2f} p90={s.quantile(.9):.2f}")

    print("\n=== textos compartidos entre cuentas (mismo norm) ===")
    # cuantas cuentas distintas usan cada texto
    por_texto = c.groupby("norm").cuenta_id.nunique()
    print("textos usados por >1 cuenta:", int((por_texto > 1).sum()), "de", len(por_texto))
    cuenta_textos_comp = c[c.norm.isin(por_texto[por_texto >= 3].index)].groupby("cuenta_id").size()
    print("cuentas con >=1 texto compartido por >=3 cuentas:",
          int((cuenta_textos_comp > 0).sum()))
    tmp = f.set_index("cuenta_id").join(cuenta_textos_comp.rename("txt_comp")).fillna({"txt_comp": 0})
    print(tmp.groupby("campana").txt_comp.median().to_string())

    # --- co-actividad: pares de cuentas en la misma publicacion en ventanas cortas
    print("\n=== co-actividad en ventanas (misma publicacion) ===")
    # muestreo sobre cuentas candidatas + normales
    c2 = c.sort_values("fecha")
    # pares por publicacion con diferencia < 15 min
    pares = Counter()
    for pub, sub in c2.groupby("publicacion_id"):
        sub = sub[["cuenta_id", "fecha"]].sort_values("fecha")
        t = sub.fecha.values.astype("datetime64[s]").astype(np.int64)
        u = sub.cuenta_id.values
        # ventana deslizante
        j = 0
        for i in range(len(t)):
            if j < i:
                j = i
            while j + 1 < len(t) and t[j + 1] - t[i] <= 900:
                j += 1
            for k in range(i + 1, j + 1):
                if u[k] != u[i]:
                    pares[tuple(sorted((u[i], u[k])))] += 1
    print("pares con >=1 coactivacion <15min:", len(pares))
    if pares:
        cnt = Counter()
        for (a, b), n in pares.items():
            cnt[a] += n
            cnt[b] += n
        # top cuentas por coactividad
        rank = pd.Series(cnt).sort_values(ascending=False)
        cam_de = dict(zip(f.cuenta_id, f.campana))
        top = pd.DataFrame({"pares": rank.head(40)})
        top["campana"] = [cam_de.get(i, "") for i in top.index]
        print(top.to_string())

    # --- respuestas entre cuentas
    print("\n=== respuestas entre cuentas ===")
    mapa_cuenta = dict(zip(c.comentario_id, c.cuenta_id))
    rep = c[c.respuesta_a != ""].copy()
    rep["padre_cuenta"] = rep.respuesta_a.map(mapa_cuenta)
    rep = rep[rep.padre_cuenta.notna() & (rep.padre_cuenta != rep.cuenta_id)]
    print("respuestas entre cuentas distintas:", len(rep))
    par_rep = Counter(tuple(sorted((a, b))) for a, b in zip(rep.cuenta_id, rep.padre_cuenta))
    cnt = Counter()
    for (a, b), n in par_rep.items():
        cnt[a] += n
        cnt[b] += n
    rank = pd.Series(cnt).sort_values(ascending=False)
    top = pd.DataFrame({"respuestas": rank.head(30)})
    top["campana"] = [cam_de.get(i, "") for i in top.index]
    print(top.to_string())

    # --- grafo de coactividad -> comunidades
    print("\n=== cuentas de campana: cuantas aparecen en top coactividad ===")
    for cam in ("C1", "C2", "C3"):
        ids = set(f[f.campana == cam].cuenta_id)
        print(f"  {cam}: {len(ids)} cuentas, en pares <15min: {len([i for i in ids if i in cnt])}")


if __name__ == "__main__":
    main()
