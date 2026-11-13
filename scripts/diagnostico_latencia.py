"""Latencia de respuesta: senal de coordinacion C2."""
import pandas as pd

RAIZ = __file__.rsplit("\\", 2)[0]


def main():
    import pathlib
    raiz = pathlib.Path(__file__).resolve().parent.parent
    c = pd.read_csv(raiz / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(raiz / "datos/practica/verdad/cuentas.csv.gz", dtype=str, keep_default_na=False)
    c["fecha"] = pd.to_datetime(c.fecha, utc=True)
    c = c.merge(v, on="cuenta_id", how="left")
    c["campana_id"] = c.campana_id.fillna("")
    cam = dict(zip(c.cuenta_id, c.campana_id))
    fec = dict(zip(c.comentario_id, c.fecha))
    m = dict(zip(c.comentario_id, c.cuenta_id))
    rep = c[c.respuesta_a != ""].copy()
    rep["padre"] = rep.respuesta_a.map(m)
    rep["tpadre"] = rep.respuesta_a.map(fec)
    rep = rep[rep.padre.notna() & (rep.padre != rep.cuenta_id)]
    rep["lat"] = (rep.fecha - rep.tpadre).dt.total_seconds()
    rep["cA"] = rep.cuenta_id.map(cam)
    rep["cB"] = rep.padre.map(cam)

    print("latencia de respuesta (s) por tipo de par:")
    for k in ("C1", "C2", "C3", ""):
        s = rep[(rep.cA == k) & (rep.cB == k)].lat
        if len(s):
            print(f"  {k or 'normal'}: n={len(s)} mediana={s.median():.0f} "
                  f"p10={s.quantile(.1):.0f} p90={s.quantile(.9):.0f} "
                  f"P(<300s)={(s < 300).mean():.3f} P(<60s)={(s < 60).mean():.3f}")

    print("\npares entre campanas distintas:")
    cross = rep[(rep.cA != "") & (rep.cB != "") & (rep.cA != rep.cB)]
    print(cross.groupby(["cA", "cB"]).size().to_string() if len(cross) else "  (ninguno)")

    print("\ncampana -> normal:")
    print(rep[(rep.cA != "") & (rep.cB == "")].groupby("cA").size().to_string())
    print("normal -> campana:")
    print(rep[(rep.cA == "") & (rep.cB != "")].groupby("cB").size().to_string())

    print("\n--- latencia intra-C2 ---")
    sub = rep[(rep.cA == "C2") & (rep.cB == "C2")]
    print(sub.lat.describe().round(1).to_string())
    print("\n--- latencia intra-C1 ---")
    sub = rep[(rep.cA == "C1") & (rep.cB == "C1")]
    print(sub.lat.describe().round(1).to_string())
    print("\n--- latencia intra-C3 ---")
    sub = rep[(rep.cA == "C3") & (rep.cB == "C3")]
    print(sub.lat.describe().round(1).to_string())

    # cuentas C2: a quien responden y quien les responde
    print("\n--- socios de respuesta de las cuentas C2 ---")
    c2 = [i for i in cam if cam[i] == "C2"]
    for i in c2[:8]:
        socios = rep[(rep.cuenta_id == i) | (rep.padre == i)]
        contra = socios.apply(lambda r: r.padre if r.cuenta_id == i else r.cuenta_id, axis=1)
        print(f"  {i}: socios={len(socios)} de_campana={sum(1 for x in contra if cam.get(x))} "
              f"lat_mediana={socios.lat.median():.0f}s")


if __name__ == "__main__":
    main()
