"""Modelo de scoring por cuenta: separabilidad de C2 con features estructurales (validacion cruzada)."""
import pathlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

RAIZ = pathlib.Path(__file__).resolve().parent.parent


def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str, keep_default_na=False)
    cu = pd.read_csv(RAIZ / "datos/practica/cuentas.csv.gz", dtype=str, keep_default_na=False)
    v = pd.read_csv(RAIZ / "datos/practica/verdad/cuentas.csv.gz", dtype=str, keep_default_na=False)
    c["fecha"] = pd.to_datetime(c.fecha, utc=True)
    cu["fecha_creacion"] = pd.to_datetime(cu.fecha_creacion, utc=True)
    ini = c.fecha.min()
    cam = dict(zip(v.cuenta_id, v.campana_id))
    c["campana"] = c.cuenta_id.map(cam).fillna("")
    c["hora"] = c.fecha.dt.hour
    c["nt"] = c.texto.str.split().str.len()
    c["norm"] = (c.texto.str.lower().str.replace(r"http\S+", " ", regex=True)
                 .str.replace(r"[^a-z0-9\u00f1 ]", " ", regex=True)
                 .str.replace(r"\s+", " ", regex=True).str.strip())

    g = c.groupby("cuenta_id")
    f = pd.DataFrame({
        "n": g.size(),
        "pubs": g.publicacion_id.nunique(),
        "nt": g.nt.mean(),
        "noches": g.apply(lambda s: float((s.hora.between(1, 5)).mean()), include_groups=False),
        "hora_std": g.apply(lambda s: float(s.hora.std()), include_groups=False),
        "unicos": g.norm.nunique(),
        "rep_ratio": g.apply(lambda s: float((s.respuesta_a != "").mean()), include_groups=False),
        "seguidores": g.apply(lambda s: 0.0, include_groups=False),
    })
    f["seguidores"] = cu.set_index("cuenta_id").seguidores.astype(float).reindex(f.index)
    f["siguiendo"] = cu.set_index("cuenta_id").siguiendo.astype(float).reindex(f.index)
    f["antiguedad"] = ((ini - cu.set_index("cuenta_id").fecha_creacion).dt.total_seconds() / 86400).reindex(f.index)
    f["ratio_dup"] = 1 - f.unicos / f.n
    f["usuario_len"] = cu.set_index("cuenta_id").usuario.str.len().reindex(f.index)
    f["sufijo_digitos"] = cu.set_index("cuenta_id").usuario.str.extract(r"(\d+)$")[0].fillna("0").str.len().reindex(f.index).astype(int)

    # replies
    m = dict(zip(c.comentario_id, c.cuenta_id))
    rep = c[c.respuesta_a != ""].copy()
    rep["padre"] = rep.respuesta_a.map(m)
    rep = rep[rep.padre.notna() & (rep.padre != rep.cuenta_id)]
    amigos = defaultdict(set)
    for a, b in zip(rep.cuenta_id, rep.padre):
        amigos[a].add(b)
        amigos[b].add(a)
    f["amigos_rep"] = pd.Series({i: len(a) for i, a in amigos.items()}).reindex(f.index).fillna(0)

    pub_de = c.groupby("cuenta_id").publicacion_id.apply(set).to_dict()
    cob = {}
    for u, vs in amigos.items():
        vals = [len(pub_de[u] & pub_de[x]) / min(len(pub_de[u]), len(pub_de[x])) for x in vs]
        cob[u] = float(np.mean(vals)) if vals else 0.0
    f["cob_amigos"] = pd.Series(cob).reindex(f.index).fillna(0)

    txt_cuentas = c.groupby("norm").cuenta_id.nunique()
    f["score_plantilla"] = c.groupby("cuenta_id").norm.apply(
        lambda s: float(np.mean([txt_cuentas[t] for t in s])))
    # longitud media de textos
    f["nt_std"] = g.nt.std().fillna(0)
    f["campana"] = [cam.get(i, "") for i in f.index]
    f["y"] = (f.campana != "").astype(int)
    f = f.fillna(0.0)

    cols = ["n", "pubs", "nt", "noches", "hora_std", "rep_ratio", "seguidores", "siguiendo",
            "antiguedad", "ratio_dup", "usuario_len", "sufijo_digitos", "amigos_rep",
            "cob_amigos", "score_plantilla", "nt_std"]
    X = f[cols].astype(float).values
    y = f.y.values
    print("X", X.shape, "positivos", y.sum())

    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for nombre, modelo in [
        ("logreg", lambda: LogisticRegression(max_iter=2000, C=0.5)),
        ("rf", lambda: RandomForestClassifier(400, random_state=0, n_jobs=1, min_samples_leaf=2)),
        ("gb", lambda: GradientBoostingClassifier(random_state=0, n_estimators=200, max_depth=3)),
    ]:
        oof = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            esc = StandardScaler().fit(X[tr])
            mdl = modelo().fit(esc.transform(X[tr]), y[tr])
            oof[te] = mdl.predict_proba(esc.transform(X[te]))[:, 1]
        print(f"\n{nombre}: AUC global={roc_auc_score(y, oof):.4f}")
        for camp in ("C1", "C2", "C3"):
            msk = (f.campana == camp).values
            print(f"  {camp}: score medio={oof[msk].mean():.3f} min={oof[msk].min():.3f} "
                  f"p10={np.percentile(oof[msk], 10):.3f}")
        print(f"  negativos: media={oof[y == 0].mean():.4f} p99={np.percentile(oof[y == 0], 99):.4f} "
              f"p99.9={np.percentile(oof[y == 0], 99.9):.4f} max={oof[y == 0].max():.4f}")
        for th in (0.1, 0.2, 0.3, 0.5, 0.7):
            sel = oof >= th
            tp = int((sel & (y == 1)).sum())
            fp = int((sel & (y == 0)).sum())
            por = Counter(f.campana[sel & (y == 1)])
            print(f"  th={th}: marcadas={sel.sum()} TP={tp} FP={fp} P={tp/max(sel.sum(),1):.3f} "
                  f"R={tp/y.sum():.3f} por_campana={dict(por)}")


if __name__ == "__main__":
    main()
