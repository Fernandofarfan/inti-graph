"""Embeddings semanticos (MiniLM multilingue cuantizado ONNX) + clustering de narrativas."""
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.preprocessing import normalize as l2
from tokenizers import Tokenizer

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from evaluacion.puntuar import bcubed  # noqa: E402

MODELO = RAIZ / "modelos" / "minilm"
CACHE = RAIZ / "salida" / "embeddings_unicos.npz"

ABREV = {"q": "que", "xq": "porque", "x": "por", "tmb": "tambien", "pa": "para",
         "tdo": "todo", "d": "de", "xfa": "por favor"}
STOP = set("""a al algo con como de del el ella en es esa ese eso esta este esto la las le lo los mas me mi
muy no nos o para pero por que se si sin su sus te todo todos un una uno y ya ahi aca hay""".split())


def normalize(t):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[^a-z0-9\u00f1\s]", " ", t)
    t = re.sub(r"(\w)\1{2,}", r"\1", t)
    return " ".join(ABREV.get(w, w) for w in t.split())


def codificar(textos, lote=64, maxlen=64):
    tok = Tokenizer.from_file(str(MODELO / "tokenizer.json"))
    tok.enable_truncation(max_length=maxlen)
    tok.enable_padding()
    sess = ort.InferenceSession(str(MODELO / "model_quantized.onnx"),
                                providers=["CPUExecutionProvider"])
    nombres = {i.name for i in sess.get_inputs()}
    salida = np.empty((len(textos), 384), dtype=np.float32)
    t0 = time.time()
    for ini in range(0, len(textos), lote):
        bloque = textos[ini:ini + lote]
        enc = tok.encode_batch(bloque)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.float32)
        feed = {"input_ids": ids, "attention_mask": mask.astype(np.int64)}
        if "token_type_ids" in nombres:
            feed["token_type_ids"] = np.zeros_like(ids)
        h = sess.run(None, feed)[0]
        peso = mask[:, :, None]
        salida[ini:ini + len(bloque)] = (h * peso).sum(1) / np.clip(peso.sum(1), 1e-9, None)
        if ini % (lote * 50) == 0:
            print(f"  {ini}/{len(textos)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  embeddings en {time.time()-t0:.1f}s", flush=True)
    return l2(salida)


def main():
    c = pd.read_csv(RAIZ / "datos/practica/comentarios.csv.gz", dtype=str,
                    keep_default_na=False, usecols=["comentario_id", "texto"])
    v = pd.read_csv(RAIZ / "datos/practica/verdad/comentarios.csv.gz", dtype=str,
                    keep_default_na=False, usecols=["comentario_id", "narrativa_id"])
    d = c.merge(v, on="comentario_id", how="left")
    d["nt"] = d.texto.str.split().str.len()
    d["norm"] = d.texto.map(normalize)
    mask = d.nt >= 9
    uniq = (d[mask].groupby("norm", as_index=False)
            .agg(narrativa_id=("narrativa_id", "first"), peso=("comentario_id", "size")))
    norm2idx = {x: i for i, x in enumerate(uniq.norm)}
    cid_norm = dict(zip(d.comentario_id[mask], d.norm[mask]))
    verdad = dict(zip(d.comentario_id[mask], d.narrativa_id[mask]))
    y = np.asarray(uniq.narrativa_id, dtype=object)
    print("textos unicos:", len(uniq), flush=True)

    if CACHE.exists():
        E = np.load(CACHE)["E"]
        print("embeddings desde cache", E.shape)
    else:
        E = codificar(uniq.norm.tolist())
        (RAIZ / "salida").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, E=E)
        print("embeddings guardados", E.shape)

    def ev(lab, nombre):
        pred = {cid: f"n{lab[norm2idx[nm]]}" for cid, nm in cid_norm.items()}
        p, r, f1 = bcubed(verdad, pred)
        tam = np.bincount(lab)
        print(f"  {nombre}: grupos={len(tam)} max={tam.max()} P={p:.3f} R={r:.3f} F1={f1:.3f}")
        return f1

    Ec = l2(E - E.mean(0, keepdims=True))
    for nombre, X in (("crudo", E), ("centrado", Ec)):
        print(f"\n--- {nombre} ---")
        for nc in (11, 13, 15, 18, 24):
            lab = MiniBatchKMeans(n_clusters=nc, random_state=0, n_init=10,
                                  batch_size=8192, max_iter=500).fit_predict(X)
            ev(lab, f"kmeans k={nc}")

    print("\n--- dos etapas: kmeans fino + ward sobre centroides ---")
    for kf in (40, 60, 90, 130):
        km = MiniBatchKMeans(n_clusters=kf, random_state=0, n_init=5, batch_size=8192,
                             max_iter=500).fit_predict(Ec)
        C = np.vstack([Ec[km == i].mean(0) for i in range(kf)])
        C = l2(C)
        D = 1.0 - C @ C.T
        np.fill_diagonal(D, 0.0)
        D = np.clip((D + D.T) / 2.0, 0, None)
        Z = linkage(squareform(D, checks=False), method="average")
        for nc in (13, 16, 20):
            g = fcluster(Z, t=nc, criterion="maxclust")
            ev(g[km], f"kf={kf} ward nc={nc}")

    print("\n--- HDBSCAN ---")
    for mcs, ms in ((100, 10), (300, 10), (800, 20)):
        t0 = time.time()
        lab = HDBSCAN(min_cluster_size=mcs, min_samples=ms, n_jobs=1).fit_predict(Ec)
        print(f"  hdbscan mcs={mcs} ({time.time()-t0:.0f}s)", end=" ")
        ev(lab, "hdbscan")


if __name__ == "__main__":
    main()
