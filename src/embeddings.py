"""Embeddings semanticos con MiniLM multilingue cuantizado (ONNX) y cache en disco."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .comun import SALIDA

DIM = 384
LOTE = 64
MAX_LEN = 64


def _sesion():
    import onnxruntime as ort
    from tokenizers import Tokenizer

    from . import modelo

    ruta_modelo = modelo.asegurar() / "model_quantized.onnx"
    if not ruta_modelo.exists():
        raise FileNotFoundError(
            f"Falta {ruta_modelo}. Corre scripts/descargar_modelo.py"
        )
    tok = Tokenizer.from_file(str(ruta_modelo.parent / "tokenizer.json"))
    tok.enable_truncation(max_length=MAX_LEN)
    tok.enable_padding()
    sess = ort.InferenceSession(str(ruta_modelo), providers=["CPUExecutionProvider"])
    return tok, sess


def codificar(textos: list[str], lote: int = LOTE, verbose: bool = True) -> np.ndarray:
    """Media ponderada por attention mask de los estados ocultos, normalizada L2."""
    tok, sess = _sesion()
    nombres = {i.name for i in sess.get_inputs()}
    salida = np.empty((len(textos), DIM), dtype=np.float32)
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
        if verbose and ini % (lote * 100) == 0:
            print(f"    embeddings {ini}/{len(textos)}", flush=True)
    norma = np.linalg.norm(salida, axis=1, keepdims=True)
    return (salida / np.clip(norma, 1e-9, None)).astype(np.float32)


def _huella(textos: list[str]) -> str:
    h = hashlib.blake2b(digest_size=8)
    for t in textos:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


def embeddings_unicos(textos: list[str], etiqueta: str = "base", verbose: bool = True) -> np.ndarray:
    """Embeddings de una lista de textos unicos con cache por huella de contenido."""
    SALIDA.mkdir(parents=True, exist_ok=True)
    huella = _huella(textos)
    cache = SALIDA / f"embeddings_{etiqueta}_{huella}.npz"
    if cache.exists():
        if verbose:
            print(f"    embeddings desde cache ({cache.name})", flush=True)
        return np.load(cache)["E"]
    E = codificar(textos, verbose=verbose)
    np.savez_compressed(cache, E=E)
    if verbose:
        print(f"    embeddings guardados en {cache.name}", flush=True)
    return E
