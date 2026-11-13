"""Descarga y verificacion del modelo de embeddings (MiniLM multilingue ONNX int8).

El modelo pesa ~118 MB y supera el limite de archivo de GitHub, por eso no se
versiona: se baja una sola vez desde Hugging Face (licencia Apache-2.0) y queda en
`modelos/minilm/`. El pipeline llama a `asegurar()` antes de usarlo, de modo que un
clon nuevo solo necesita conexion la primera vez.
"""
from __future__ import annotations

import json
import os
import shutil
import urllib.request
from pathlib import Path

from .comun import MODELOS

REPO = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
ARCHIVOS = {
    "model_quantized.onnx": f"{BASE}/onnx/model_quantized.onnx",
    "tokenizer.json": f"{BASE}/tokenizer.json",
    "tokenizer_config.json": f"{BASE}/tokenizer_config.json",
    "special_tokens_map.json": f"{BASE}/special_tokens_map.json",
}
MIN_BYTES = {
    "model_quantized.onnx": 100_000_000,
    "tokenizer.json": 10_000_000,
    "tokenizer_config.json": 100,
    "special_tokens_map.json": 100,
}


def _descargar(url: str, destino: Path, verbose: bool = True) -> None:
    tmp = destino.with_suffix(destino.suffix + ".part")
    peticion = urllib.request.Request(url, headers={"User-Agent": "inti-graph/1.0"})
    with urllib.request.urlopen(peticion, timeout=120) as respuesta, tmp.open("wb") as f:
        total = int(respuesta.headers.get("Content-Length") or 0)
        leido = 0
        while True:
            bloque = respuesta.read(1024 * 1024)
            if not bloque:
                break
            f.write(bloque)
            leido += len(bloque)
            if verbose and total:
                print(f"    {destino.name}: {100 * leido / total:5.1f}%", end="\r", flush=True)
    if verbose:
        print(" " * 40, end="\r")
    os.replace(tmp, destino)


def asegurar(destino: Path | None = None, forzar: bool = False,
             verbose: bool = True) -> Path:
    """Garantiza que el modelo y el tokenizer esten disponibles en disco."""
    destino = destino or (MODELOS / "minilm")
    destino.mkdir(parents=True, exist_ok=True)
    faltan = [n for n in ARCHIVOS
              if forzar or not (destino / n).exists()
              or (destino / n).stat().st_size < MIN_BYTES[n]]
    if not faltan:
        return destino
    if verbose:
        print(f"[modelo] faltan {len(faltan)} archivo(s); descargando {REPO}")
    for nombre in faltan:
        try:
            _descargar(ARCHIVOS[nombre], destino / nombre, verbose=verbose)
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo descargar {nombre} desde {ARCHIVOS[nombre]}: {exc}. "
                f"Si no hay red, copia el modelo a mano en {destino}."
            ) from exc
    if verbose:
        print(f"[modelo] listo en {destino}")
    return destino


def estado(destino: Path | None = None) -> dict:
    destino = destino or (MODELOS / "minilm")
    return {n: (destino / n).exists() for n in ARCHIVOS}


if __name__ == "__main__":
    print(json.dumps(estado(), ensure_ascii=False, indent=2))
