"""Descarga el modelo de embeddings si falta.

Uso:
    python scripts/descargar_modelo.py [--forzar]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import modelo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forzar", action="store_true", help="vuelve a bajar todo")
    args = ap.parse_args()
    ruta = modelo.asegurar(forzar=args.forzar)
    print(f"modelo disponible en {ruta}")


if __name__ == "__main__":
    main()
