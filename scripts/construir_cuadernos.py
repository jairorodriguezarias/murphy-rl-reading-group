"""Convierte las lecciones Markdown en notebooks con código realmente ejecutable."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "lecciones"
DEST = ROOT / "cuadernos"

def parse_lesson(text):
    """Solo los bloques ~~~python se ejecutan; el resto conserva Markdown."""
    chunks = re.split(r"(?m)^~~~python\s*\n(.*?)^~~~\s*$", text, flags=re.S)
    cells = []
    for index, chunk in enumerate(chunks):
        kind = "code" if index % 2 else "markdown"
        parts = [chunk] if kind == "code" else chunk.split("<!-- celda -->")
        for part in parts:
            source = part.strip()
            if not source:
                continue
            cell = {"cell_type": kind, "id": f"celda-{len(cells):03d}",
                    "metadata": {}, "source": source + "\n"}
            if kind == "code":
                compile(source, "<leccion>", "exec")
                cell.update(execution_count=None, outputs=[])
            cells.append(cell)
    return cells

def build_notebooks(chapter=None):
    DEST.mkdir(exist_ok=True)
    paths = sorted(SOURCE.glob("[0-9][0-9]_*.md"))
    if [int(p.stem.split("_", 1)[0]) for p in paths] != list(range(10)):
        raise ValueError("Se esperan preparación 00, capítulos 01–08 y taller 09.")
    if chapter is not None:
        paths = [p for p in paths if int(p.stem.split("_", 1)[0]) == chapter]
        if not paths:
            raise ValueError(f"No existe la lección {chapter}.")
    for path in paths:
        content = path.read_text(encoding="utf-8")
        cells = parse_lesson(content)
        nb = {"cells": cells, "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": ".".join(map(str, sys.version_info[:3]))},
            "murphy": {"formato": "taller_diagnostico" if path.stem.startswith("09_") else "capitulo_intercalado", "autocontenido": True,
                       "fuente": str(path.relative_to(ROOT)),
                       "sha256_fuente": hashlib.sha256(path.read_bytes()).hexdigest()}},
            "nbformat": 4, "nbformat_minor": 5}
        (DEST / (path.stem + ".ipynb")).write_text(
            json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        count = sum(c["cell_type"] == "code" for c in cells)
        print(f"{path.stem}: {len(cells)} celdas, {count} de código")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capitulo", type=int, choices=range(10), help="Generar solo un cuaderno (0 = preparación; 9 = taller de diagnóstico).")
    build_notebooks(parser.parse_args().capitulo)
