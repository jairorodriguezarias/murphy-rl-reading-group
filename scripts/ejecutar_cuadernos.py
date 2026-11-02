"""Ejecuta cada notebook en una carpeta vacía y guarda sus salidas en el repositorio."""
from pathlib import Path
import argparse
import os
import tempfile
import time
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

def main(chapter=None):
    paths = sorted((ROOT/"cuadernos").glob("*.ipynb"))
    if chapter is not None:
        paths = [p for p in paths if int(p.stem.split("_", 1)[0]) == chapter]
        if not paths:
            raise ValueError(f"No existe el cuaderno {chapter}.")
    with tempfile.TemporaryDirectory(prefix="murphy-jupyter-") as runtime:
        os.environ["JUPYTER_RUNTIME_DIR"] = runtime
        for path in paths:
            start = time.perf_counter()
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            with tempfile.TemporaryDirectory(prefix="murphy-cuaderno-") as isolated:
                client = NotebookClient(nb, timeout=180, kernel_name="python3",
                                        resources={"metadata": {"path": isolated}})
                client.execute()
            elapsed = time.perf_counter()-start
            nb.metadata["murphy"]["ejecucion"] = {
                "directorio_aislado": True, "segundos": round(elapsed, 2)}
            nbformat.write(nb, path)
            print(f"{path.name}: ejecutado en carpeta vacía ({elapsed:.1f} s)", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capitulo", type=int, choices=range(10), help="Ejecutar solo un cuaderno (0 = preparación; 9 = taller de diagnóstico).")
    main(parser.parse_args().capitulo)
