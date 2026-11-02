"""Ejecuta cada notebook en una carpeta vacía y guarda sus salidas en el repositorio."""
from pathlib import Path
import os
import tempfile
import time
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

def main():
    with tempfile.TemporaryDirectory(prefix="murphy-jupyter-") as runtime:
        os.environ["JUPYTER_RUNTIME_DIR"] = runtime
        for path in sorted((ROOT/"cuadernos").glob("*.ipynb")):
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
    main()
