"""Ejecuta y guarda las salidas de todos los cuadernos desde su propia carpeta."""
from pathlib import Path
import os
import tempfile
import time
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

def main():
    # Los ficheros de conexión temporales no se escriben en la configuración del usuario.
    with tempfile.TemporaryDirectory(prefix="murphy-jupyter-") as runtime:
        os.environ["JUPYTER_RUNTIME_DIR"] = runtime
        for path in sorted((ROOT/"cuadernos").glob("*.ipynb")):
            start = time.perf_counter()
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            client = NotebookClient(nb, timeout=180, kernel_name="python3",
                                    resources={"metadata": {"path": str(path.parent)}})
            client.execute()
            nbformat.write(nb, path)
            print(f"{path.name}: ejecutado ({time.perf_counter()-start:.1f} s)", flush=True)

if __name__ == "__main__":
    main()
