"""Uso: python -m experimentos.ejecutar --capitulo todos --replicas 20."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import matplotlib

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capitulo", choices=["todos"]+[str(i) for i in range(1, 9)], default="todos")
    parser.add_argument("--semilla", type=int, default=2026)
    parser.add_argument("--replicas", type=int, default=20)
    parser.add_argument("--salida", type=Path, default=Path("resultados"))
    args = parser.parse_args()
    if args.replicas < 2:
        parser.error("Usa al menos dos réplicas para estimar la variabilidad.")
    if args.semilla < 0:
        parser.error("La semilla debe ser no negativa.")
    chapters = range(1, 9) if args.capitulo == "todos" else [int(args.capitulo)]
    manifest_path = args.salida / "manifest.json"
    args.salida.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.glob("*.py"))}
    for chapter in chapters:
        module = importlib.import_module(f"experimentos.cap{chapter:02d}")
        started = time.perf_counter()
        metrics = module.ejecutar(args.salida/f"cap{chapter:02d}", seed=args.semilla, replicas=args.replicas)
        elapsed = time.perf_counter()-started
        manifest[f"cap{chapter:02d}"] = {
            "semilla": args.semilla, "replicas": args.replicas, "segundos": elapsed,
            "python": platform.python_version(), "numpy": np.__version__,
            "matplotlib": matplotlib.__version__, "source_sha256": source_hashes}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
        print(f"Capítulo {chapter}: completado en {elapsed:.2f} s → {args.salida / f'cap{chapter:02d}'}", flush=True)

if __name__ == "__main__":
    main()
