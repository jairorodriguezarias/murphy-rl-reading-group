"""Comprueba que el taller detecta los fallos y acepta reparaciones sin alterar sus pruebas."""
import ast
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import re
import unittest
from unittest.mock import patch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.construir_cuadernos import parse_lesson

ROOT = Path(__file__).resolve().parents[1]
LESSON = ROOT/"lecciones/09_detectives_rl.md"

def ejecutar_taller(reparar=None):
    texto = LESSON.read_text(encoding="utf-8")
    soluciones = {}
    fence = chr(96)*3
    for bloque in re.findall(rf"(?ms)^{fence}python\n(.*?)^{fence}\s*$", texto):
        for nodo in ast.parse(bloque).body:
            if isinstance(nodo, ast.FunctionDef):
                soluciones[nodo.name] = nodo
    if len(soluciones) != 8:
        raise AssertionError("Deben existir ocho soluciones desplegables")
    elegidas = set(soluciones) if reparar == "todas" else set(reparar or [])
    namespace = {}
    sustituidas = set()
    with redirect_stdout(io.StringIO()), patch("matplotlib.pyplot.show", side_effect=lambda: plt.close("all")):
        for celda in parse_lesson(texto):
            if celda["cell_type"] != "code":
                continue
            arbol = ast.parse(celda["source"])
            for i, nodo in enumerate(arbol.body):
                if isinstance(nodo, ast.FunctionDef) and nodo.name in elegidas:
                    arbol.body[i] = copy.deepcopy(soluciones[nodo.name])
                    sustituidas.add(nodo.name)
            exec(compile(ast.fix_missing_locations(arbol), str(LESSON), "exec"), namespace)
    if sustituidas != elegidas:
        raise AssertionError(f"Soluciones sin aplicar: {elegidas-sustituidas}")
    return namespace

class Taller(unittest.TestCase):
    def test_version_inicial_detecta_ocho_fallos(self):
        ns = ejecutar_taller()
        registro = ns["REGISTRO"]
        self.assertEqual(set(registro), {f"R{i}" for i in range(1, 9)})
        self.assertTrue(all(not r["resuelto"] for r in registro.values()))
        self.assertGreater(sum(r["pasan"] for r in registro.values()), 0)
        # El panel debe volver a comprobar, no conservar un aprobado antiguo.
        ns["REGISTRO"]["R1"]["resuelto"] = True
        with redirect_stdout(io.StringIO()):
            ns["probar"]("R1", ns["BATERIA"]["R1"])
        self.assertFalse(ns["REGISTRO"]["R1"]["resuelto"])

    def test_soluciones_pasan_todas_las_comprobaciones(self):
        ns = ejecutar_taller("todas")
        self.assertTrue(all(r["resuelto"] for r in ns["REGISTRO"].values()))
        self.assertTrue(all(r["pasan"] == r["total"] for r in ns["REGISTRO"].values()))

    def test_una_reparacion_no_aprueba_los_otros_retos(self):
        ns = ejecutar_taller({"retorno_reto"})
        resueltos = {k for k, v in ns["REGISTRO"].items() if v["resuelto"]}
        self.assertEqual(resueltos, {"R1"})

if __name__ == "__main__":
    unittest.main()
