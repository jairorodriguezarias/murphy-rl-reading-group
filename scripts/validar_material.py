"""Comprueba enlaces, fórmulas, resultados finitos y cuadernos ejecutados."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.refs = set(), []
        self.maths = self.images = self.scripts = 0
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            assert attrs["id"] not in self.ids, attrs["id"]
            self.ids.add(attrs["id"])
        self.maths += tag == "math"
        self.scripts += tag == "script"
        if tag == "img":
            self.images += 1
            assert attrs.get("src", "").startswith("data:image/png;base64,")
            assert attrs.get("alt"), "Figura sin texto alternativo"
        if "href" in attrs:
            self.refs.append(attrs["href"])

def main():
    checked = 0
    for path in [ROOT/"README.md", *sorted((ROOT/"docs").glob("*.md")), ROOT/"fuentes/README.md",
                 ROOT/"resultados/INFORME.md"]:
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            assert (path.parent/target.split("#")[0]).exists(), (path, target)
            checked += 1
    page = Page()
    raw = (ROOT/"GUIA_LECTURA.html").read_text()
    page.feed(raw)
    assert page.maths > 50, page.maths
    assert page.images == 8, page.images
    assert page.scripts == 0
    assert "MATHPLACEHOLDER" not in raw
    for target in page.refs:
        if target.startswith("#"):
            assert target[1:] in page.ids, target
        elif not target.startswith(("http://", "https://", "mailto:")):
            assert (ROOT/target).exists(), target
    notebooks = sorted((ROOT/"cuadernos").glob("*.ipynb"))
    assert len(notebooks) == 8
    for i, path in enumerate(notebooks, start=1):
        nb = json.loads(path.read_text())
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert all(c["execution_count"] is not None for c in code), path
        assert all(o["output_type"] != "error" for c in code for o in c["outputs"]), path
        assert any("image/png" in o.get("data", {}) for c in code for o in c["outputs"]), path
        for filename in ["datos.npz", "metricas.csv", "resumen.json", "figura.png", "figura.svg"]:
            assert (ROOT/f"resultados/cap{i:02d}"/filename).stat().st_size > 0
        with np.load(ROOT/f"resultados/cap{i:02d}/datos.npz") as data:
            with np.load(ROOT/f"resultados/cuadernos/cap{i:02d}/datos.npz") as notebook_data:
                assert set(data.files) == set(notebook_data.files)
                for key in data.files:
                    assert np.isfinite(data[key]).all(), (i, key)
                    np.testing.assert_array_equal(data[key], notebook_data[key])
    manifest = json.loads((ROOT/"resultados/manifest.json").read_text())
    for value in manifest.values():
        for name, sha in value["source_sha256"].items():
            assert hashlib.sha256((ROOT/"experimentos"/name).read_bytes()).hexdigest() == sha, name
    source = json.loads((ROOT/"fuentes/metadata.json").read_text())
    assert hashlib.sha256((ROOT/"fuentes/murphy_rl_2412.05265v5.pdf").read_bytes()).hexdigest() == source["sha256"]
    result = {"enlaces_locales_markdown": checked, "formulas_mathml": page.maths,
              "figuras_html_incrustadas": page.images, "cuadernos_ejecutados": len(notebooks),
              "datos_cuadernos_iguales_a_referencia": True, "hashes_codigo_y_pdf": "correctos"}
    (ROOT/"resultados/validacion_material.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
