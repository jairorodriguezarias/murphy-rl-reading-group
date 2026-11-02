"""Comprueba la guía, las lecciones ejecutadas y los datos de referencia archivados."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import re
import numpy as np
from construir_cuadernos import parse_lesson

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

def check_links(text, parent):
    checked = 0
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        if target.startswith(("https://", "http://", "#", "mailto:")):
            continue
        assert (parent/target.split("#")[0]).exists(), (parent, target)
        checked += 1
    return checked

def source(cell):
    return "".join(cell["source"])

def main():
    markdown_files = [
        ROOT/"README.md", *sorted((ROOT/"docs").glob("*.md")),
        *sorted((ROOT/"lecciones").glob("*.md")), ROOT/"cuadernos/README.md",
        ROOT/"fuentes/README.md", ROOT/"resultados/INFORME.md",
        ROOT/"resultados/cuadernos/README.md",
    ]
    checked = sum(check_links(path.read_text(), path.parent) for path in markdown_files)
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
    assert [int(p.stem.split("_", 1)[0]) for p in notebooks] == list(range(9))
    lessons = []
    for path in notebooks:
        i = int(path.stem.split("_", 1)[0])
        nb = json.loads(path.read_text())
        meta = nb["metadata"]["murphy"]
        lesson = ROOT/meta["fuente"]
        assert lesson.stem == path.stem
        assert meta["sha256_fuente"] == hashlib.sha256(lesson.read_bytes()).hexdigest(), path
        expected = parse_lesson(lesson.read_text())
        assert [(c["cell_type"], source(c)) for c in nb["cells"]] == [
            (c["cell_type"], source(c)) for c in expected], path
        assert meta["ejecucion"]["directorio_aislado"] is True
        code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        markdown = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
        assert [c["execution_count"] for c in code] == list(range(1, len(code)+1)), path
        assert all(o["output_type"] != "error" for c in code for o in c["outputs"]), path
        images = sum("image/png" in o.get("data", {}) for c in code for o in c["outputs"])
        assert images > 0, path
        for cell in markdown:
            checked += check_links(source(cell), path.parent)
        lessons.append({
            "cuaderno": path.name, "celdas": len(nb["cells"]), "celdas_codigo": len(code),
            "palabras_markdown": sum(len(source(c).split()) for c in markdown),
            "figuras": images, "ejecutado_en_carpeta_vacia": True,
        })

        # El preparatorio 00 no corresponde a un experimento de referencia.
        if i == 0:
            continue
        # El archivo histórico es independiente de las nuevas lecciones.
        for filename in ["datos.npz", "metricas.csv", "resumen.json", "figura.png", "figura.svg"]:
            assert (ROOT/f"resultados/cap{i:02d}"/filename).stat().st_size > 0
        with np.load(ROOT/f"resultados/cap{i:02d}/datos.npz") as data:
            with np.load(ROOT/f"resultados/cuadernos/cap{i:02d}/datos.npz") as archived:
                assert set(data.files) == set(archived.files)
                for key in data.files:
                    assert np.isfinite(data[key]).all(), (i, key)
                    np.testing.assert_array_equal(data[key], archived[key])

    manifest = json.loads((ROOT/"resultados/manifest.json").read_text())
    for value in manifest.values():
        for name, sha in value["source_sha256"].items():
            assert hashlib.sha256((ROOT/"experimentos"/name).read_bytes()).hexdigest() == sha, name
    book = json.loads((ROOT/"fuentes/metadata.json").read_text())
    assert hashlib.sha256((ROOT/"fuentes/murphy_rl_2412.05265v5.pdf").read_bytes()).hexdigest() == book["sha256"]
    result = {
        "enlaces_locales_markdown": checked, "formulas_mathml": page.maths,
        "figuras_html_incrustadas": page.images, "cuadernos_ejecutados": len(notebooks),
        "celdas_codigo_ejecutadas": sum(x["celdas_codigo"] for x in lessons),
        "figuras_cuadernos": sum(x["figuras"] for x in lessons),
        "palabras_markdown_cuadernos": sum(x["palabras_markdown"] for x in lessons),
        "fuentes_y_celdas_coinciden": True,
        "datos_archivados_iguales_a_referencia": True,
        "hashes_codigo_y_pdf": "correctos", "detalle_cuadernos": lessons,
    }
    (ROOT/"resultados/validacion_material.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
