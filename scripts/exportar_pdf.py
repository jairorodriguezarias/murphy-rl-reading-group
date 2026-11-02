"""Exportación opcional: uv run --with playwright python scripts/exportar_pdf.py.
Requiere un Chromium instalado por Playwright, o --navegador /ruta/al/ejecutable.
"""
import argparse
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navegador")
    parser.add_argument("--capturas", type=Path)
    args = parser.parse_args()
    options = {"headless": True}
    if args.navegador:
        options["executable_path"] = args.navegador
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(**options)
        page = browser.new_page(viewport={"width": 1360, "height": 950})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto((ROOT/"GUIA_LECTURA.html").as_uri())
        page.wait_for_load_state("networkidle")
        assert not errors, errors
        assert page.locator("math").count() > 50
        assert page.locator("img").evaluate_all("(xs) => xs.every(x => x.complete && x.naturalWidth > 0)")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        if args.capturas:
            args.capturas.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(args.capturas/"portada.png"))
            page.locator("#docs-06_llms .formula").first.scroll_into_view_if_needed()
            page.screenshot(path=str(args.capturas/"formulas.png"))
            page.set_viewport_size({"width": 390, "height": 844})
            page.evaluate("scrollTo(0, 0)")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(args.capturas/"movil.png"))
        page.pdf(path=str(ROOT/"GUIA_LECTURA.pdf"), format="A4",
                 margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
                 print_background=True, display_header_footer=True, outline=True, tagged=True,
                 header_template="<span></span>",
                 footer_template='<div style="font-size:9px;width:100%;text-align:center;color:#587079">Guía de lectura · Murphy v5 · <span class="pageNumber"></span> / <span class="totalPages"></span></div>')
        browser.close()
    print("Guía renderizada sin errores ni desbordamiento horizontal; PDF exportado.")

if __name__ == "__main__":
    main()
