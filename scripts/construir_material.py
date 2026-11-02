"""Genera informe, cuadernos y una guía HTML autocontenida (MathML y PNG incrustados)."""
from pathlib import Path
import base64
import hashlib
import html
import json
import re
import sys
import markdown
from latex2mathml.converter import convert
from construir_cuadernos import build_notebooks

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT/"docs").glob("*.md"))

def read_json(path):
    return json.loads(path.read_text())

def build_report():
    data = {i: read_json(ROOT/f"resultados/cap{i:02d}/resumen.json") for i in range(1, 9)}
    def fmt(d):
        return f'{d["media"]:.4f} ± {d["ic95_aprox_semianchura"]:.4f}'
    rows = [
        ("1", "Acierto sin memoria", fmt(data[1]["sin_memoria"])),
        ("1", "Acierto con memoria", fmt(data[1]["con_memoria"])),
        ("2", "RMSE final de Q", fmt(data[2]["rmse_final"])),
        ("2", "Valor de política final", fmt(data[2]["valor_final"])),
        ("3", "Retorno sin baseline, restando 5", fmt(data[3]["final"]["Sin baseline"])),
        ("3", "Retorno con baseline, restando 5", fmt(data[3]["final"]["Baseline aprendido"])),
        ("4", "RMSE con 0 pasos de planificación", fmt(data[4]["0"]["rmse_final"])),
        ("4", "RMSE con 5 pasos de planificación", fmt(data[4]["5"]["rmse_final"])),
        ("5", "NashConv de políticas actuales", fmt(data[5]["nashconv_actual_final"])),
        ("5", "NashConv de políticas medias", fmt(data[5]["nashconv_media_final"])),
        ("6", "Corrección, proxy sin KL", fmt(data[6]["Proxy β=0"]["acierto_final"])),
        ("6", "Corrección, proxy con β=2", fmt(data[6]["Proxy β=2"]["acierto_final"])),
        ("6", "Corrección, verificador con β=0.1", fmt(data[6]["Verificador β=0.1"]["acierto_final"])),
    ]
    lines = ["# Resultados observados", "",
        "Ejecución de referencia: semilla base 2026, 20 réplicas. Los valores con ± indican media y semianchura del IC normal aproximado del 95%. Las cifras se generan a partir de los JSON; no son resultados del libro.", "",
        "| Cap. | Métrica | Resultado |", "|---|---|---:|"]
    lines += [f"| {c} | {metric} | {value} |" for c, metric, value in rows]
    lines += ["", "## Varianza del gradiente", "",
        f'Con política fija, la traza de covarianza baja de {data[3]["varianza_sin_baseline"]:.3f} a {data[3]["varianza_con_baseline"]:.3f}. La reducción de varianza no produce una mejora final clara en la recompensa de este ejemplo.', "",
        "## Exploración y cambio del entorno", "",
        "| Método | Regret estacionario | Regret cambiante |", "|---|---:|---:|"]
    for method in data[7]["Estacionario"]:
        lines.append(f'| {method} | {fmt(data[7]["Estacionario"][method])} | {fmt(data[7]["Cambiante"][method])} |')
    lines += ["", "Thompson estándar destaca en el entorno fijo. El olvido ayuda a Thompson en el cambiante, pero UCB1 logra aquí el menor regret cambiante. No hay un ganador universal.", "",
        "## Sesgo de selección", "",
        f'Con algoritmos de igual calidad real, comparar medias produce {data[8]["diferencia_media_observada"]:.4f}; seleccionar la mejor semilla de A produce {data[8]["ventaja_seleccionando_mejor_semilla"]:.4f}; evaluar de nuevo produce {data[8]["ventaja_evaluacion_independiente"]:.4f}. Son promedios de 2.000 estudios sintéticos.', "",
        "## Tiempo y alcance", ""]
    manifest = read_json(ROOT/"resultados/manifest.json")
    elapsed = sum(v["segundos"] for v in manifest.values())
    lines += [f"Los ocho módulos tardaron {elapsed:.2f} segundos en esta máquina, sin incluir instalación, creación de cuadernos ni validación. El tiempo en otro equipo puede variar.", "",
        "Dyna compara presupuestos iguales de interacción, con cómputo diferente. El laboratorio de LLM es categórico. Las curvas no establecen resultados generales ni reproducen benchmarks de redes profundas.", "",
        "Consulta los JSON y arrays por capítulo para más métricas y precisión. No se ha elegido solo la mejor semilla."]
    (ROOT/"resultados/INFORME.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

def render_doc(path):
    text = path.read_text()
    maths = {}
    def math_token(m):
        display = m.group(1) is not None
        tex = (m.group(1) if display else m.group(2)).strip()
        key = f"MATHPLACEHOLDER{len(maths):05d}END"
        value = convert(tex, display="block" if display else "inline")
        maths[key] = f'<div class="formula">{value}</div>' if display else value
        return "\n\n"+key+"\n\n" if display else key
    text = re.sub(r"\$\$(.+?)\$\$|\$([^\n$]+)\$", math_token, text, flags=re.S)
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    for key, value in maths.items():
        if value.startswith("<div"):
            body = body.replace("<p>"+key+"</p>", value)
        body = body.replace(key, value)
    def image_src(m):
        name = m.group(1)
        if name.startswith(("http", "data:")):
            return m.group(0)
        image_path = (path.parent/name).resolve()
        encoded = base64.b64encode(image_path.read_bytes()).decode()
        return 'src="data:image/png;base64,'+encoded+'"'
    body = re.sub(r'src="([^"]+)"', image_src, body)
    # Ajustar enlaces locales para que la página pueda abrirse desde la raíz.
    def href(m):
        target = m.group(1)
        if target.startswith(("http:", "https:", "#", "mailto:")):
            return m.group(0)
        resolved = (path.parent/target).resolve()
        return 'href="'+str(resolved.relative_to(ROOT))+'"'
    body = re.sub(r'href="([^"]+)"', href, body)
    return body

def build_html():
    docs = DOCS + [ROOT/"resultados/INFORME.md", ROOT/"fuentes/README.md"]
    toc, sections = [], []
    for path in docs:
        anchor = path.parent.name+"-"+path.stem
        title = path.read_text().splitlines()[0].lstrip("# ")
        toc.append(f'<a href="#{anchor}">{html.escape(title)}</a>')
        sections.append(f'<section id="{anchor}">'+render_doc(path)+"</section>")
    css = """
:root{--ink:#203239;--muted:#587079;--accent:#086b69;--line:#d6e2e2;--paper:#fff}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:#f1f5f4;font:17px/1.7 system-ui,sans-serif}
header{background:#103d44;color:#fff;padding:60px 7vw 45px}header p{max-width:850px;color:#d5e7e8}
header .tag{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:#a6e1d2}
h1{font-size:2.25rem;line-height:1.2;letter-spacing:-.03em}h2{font-size:1.4rem;margin-top:2.3rem;color:var(--accent)}
.layout{display:grid;grid-template-columns:260px minmax(0,950px);gap:35px;max-width:1350px;margin:35px auto;padding:0 25px}
nav{align-self:start;position:sticky;top:20px;max-height:94vh;overflow:auto;font-size:13px}
nav a{display:block;padding:9px 12px;border-left:2px solid var(--line);text-decoration:none}
nav a:hover{background:#dfece9;border-color:var(--accent)}a{color:var(--accent)}
main{min-width:0}section{background:var(--paper);padding:42px 48px;margin-bottom:30px;border:1px solid var(--line);border-radius:10px;scroll-margin-top:20px}
section h1{font-size:1.9rem}table{width:100%;border-collapse:collapse;font-size:.88em;display:block;overflow-x:auto}
th{text-align:left;background:#edf4f2}th,td{padding:10px 13px;border-bottom:1px solid var(--line)}
img{width:100%;height:auto;border:1px solid var(--line);border-radius:6px}
pre{overflow:auto;background:#102e36;color:#e3f1ee;padding:18px;border-radius:7px;font-size:.83em;line-height:1.5}
code{font-family:ui-monospace,monospace}.formula{overflow-x:auto;padding:18px;background:#f4f8f7;margin:20px 0}math{font-size:1.05em}
p,li,td,th{overflow-wrap:anywhere}math[display=inline]{display:inline-block;max-width:100%;overflow-x:auto;vertical-align:middle}li{margin:.35em 0}footer{padding:30px;text-align:center;font-size:13px;color:var(--muted)}
@media(max-width:950px){.layout{display:block;padding:0 12px}nav{position:static;max-height:260px;margin-bottom:25px}section{padding:25px}header{padding:35px 25px}}
@media print{body{background:white;font-size:11pt;line-height:1.45}h2{margin-top:1.5rem}section h1{font-size:20pt}.formula{padding:10px;margin:12px 0}th,td{padding:6px 8px}p{orphans:3;widows:3}header{background:white;color:#203239;padding:0}header p{color:#203239}nav{display:block;position:static;max-height:none;columns:2;margin:20px 0}nav a{border:0;padding:4px}.layout{display:block;margin:0;padding:0}section{border:0;padding:0;break-before:page}h1,h2{break-after:avoid}img,table,.formula{break-inside:avoid}}
"""
    document = '<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Murphy · Guía del grupo de lectura</title><style>'+css+'</style></head><body>'
    document += '<header><div class="tag">Grupo de lectura · arXiv v5</div><h1>Aprender a decidir</h1><p>Guía de <em>Reinforcement Learning: An Overview</em>, de Kevin P. Murphy. Conocimientos previos, resúmenes y ocho laboratorios en CPU, con resultados y preguntas para discutir.</p><p>Las explicaciones y experimentos son una adaptación didáctica propia. Siete capítulos técnicos y un cierre metodológico para los agradecimientos.</p></header>'
    document += '<div class="layout"><nav aria-label="Índice">'+''.join(toc)+'<a href="README.md">Instrucciones y cuadernos</a></nav><main>'+''.join(sections)+'</main></div>'
    document += '<footer>Fuente: Kevin P. Murphy · arXiv:2412.05265v5 · CC BY 4.0. Guía adaptada en español; sin respaldo implícito del autor. Fórmulas MathML y figuras incluidas para lectura sin conexión.</footer></body></html>'
    (ROOT/"GUIA_LECTURA.html").write_text(document, encoding="utf-8")

if __name__ == "__main__":
    build_report()
    build_notebooks()
    build_html()
    print("Generados: informe, cuadernos y GUIA_LECTURA.html")
