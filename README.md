# Grupo de lectura · Reinforcement Learning: An Overview

Guía en español de **Kevin P. Murphy, arXiv:2412.05265v5**, con conocimientos previos, resúmenes, preguntas de discusión y laboratorios reproducibles.

**Empieza por [el notebook 00 de preparación](cuadernos/00_preparacion.ipynb)** y continúa con [los capítulos en notebooks](cuadernos/README.md). Para leer sin ejecutar código, consulta [la guía resumida para navegador](GUIA_LECTURA.html) o [la guía en PDF](GUIA_LECTURA.pdf). Puedes empezar por [los conocimientos previos](docs/00_conocimientos_previos.md) y consultar [el PDF original descargado](fuentes/murphy_rl_2412.05265v5.pdf). La guía HTML contiene las figuras y las fórmulas; se puede leer sin conexión.

La v5 tiene 253 páginas: siete capítulos de contenido, el capítulo 8 de agradecimientos y bibliografía. Incluimos un laboratorio por capítulo de contenido y una práctica metodológica propia para el apartado 8. Los experimentos son ilustraciones pequeñas, no reproducciones de los grandes sistemas del libro.

## Material por capítulo

| Capítulo | Páginas PDF | Resumen | Notebook: lectura y código |
|---|---:|---|---|
| 00. Preparación (añadido) | — | [Conocimientos previos](docs/00_conocimientos_previos.md) | [NumPy, matemáticas y autoevaluación](cuadernos/00_preparacion.ipynb) |
| 1. Introduction | 13–30 | [Introducción](docs/01_introduccion.md) | [Retorno, memoria, creencias y shaping](cuadernos/01_introduccion.ipynb) |
| 2. Value-based RL | 31–50 | [Valores](docs/02_valores.md) | [Bellman, TD, Q-learning y sobreestimación](cuadernos/02_valores.ipynb) |
| 3. Policy-based RL | 51–82 | [Políticas](docs/03_politicas.md) | [REINFORCE, GAE, PPO y entropía](cuadernos/03_politicas.ipynb) |
| 4. Model-based RL | 83–120 | [Modelos](docs/04_modelos.md) | [MPC, Dyna y representación sucesora](cuadernos/04_modelos.ipynb) |
| 5. Multi-agent RL | 121–146 | [Multiagente](docs/05_multiagente.md) | [Equilibrios, cooperación y regret matching](cuadernos/05_multiagente.ipynb) |
| 6. LLMs and RL | 147–172 | [LLM](docs/06_llms.md) | [RLOO, KL, DPO y selección de respuestas](cuadernos/06_llms.ipynb) |
| 7. Other topics in RL | 173–198 | [Otros temas](docs/07_otros_temas.md) | [Exploración, riesgo, jerarquías y offline RL](cuadernos/07_otros_temas.ipynb) |
| 8. Acknowledgements | 199 | [Agradecimientos](docs/08_agradecimientos.md) | [Selección, intervalos y reproducibilidad](cuadernos/08_agradecimientos.ipynb) |
| 09. Taller añadido | — | Diagnosticar, corregir y comprobar | [Detectives de RL: ocho programas con fallos](cuadernos/09_detectives_rl.ipynb) |

Complementos: [plan de sesiones y glosario](docs/09_plan_y_glosario.md), [resultados observados](resultados/INFORME.md), [protocolo experimental](docs/10_reproducibilidad.md), [procedencia y atribución](fuentes/README.md).

**Para ponerte a prueba:** abre [09 · Detectives de RL](cuadernos/09_detectives_rl.ipynb). Incluye ocho funciones con errores intencionados, 34 comprobaciones, pistas y soluciones desplegables. El punto de partida muestra 0/8 retos resueltos: debes editar las funciones y volver a comprobarlas.

## Ejecutar los experimentos

Requieren Python 3.11 o posterior, NumPy y Matplotlib. Funcionan en CPU, sin cuentas, claves API, modelos descargados ni datasets externos. La instalación inicial de dependencias requiere acceso a Internet.

Con **uv**:

~~~bash
uv sync --frozen
uv run python -m experimentos.ejecutar --capitulo todos
uv run python -m experimentos.ejecutar --capitulo 4 --semilla 2026 --replicas 20
~~~

Con **Python y pip**:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m experimentos.ejecutar --capitulo todos
~~~

En Windows, activa el entorno con .venv\Scripts\activate. Los comandos se ejecutan desde la carpeta raíz del proyecto.

La ejecución completa de referencia usa 20 réplicas por experimento. Guarda en resultados/capXX/ una figura PNG y otra SVG, un resumen JSON, métricas finales CSV y datos NumPy comprimidos. El manifiesto registra parámetros, versiones, tiempos y hashes del código. Usa **--salida resultados/mi_prueba** para conservar intacta la ejecución de referencia.

El capítulo 8 simula 2.000 estudios de 20 semillas por algoritmo; su parámetro --replicas controla el mínimo de estudios mediante max(2000, replicas*100). En el capítulo 7 las trayectorias se generan por lotes a partir de la semilla base.

## Abrir los cuadernos

~~~bash
uv sync --frozen --extra notebooks
uv run --extra notebooks jupyter lab cuadernos
~~~

Los diez notebooks incluyen el preparatorio 00, ocho lecturas de capítulos y un taller de diagnóstico, con más de 15.000 palabras de explicación, 119 celdas de código y 32 figuras. El preparatorio añade doce preguntas con corrección y una ruta de repaso; las respuestas pendientes se conservan como pendientes. Alternan conceptos, derivaciones, cálculos, experimentos y ejercicios con soluciones orientativas. El código está implementado en sus celdas y cada notebook funciona por separado con NumPy y Matplotlib; todos se han ejecutado en carpetas vacías. Ya contienen sus salidas. Selecciona el kernel del entorno del proyecto si tu editor usa otro Python.

Consulta [el índice y la forma de trabajar](cuadernos/README.md). Los notebooks tienen ejemplos y presupuestos propios; las ejecuciones originales de los módulos siguen disponibles en resultados/capXX/. El directorio resultados/cuadernos/ conserva los datos de la primera versión, como indica [su nota de archivo](resultados/cuadernos/README.md).

## Verificar y regenerar

~~~bash
uv run python -m unittest discover -s tests -v
uv run --extra docs python scripts/construir_material.py
uv run --extra notebooks python scripts/ejecutar_cuadernos.py
uv run --extra docs python scripts/validar_material.py
~~~

El constructor regenera la guía HTML, el informe y los cuadernos **sin salidas**; después el ejecutor de cuadernos los ejecuta y vuelve a guardar sus salidas. Para editar las lecciones usa [las fuentes de lecciones/](lecciones/README.md). Los resúmenes breves viven en docs/ y los experimentos de referencia en experimentos/. Para regenerar solo los notebooks, usa **uv run python scripts/construir_cuadernos.py** y luego el ejecutor; guarda antes cualquier edición manual que quieras conservar.

Para regenerar el PDF, instala Chromium con **uv run --with playwright playwright install chromium** y ejecuta **uv run --with playwright python scripts/exportar_pdf.py**. También puedes usar la opción Imprimir del navegador.

## Alcance

Se asume Python básico y se incluye repaso matemático. Las bandas de las curvas son intervalos normales aproximados del 95% de la media entre réplicas, no garantías de rendimiento ni intervalos simultáneos sobre toda la curva. Las gráficas no usan suavizado.

Los laboratorios tabulares o categóricos aíslan ideas; no entrenan DQN, PPO completo, Dreamer ni un LLM real. Los notebooks añaden ejemplos de ingredientes como GAE, el objetivo PPO y la pérdida DPO. El del capítulo 7 incluye prácticas sobre exploración, distribuciones de retorno, curiosidad, opciones, imitación y datos offline. Las limitaciones concretas aparecen junto a cada práctica.

**Fuente:** [Reinforcement Learning: An Overview, Kevin P. Murphy, v5](https://arxiv.org/abs/2412.05265v5), publicado bajo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Estos resúmenes en español son una adaptación; los ejemplos, el código y las figuras experimentales se han creado para esta guía. No implican revisión ni respaldo del autor.
