# Leer el libro ejecutando código

Estos diez notebooks incluyen un preparatorio, ocho lecturas de capítulos y un taller de diagnóstico en español: explicación, ecuaciones, ejemplos, código editable, interpretación y ejercicios con soluciones orientativas. Incluyen más de 15.000 palabras de explicación, **119 celdas de código y 32 figuras** ya ejecutadas.

**Empieza por [00 · Preparación](00_preparacion.ipynb).** Practica NumPy, probabilidad, Bayes, álgebra, cálculo y evaluación; después rellena doce preguntas. El corrector distingue respuestas pendientes e incorrectas e indica qué apartado repasar. Las soluciones razonadas están desplegadas al final solo cuando las abres.

**Para practicar corrigiendo errores, abre [09 · Detectives de RL](09_detectives_rl.ipynb).** Son ocho retos con 34 comprobaciones: descuento, terminales, softmax por lotes, GAE, RLOO, KL, fuga de datos y réplicas repetidas. Las funciones iniciales están mal a propósito; los mensajes FALLA y el panel 0/8 son el punto de partida. Corrige una función, reejecuta sus pruebas y consulta las pistas o soluciones cuando las necesites.

Cada notebook funciona por separado con NumPy y Matplotlib. El código está escrito en sus celdas: no importa módulos del proyecto, no lanza comandos externos y no necesita archivos de datos. Para comprobarlo, los ejecutamos en carpetas temporales vacías.

| Capítulo | Abre el notebook | Qué vas a calcular y modificar |
|---|---|---|
| 00 · Preparatorio | [Conocimientos previos](00_preparacion.ipynb) | Arrays y ejes, esperanza, Bayes, matrices, gradientes, softmax y autoevaluación |
| 1 | [Introducción](01_introduccion.ipynb) | Retorno y descuento, memoria con ruido, creencias bayesianas y shaping |
| 2 | [Valores](02_valores.ipynb) | Modelo de un MDP, Bellman, TD y Monte Carlo, Q-learning, sesgo del máximo y HER |
| 3 | [Políticas](03_politicas.ipynb) | Gradiente de REINFORCE, baselines, GAE, objetivo PPO, importancia y entropía |
| 4 | [Modelos](04_modelos.ipynb) | Error acumulado, control lineal y MPC, selección en un árbol, Dyna y representación sucesora |
| 5 | [Multiagente](05_multiagente.ipynb) | Mejores respuestas, NashConv, cooperación, correlación, Shapley y regret matching |
| 6 | [LLM y RL](06_llms.ipynb) | RLOO, gradiente KL, óptimo analítico, recompensa proxy, DPO, best-of-N y máscaras |
| 7 | [Otros temas](07_otros_temas.ipynb) | Bandits, incertidumbre, riesgo, curiosidad, opciones, imitación y conservadurismo offline |
| 8 | [Cierre metodológico](08_agradecimientos.ipynb) | Selección de semillas, evaluación independiente, intervalos y diseño de experimentos |
| 09 · Taller añadido | [Detectives de RL](09_detectives_rl.ipynb) | Ocho errores intencionados, pruebas, pistas y soluciones para corregirlos |

## Cómo usarlos

Desde la raíz del repositorio:

~~~bash
uv sync --frozen --extra notebooks
uv run --extra notebooks jupyter lab cuadernos
~~~

Abre el capítulo y elige **Restart Kernel and Run All Cells** para reproducirlo. También puedes usar VS Code con un kernel que tenga NumPy y Matplotlib. GitHub permite leer las explicaciones y las salidas guardadas sin instalar nada.

1. Lee una sección y predice qué mostrará el siguiente bloque.
2. Ejecuta el código y compara con tu predicción.
3. Cambia un parámetro y vuelve a ejecutar desde la celda que lo define.
4. Haz el ejercicio antes de desplegar su solución.
5. Lleva al grupo una conclusión respaldada y una limitación.

Se asumen variables, funciones y bucles básicos de Python. El [notebook 00](00_preparacion.ipynb) prepara las herramientas matemáticas y de NumPy; el [repaso breve](../docs/00_conocimientos_previos.md) sirve como consulta de notación. El preparatorio es material añadido, no un capítulo 0 del libro. Las secciones remiten al capítulo correspondiente de [Murphy, v5](https://arxiv.org/abs/2412.05265v5). Son una adaptación didáctica propia, no una reproducción completa del texto. El capítulo 8 original contiene agradecimientos: su práctica estadística es un añadido de esta guía.

## Fuentes editables y resultados

Los originales en Markdown con bloques ejecutables están en [lecciones/](../lecciones/README.md). Edita esos archivos para mantener cambios que quieras regenerar; puedes experimentar libremente en una copia del notebook.

Los presupuestos y semillas de estas lecciones se declaran en cada celda. Las cifras pueden diferir de los experimentos originales del proyecto, que conservan sus propias ejecuciones de referencia en resultados/capXX/. Las nuevas lecciones guardan sus salidas dentro del notebook.

Los ejemplos implementan mecanismos pequeños. Visualizar el objetivo PPO no equivale a entrenar PPO completo; el catálogo de respuestas del capítulo 6 no es un LLM. El texto explica esas limitaciones junto a cada práctica.
