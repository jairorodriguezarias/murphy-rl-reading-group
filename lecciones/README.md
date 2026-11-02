# Fuentes de las lecciones ejecutables

Cada archivo combina explicación en español y bloques de código. El generador convierte los bloques delimitados por **~~~python** en celdas ejecutables y conserva el resto como Markdown. El marcador **<!-- celda -->** permite dividir una explicación extensa sin añadir un bloque de código.

- [00 · Preparación y autoevaluación](00_preparacion.md)
- [1 · Introducción](01_introduccion.md)
- [2 · Valores](02_valores.md)
- [3 · Políticas](03_politicas.md)
- [4 · Modelos](04_modelos.md)
- [5 · Multiagente](05_multiagente.md)
- [6 · LLM](06_llms.md)
- [7 · Otros temas](07_otros_temas.md)
- [8 · Agradecimientos y cierre](08_agradecimientos.md)

Para regenerar solo los cuadernos, desde la raíz:

~~~bash
uv run python scripts/construir_cuadernos.py
uv run --extra notebooks python scripts/ejecutar_cuadernos.py
uv run --extra docs python scripts/validar_material.py
~~~

El primer comando reemplaza los notebooks y elimina sus salidas; guarda antes cualquier cambio manual que quieras conservar. El segundo ejecuta cada uno en una carpeta vacía y guarda resultados dentro de su archivo. El tercero verifica ejecución, enlaces, correspondencia con estas fuentes y los materiales de referencia.

Para regenerar y ejecutar únicamente el preparatorio sin reemplazar los otros cuadernos:

~~~bash
uv run python scripts/construir_cuadernos.py --capitulo 0
uv run --extra notebooks python scripts/ejecutar_cuadernos.py --capitulo 0
~~~

La opción --capitulo de estos dos scripts acepta 0–8. Los módulos de experimentos/ conservan sus capítulos 1–8.

La metadata de cada notebook conserva el SHA-256 de su lección y si se ejecutó en un directorio aislado. [Índice para lectores](../cuadernos/README.md).

Las explicaciones adaptan el libro de Kevin P. Murphy, [arXiv:2412.05265v5](https://arxiv.org/abs/2412.05265v5), bajo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Código y ejemplos propios, sin respaldo implícito del autor.
