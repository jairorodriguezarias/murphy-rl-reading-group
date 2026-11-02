# Capítulo 5 · Aprendizaje por refuerzo multiagente

**Lectura:** pp. 121–146; §§5.1–5.3. [Capítulo original](https://arxiv.org/html/2412.05265v5#Ch5).

**Idea central:** cuando otros agentes también aprenden, “hacer lo mejor” depende de sus estrategias y del concepto de solución que se adopte.

## Resumen

§5.1 amplía el modelo del entorno. Los juegos de forma normal describen decisiones conjuntas; los juegos estocásticos incorporan estados y transiciones; los POSG incluyen observaciones parciales. Las recompensas pueden ser compartidas, de suma cero o generales. En juegos extensivos importan el orden de las decisiones y los conjuntos de información: dos historias pueden ser indistinguibles para un jugador.

Desde la perspectiva de un agente, cambiar la política de otro cambia el entorno efectivo, aunque las reglas permanezcan fijas. Esta no estacionariedad complica aplicar algoritmos diseñados para un único agente. La información pública y privada también condiciona qué puede aprenderse y qué puede utilizarse al actuar.

§5.2 distingue conceptos que no deben confundirse. Una mejor respuesta es óptima frente a rivales fijos. En suma cero de dos jugadores, minimax busca robustez frente al adversario. Un equilibrio de Nash no permite mejora unilateral, pero puede ser socialmente malo o no ser único. El equilibrio correlacionado permite recomendaciones dependientes de una señal común. Pareto, bienestar y equidad expresan criterios diferentes.

La explotabilidad y NashConv miden vulnerabilidad frente a mejores respuestas. Una recompensa media de cero no basta para certificar equilibrio: una estrategia puede empatar contra un rival particular y ser muy explotable por otro. El arrepentimiento compara resultados con acciones o políticas alternativas. Los valores de Shapley estudian contribuciones individuales a resultados compartidos. Stackelberg modela compromiso y respuesta entre líder y seguidor.

§5.3 recorre estrategias de aprendizaje. Centralizar todas las decisiones es sencillo conceptualmente, pero puede ser costoso o incompatible con información local. Aprender de forma independiente es práctico, aunque puede oscilar. CTDE permite críticos con información global durante entrenamiento y políticas locales durante ejecución. VDN y QMIX descomponen valores conjuntos bajo restricciones que facilitan escoger acciones por separado.

El self-play aprovecha la simetría, pero no ofrece convergencia universal. Fictitious play responde a la estrategia media del rival; las poblaciones y PSRO amplían el conjunto de oponentes para mejorar robustez. CFR minimiza arrepentimiento contrafactual en conjuntos de información; sus garantías clásicas se refieren a políticas medias y juegos bajo supuestos concretos, como dos jugadores de suma cero y recuerdo perfecto.

El capítulo cierra con métodos regularizados como MMD, comparaciones con PPO, planificación al decidir y aplicaciones a agentes lingüísticos. Es crucial separar resultados demostrados en juegos tabulares de observaciones empíricas con aproximación neuronal.

## Ecuaciones esenciales

Un equilibrio de Nash satisface, para cada jugador $i$,

$$
J_i(\pi_i,\pi_{-i})\geq J_i(\pi'_i,\pi_{-i})
\quad\text{para cualquier }\pi'_i.
$$

En un juego matricial de suma cero con matriz de pagos $A$, el laboratorio usa

$$
\operatorname{NashConv}(p,q)=\max_i(Aq)_i-\min_j(p^\top A)_j.
$$

Es la suma de las dos mejoras unilaterales posibles. Algunas convenciones de explotabilidad dividen esta cantidad entre dos; aquí no lo hacemos.

## Experimento 5 · Piedra, papel y tijera

**Pregunta:** ¿la estrategia actual y la estrategia media se comportan igual durante el aprendizaje?

Dos jugadores usan regret matching durante 4.000 iteraciones. Mantienen el arrepentimiento acumulado de cada acción y asignan probabilidad proporcional a su parte positiva. Se inicializan con distribuciones aleatorias para evitar el caso trivial de empezar exactamente en el equilibrio uniforme.

Calculamos los pagos esperados de todas las acciones frente al rival. Es un experimento de información completa, no un aprendizaje basado únicamente en el premio de una acción muestreada.

~~~bash
uv run python -m experimentos.ejecutar --capitulo 5
~~~

**Métricas:** NashConv de las políticas actuales y de las medias temporales; distribución media sobre las tres acciones. Las 20 réplicas corresponden a inicializaciones diferentes.

**Resultado observado:** la política media se aproxima a $(1/3,1/3,1/3)$ y su NashConv final ronda 0.031. La de las políticas actuales ronda 1.67: no se estabilizan del mismo modo. La media entre semillas que se dibuja en una curva no sustituye a la media temporal que usa el algoritmo.

**Modificación:** enfrenta la política aprendida a un rival que siempre juega piedra. Una mejor respuesta explotará su debilidad, mientras la estrategia de equilibrio prioriza robustez ante cualquier rival.

**Límite:** implementamos regret matching en un juego matricial, un caso elemental relacionado con CFR. No implementamos CFR para póquer ni demostramos resultados para cooperación o suma general.

![Resultados del capítulo 5](../resultados/cap05/figura.png)

## Para el grupo

1. ¿Por qué “nadie quiere desviarse” no equivale a “todos están satisfechos”?
2. ¿Qué información puede usar un crítico centralizado que no debería recibir el actor al ejecutar?
3. ¿Qué conclusiones cambiarían si solo observáramos pagos de acciones elegidas?

**Conexión:** no estacionariedad, preferencias y self-play reaparecen al entrenar sistemas con LLM.
