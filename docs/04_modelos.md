# Capítulo 4 · Métodos basados en modelos

**Lectura:** pp. 83–120; §§4.1–4.5. [Capítulo original](https://arxiv.org/html/2412.05265v5#Ch4).

**Idea central:** aprender cómo funciona el entorno permite convertir cómputo en experiencia imaginada; la utilidad de esa experiencia depende de la fidelidad del modelo.

## Resumen

§4.1 plantea el intercambio fundamental: una interacción real puede ser costosa, mientras que simularla puede ser barato. Aprender un modelo de transiciones y recompensas permite buscar acciones o entrenar políticas dentro del modelo. Sin embargo, una política puede descubrir y explotar errores de predicción. Por eso la recogida de datos, el ajuste del modelo y la mejora de la política suelen alternarse.

§4.2 estudia la planificación en el momento de decidir. La búsqueda hacia delante considera futuros posibles; branch and bound y métodos heurísticos reducen la exploración del árbol. MCTS asigna simulaciones de forma selectiva, combinando estimaciones de valor y exploración. AlphaZero usa reglas conocidas; MuZero aprende representaciones y una dinámica orientada a recompensas, valores y políticas.

El control predictivo, MPC, optimiza una secuencia de acciones, ejecuta la primera y vuelve a planificar tras observar el nuevo estado. La secuencia interna es de lazo abierto, aunque el controlador completo incorpora realimentación al replanificar. En entornos estocásticos, una política contingente puede superar a cualquier secuencia fija porque anticipa cómo aprovechar futuras observaciones. LQR, CEM, random shooting, MPPI y métodos con procesos gaussianos ofrecen distintas formas de resolver o aproximar la optimización.

§4.3 cambia el momento de usar el modelo: la planificación en segundo plano produce datos imaginados que mejoran una política reactiva. Dyna combina actualizaciones a partir de datos reales y simulados. La perspectiva de juegos y optimización a dos niveles ayuda a pensar cómo coordinar el modelo y la política. Aquí “planificación offline” significa fuera del instante de decisión; no implica necesariamente el conjunto fijo de datos de offline RL del capítulo 7.

§4.4 pregunta qué debería predecir el modelo. Reconstruir observaciones completas proporciona una señal rica, pero puede gastar capacidad en detalles irrelevantes. Dreamer aprende una representación dinámica y entrena en imaginación; IRIS usa representaciones discretas y modelos autorregresivos. Otras familias predicen recompensas, valores, políticas o representaciones futuras, buscando preservar información útil para decidir.

Aprender representaciones por autopredicción introduce el riesgo de colapso: mapear todas las observaciones al mismo vector hace la predicción fácil e inútil. Objetivos auxiliares, redes objetivo y regularizadores intentan evitarlo; sus garantías dependen del caso. JEPA, BYOL y TD-MPC muestran distintas combinaciones. El capítulo también aborda error acumulado, rollouts cortos, incertidumbre y la posibilidad de alternar métodos con y sin modelo.

§4.5 estudia predicciones a más de un paso. Las funciones de valor generalizadas predicen señales distintas de la recompensa. Las representaciones y características sucesoras acumulan ocupaciones o características futuras bajo una política. Así puede reevaluarse rápidamente una recompensa nueva, siempre respetando la dependencia respecto a la política y la representación. La mejora generalizada de políticas reutiliza una biblioteca de habilidades; las medidas sucesoras conectan predicción y muestreo de futuros.

## Ecuación esencial

Si la recompensa es lineal en características de transición,

$$
r(s,a,s')=\phi(s,a,s')^\top w,
\qquad
\psi^\pi(s,a)=
\mathbb{E}_\pi\left[\sum_{t\geq0}\gamma^t\phi(s_t,a_t,s_{t+1})\mid s_0=s,a_0=a\right],
$$

entonces

$$
Q^\pi_w(s,a)=\psi^\pi(s,a)^\top w.
$$

Cambiar $w$ permite evaluar otra recompensa sin reaprender $\psi^\pi$. Si cambia la dinámica o queremos una política distinta, la reutilización necesita trabajo adicional.

## Experimento 4 · Dyna-Q

**Pregunta:** ¿cuánto se gana en aprendizaje por interacción real cuando reutilizamos un modelo aprendido?

Una cadena determinista tiene nueve estados; los extremos son terminales. El agente empieza en 1 y recibe 1 al llegar a 8. Las demás recompensas son cero. Se comparan 0, 5 y 20 actualizaciones imaginadas por transición real, con el mismo presupuesto de 1.600 transiciones y 20 semillas.

Cada transición observada actualiza Q y una tabla del modelo. La planificación muestrea solo pares estado-acción ya observados y aplica otra actualización de Bellman. No se entrega el modelo real al aprendiz. En este entorno determinista, almacenar una transición basta para conocer ese par.

~~~bash
uv run python -m experimentos.ejecutar --capitulo 4
~~~

**Métricas:** error frente a $Q^*$ y metas alcanzadas durante el entrenamiento. También se registran las actualizaciones totales: 1.600, 9.600 y 33.600.

**Qué observar:** la planificación propaga antes la información de la meta y reduce el error con menos datos reales. En los resultados guardados, 5 y 20 pasos alcanzan prácticamente el mismo error final; veinte pasos no aportan una mejora proporcional a su coste.

**Modificación:** compara a igual número total de actualizaciones, no solo a igual cantidad de datos. Como extensión de implementación, añade dinámica estocástica: una tabla que recuerde únicamente la última transición ya no será un modelo fiel de su distribución.

**Límite:** aquí el modelo observado es exacto y no existe error acumulado de una red. El experimento no demuestra que planificar más siempre sea mejor ni reproduce Dreamer o MuZero.

![Resultados del capítulo 4](../resultados/cap04/figura.png)

## Para el grupo

1. ¿Cuándo vale más una transición real que miles de transiciones imaginadas?
2. ¿Por qué predecir bien los píxeles puede ser insuficiente para actuar bien?
3. ¿Qué diferencia hay entre cambiar la recompensa y cambiar la dinámica al reutilizar características sucesoras?

**Conexión:** aprender un modelo y una política a la vez anticipa los problemas de interacción y no estacionariedad del capítulo 5.
