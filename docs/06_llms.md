# Capítulo 6 · LLM y aprendizaje por refuerzo

**Lectura:** pp. 147–172; §§6.1–6.4. [Capítulo original](https://arxiv.org/html/2412.05265v5#Ch6).

**Idea central:** RL puede mejorar un modelo de lenguaje, y un modelo de lenguaje puede aportar representaciones, recompensas, modelos o políticas a un sistema de RL. Son dos direcciones diferentes.

## Resumen

§6.2 distingue preentrenamiento, ajuste supervisado y ajuste con recompensas. SFT aprende de respuestas demostradas; RLFT favorece respuestas que obtiene buenas puntuaciones al generarlas. Verificar a veces es más sencillo que producir una solución. RLVR usa señales verificables; RLHF aprende recompensas a partir de preferencias humanas; RLAIF usa evaluaciones de otros modelos. Las recompensas de resultado evalúan el final; las de proceso ofrecen información intermedia, pero pueden ser más difíciles de especificar o aprender.

El capítulo conecta la generación de tokens intermedios con cómputo adicional antes de responder. Discute qué capacidades aporta el modelo base y cuáles puede potenciar RL; no conviene interpretar automáticamente mejoras como aparición de capacidades desde cero. También separa un sistema de módulos con un objetivo común de un juego donde varios agentes persiguen objetivos propios.

En una interacción de un solo turno, el prompt puede ser contexto y la respuesta completa una acción: un bandit contextual. Si desplegamos la generación token a token, el estado es el prefijo. PPO, GRPO, DAPO, GSPO, RLOO y REINFORCE++ difieren en cómo estiman ventajas, normalizan, recortan razones de probabilidad o agregan pérdidas. GRPO estima una referencia con respuestas del mismo prompt; RLOO excluye la recompensa de la propia muestra al calcular su baseline. Esto permite prescindir de un crítico, pero requiere suficientes muestras y puede aumentar el coste.

La regularización KL intenta evitar cambios excesivos respecto al modelo de referencia. DPO conecta la solución del problema regularizado con un modelo de preferencias y optimiza pares preferido/rechazado directamente. Es una formulación distinta de un bucle online de política y recompensa. El muestreo guiado en inferencia y el ajuste con RL también pueden entenderse como maneras de aproximar una distribución favorecida por la recompensa, pagando el cómputo en momentos diferentes.

En problemas de varios turnos, acciones como herramientas y respuestas afectan a un entorno externo. Surgen consecuencias retrasadas, estados ocultos, acciones inválidas y contextos crecientes. El retorno y el crédito deben asignarse a través de la interacción. RAGEN, entrenamiento por turnos y métodos con críticos abordan partes del problema. La discusión de alineamiento y juegos de asistencia destaca que la utilidad humana puede ser incierta; optimizar un proxy fijo no elimina esa incertidumbre.

§6.3 invierte la dirección: usa LLM para interpretar observaciones, proponer recompensas, generar modelos del mundo, producir acciones o escribir código que implemente modelos y políticas. Generar un programa reutilizable puede ser mucho más rápido que llamar al modelo en cada paso, pero su corrección debe comprobarse. La latencia y el tamaño del contexto importan; se mencionan alternativas recurrentes y arquitecturas híbridas.

§6.4 muestra un ciclo de entrenamiento con Tinker y GEM: recoger episodios, calcular retornos y ventajas, preparar pérdidas por token, actualizar y registrar estabilidad. Los tokens del entorno no son decisiones del agente y se enmascaran. El ejemplo del libro usa servicios de entrenamiento; nuestro laboratorio es una reducción local y no reproduce esa infraestructura.

## Ecuaciones esenciales

Para una respuesta $y$ a un prompt $x$,

$$
J(\pi)=\mathbb{E}_{y\sim\pi(\cdot|x)}[r(x,y)]
-\beta D_{\mathrm{KL}}(\pi(\cdot|x)\|\pi_{\mathrm{ref}}(\cdot|x)).
$$

Si $\beta>0$, hay soporte apropiado y podemos representar cualquier distribución, la solución es

$$
\pi^*(y|x)=
\frac{\pi_{\mathrm{ref}}(y|x)\exp(r(x,y)/\beta)}
{\sum_{y'}\pi_{\mathrm{ref}}(y'|x)\exp(r(x,y')/\beta)}.
$$

Para un grupo de $K>1$ respuestas independientes, RLOO usa $b_i=\frac{1}{K-1}\sum_{j\ne i}r_j$. La recompensa de la propia muestra no entra en su baseline.

## Experimento 6 · Optimizar el proxy equivocado

**Pregunta:** ¿puede mejorar la recompensa mientras empeora la respuesta real?

Un único prompt tiene cuatro respuestas posibles:

| Respuesta | Probabilidad inicial | Recompensa real | Recompensa proxy |
|---|---:|---:|---:|
| Correcta breve | 0.5 | 1 | 1 |
| Correcta larga | 0.2 | 1 | 1.2 |
| Incorrecta convincente | 0.1 | 0 | 1.8 |
| Irrelevante | 0.2 | 0 | −0.2 |

Entrenamos logits categóricos con RLOO y penalización KL. Comparamos cuatro coeficientes sobre el proxy, $\beta=0,0.1,0.5,2$, y un control que optimiza la recompensa real con $\beta=0.1$.

~~~bash
uv run python -m experimentos.ejecutar --capitulo 6
~~~

**Métricas:** recompensa proxy esperada, probabilidad de corrección y KL respecto a la referencia. Hay 800 actualizaciones de 64 muestras y 20 semillas. Guardamos también el óptimo analítico: el entrenamiento con presupuesto finito no tiene por qué alcanzarlo.

**Resultado observado:** sin KL, la corrección cae desde 70% hasta aproximadamente 0.78%, aunque el proxy mejora. Una KL fuerte reduce el desplazamiento, pero no corrige la preferencia equivocada del proxy. Optimizar el verificador lleva la corrección a aproximadamente 98.7%.

**Modificación:** intercambia la puntuación proxy de una respuesta correcta y la incorrecta, o cambia las probabilidades iniciales. Observa cómo cambian la solución analítica y la velocidad de aprendizaje.

**Límite:** es una distribución sobre cuatro respuestas fijas; no hay transformer, tokens, generalización entre prompts ni entrenamiento de un reward model. Ilustra RLOO, KL y mala especificación, no mide el comportamiento de un LLM real.

![Resultados del capítulo 6](../resultados/cap06/figura.png)

## Para el grupo

1. ¿Qué diferencia hay entre un verificador correcto y uno que solo cubre algunos casos?
2. ¿Por qué una penalización KL no garantiza alineamiento?
3. ¿Cuándo deja de ser suficiente el modelo de bandit de un solo turno?

**Conexión:** la cobertura de datos, la exploración y el cambio de distribución llevan al capítulo 7.
