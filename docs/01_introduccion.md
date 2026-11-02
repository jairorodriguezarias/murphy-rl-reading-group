# Capítulo 1 · Introducción

**Lectura:** pp. 13–30 del PDF; §§1.1–1.3. [Capítulo original](https://arxiv.org/html/2412.05265v5#Ch1).

**Idea central:** antes de elegir un algoritmo, hay que especificar qué observa el agente, qué puede hacer, qué intenta optimizar y qué desconoce.

## Resumen

En §1.1 Murphy presenta RL como parte de la toma de decisiones secuencial. El agente recibe observaciones, mantiene un estado interno y elige acciones mediante una política. Sus decisiones afectan a lo que verá y podrá hacer después. La información disponible, el estado real del mundo y los parámetros del agente son objetos distintos: aprender una política y actualizar una creencia no son la misma operación.

El objetivo habitual es maximizar retorno esperado. En tareas episódicas la interacción termina; en tareas continuas puede usarse un objetivo descontado o de recompensa media. El descuento controla cuánto pesan las consecuencias lejanas y, con recompensas acotadas y $\gamma<1$, hace finita la suma infinita. Cambiarlo puede cambiar la política óptima.

En §1.2 se ordenan varios modelos. Un MDP supone que el estado relevante es observado. Un POMDP permite observaciones parciales y necesita memoria o creencias sobre el estado oculto. Un estado de creencia resume probabilísticamente el historial cuando el modelo es conocido. También podemos mantener creencias sobre parámetros desconocidos: esto conecta la decisión con el aprendizaje del entorno.

Un bandit contextual tiene una decisión por contexto sin que la acción determine los contextos futuros; aun así, la acción determina qué recompensa observamos y qué aprendemos. Los MDP condicionados por metas permiten modificar el objetivo; los MDP contextuales permiten variar la tarea o dinámica entre episodios. La identificación del mejor brazo, la optimización bayesiana, el aprendizaje activo y hasta la elección de pasos de SGD se presentan como problemas relacionados de decisión e información.

En §1.3 aparece el mapa del resto del libro: aprender valores, aprender directamente políticas o aprender modelos para planificar. Esta clasificación es independiente de usar tablas o redes neuronales, y de aprender on-policy u off-policy. La observación parcial exige memoria; el desconocimiento del modelo exige exploración. Son problemas diferentes.

La recompensa tampoco es un detalle secundario. Una señal escasa dificulta aprender; una señal incorrecta puede producir un comportamiento eficaz para la puntuación e indeseable para la tarea. El shaping añade información, pero puede modificar el objetivo. El shaping basado en potencial tiene garantías bajo condiciones apropiadas. El capítulo termina con recomendaciones sobre variabilidad, agregación y evaluación experimental.

## Ecuaciones que debes poder explicar

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k},
\qquad
J(\pi)=\mathbb{E}_\pi[G_0].
$$

$G_t$ corresponde a una trayectoria; $J$ promedia sobre las trayectorias que producen política y entorno.

Para shaping basado en potencial:

$$
r'_t=r_t+\gamma\Phi(s_{t+1})-\Phi(s_t).
$$

La suma adicional se telescopa. Con potencial acotado en horizonte infinito descontado, o potencial terminal cero en episodios, añade un término independiente de las acciones desde un estado inicial fijo. Sin cuidar los estados terminales esa conclusión puede fallar.

## Experimento 1 · Futuro y memoria

**Pregunta:** ¿cambia la mejor decisión cuando modificamos el descuento o la información disponible?

**Hipótesis:** recibir 3 en $t=4$ supera recibir 1 ahora únicamente si $3\gamma^4>1$. Por otra parte, recordar una señal inicial permite resolver una tarea cuya observación final es ambigua.

El primer panel compara esos dos retornos para 201 valores de $\gamma$. El segundo genera un bit oculto equiprobable, lo revela al inicio y lo oculta antes de decidir. Compara una decisión aleatoria sin memoria con una política que recuerda la señal. Son políticas prescritas: este laboratorio ilustra la formulación, no entrena un agente.

~~~bash
uv run python -m experimentos.ejecutar --capitulo 1
~~~

**Qué medir:** punto de cruce de retornos y proporción de aciertos. Se realizan 20 réplicas de 1.000 episodios para la tarea de memoria.

**Qué observar:** el cruce está en $\gamma=(1/3)^{1/4}\approx0.760$; la política sin memoria ronda 50% y la que recuerda alcanza 100% en este entorno sin ruido. La igualdad en el umbral significa empate.

**Modificación:** cambia el retraso de cuatro a ocho pasos. Después introduce ruido en la señal inicial y comprueba que recordar ya no basta para acertar siempre.

**Límite:** no demuestra que una red recurrente vaya a aprender esa memoria. Solo demuestra que la información necesaria no está en la última observación.

![Resultados del capítulo 1](../resultados/cap01/figura.png)

## Para el grupo

1. ¿Qué habría que incluir en el estado de un robot para que la representación sea Markoviana?
2. ¿Puede un bandit necesitar exploración aunque sus acciones no cambien el mundo?
3. ¿Cómo distinguirías un fallo de aprendizaje de un fallo de recompensa?

**Conexión:** el capítulo 2 aprende a valorar consecuencias; el 3 aprende a actuar; el 4 aprende a predecirlas.
