# Capítulo 3 · Métodos basados en políticas

**Lectura:** pp. 51–82; §§3.1–3.6. [Capítulo original](https://arxiv.org/html/2412.05265v5#Ch3).

**Idea central:** ajustar directamente la distribución de acciones permite optimizar políticas estocásticas y manejar acciones continuas, pero exige controlar la varianza y el tamaño de las actualizaciones.

## Resumen

En §3.1 se deriva el gradiente mediante la identidad de la derivada logarítmica. No hace falta diferenciar la dinámica: basta muestrear trayectorias y conocer la probabilidad de las acciones elegidas. REINFORCE aumenta la probabilidad de acciones asociadas a mejores retornos. Usar recompensa futura en vez del retorno completo evita atribuir a una acción recompensas que ocurrieron antes.

Restar un baseline que no dependa de la acción, tratado como constante en el estimador, conserva la esperanza del gradiente. Un baseline adecuado puede reducir mucho su varianza. El valor del estado es una elección habitual; no cualquier baseline reduce la varianza.

En §3.2, actor-critic combina una política con un estimador de valor. El crítico permite usar bootstrap para actualizar sin esperar al episodio completo. A2C usa un error TD como estimación de ventaja. GAE mezcla errores TD a distintos horizontes mediante $\lambda$, intercambiando sesgo y varianza. La calidad del crítico y la coordinación de las dos tasas de aprendizaje importan.

El gradiente natural tiene en cuenta cómo los cambios de parámetros modifican distribuciones, usando la geometría de Fisher. Para acciones continuas, DDPG entrena un actor determinista guiado por un crítico. TD3 combate errores y sobreestimación mediante dos críticos, actualizaciones retardadas y suavizado del objetivo. WPO explora otra geometría para mover distribuciones de acciones.

En §3.3, TRPO y PPO buscan limitar cambios destructivos. TRPO aproxima una actualización restringida por KL. PPO usa un objetivo recortado más sencillo y permite varias actualizaciones sobre un lote. El clipping limita incentivos en ese objetivo; no impone por sí solo una cota dura a toda la divergencia ni garantiza mejora en cada ejecución. VMPO ofrece una alternativa basada en distribución auxiliar y proyección.

§3.4 explica cómo reutilizar datos de otra política. El muestreo por importancia corrige el cambio de distribución cuando hay soporte suficiente, pero sus productos pueden tener mucha varianza. V-trace y Retrace introducen correcciones y truncamientos. IMPALA combina estos principios con recogida distribuida. Off-policy no significa automáticamente que un conjunto fijo y arbitrario de datos sea suficiente.

§3.5 recuerda que también existen búsqueda aleatoria, estrategias evolutivas y otros métodos sin gradientes. Finalmente, §3.6 interpreta control como inferencia: favorecer trayectorias de alta recompensa conduce a objetivos con entropía o KL. MPO, AWR/AWAC y SAC se entienden desde esa perspectiva. SAC aprende una política estocástica con un objetivo de recompensa más entropía. La discusión de inferencia activa conecta percepción, preferencias y búsqueda de información.

## Ecuaciones esenciales

Para episodios sin descuento, una forma de REINFORCE es

$$
\nabla_\theta J=
\mathbb{E}\left[\sum_t
\nabla_\theta\log\pi_\theta(a_t|s_t)(G_t-b(s_t))\right].
$$

Si $J=\mathbb{E}[\sum_t\gamma^t r_t]$ y $G_t$ descuenta desde $t$, la forma por trayectoria incluye además un factor $\gamma^t$. Otras presentaciones lo incorporan a la distribución de visitas. En el laboratorio hay un solo paso y no surge esta distinción.

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),
\qquad
\hat A_t^{\mathrm{GAE}}=\sum_{l\geq0}(\gamma\lambda)^l\delta_{t+l}.
$$

La suma se trunca y los valores terminales se tratan adecuadamente en episodios finitos.

## Experimento 3 · ¿Qué aporta un baseline?

Un bandit contextual tiene dos contextos equiprobables y tres acciones. Las medias, sin un desplazamiento común, son $(1,0,-1)$ y $(-1,0,1)$. Añadimos 5 a todas las recompensas y ruido normal con desviación 0.5.

Se entrenan dos políticas softmax con REINFORCE: una sin baseline y otra con una media móvil por contexto, calculada con lotes anteriores. Usamos 500 actualizaciones, lotes de 32 y 20 semillas.

~~~bash
uv run python -m experimentos.ejecutar --capitulo 3
~~~

**Control importante:** además del aprendizaje, medimos ambos estimadores bajo una misma política uniforme fija, con 50.000 muestras y baseline exacto 5. Comparar varianzas de políticas distintas no aislaría el efecto del baseline.

**Métricas:** recompensa esperada exacta menos el desplazamiento 5; y traza de la covarianza del gradiente en el control fijo.

**Resultado observado:** la varianza cae aproximadamente de 17.07 a 0.392. Ambas políticas alcanzan recompensas finales parecidas, alrededor de 0.967 frente al óptimo 1. Reducir varianza no implica necesariamente obtener una media final superior con este presupuesto.

**Modificación:** elimina el desplazamiento 5 y repite. Luego usa deliberadamente un baseline muy inadecuado. Razona por qué conservar la esperanza no garantiza reducir la varianza.

**Límite:** ilustra REINFORCE y baselines; no implementa PPO, GAE ni un crítico neuronal.

![Resultados del capítulo 3](../resultados/cap03/figura.png)

## Para el grupo

1. ¿Por qué un baseline dependiente de la acción necesita tratamiento adicional?
2. ¿Qué diferencia hay entre menos varianza y menos sesgo?
3. ¿Qué permite afirmar el clipping de PPO y qué habría que medir aparte?

**Conexión:** las ideas de gradiente, ventaja y KL reaparecen en el ajuste de LLM del capítulo 6.
