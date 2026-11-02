# Capítulo 2 · Métodos basados en valores

**Lectura:** pp. 31–50; §§2.1–2.5. [Capítulo original](https://arxiv.org/html/2412.05265v5#Ch2).

**Idea central:** podemos aprender cuánto conviene cada acción y elegir las de mayor valor, propagando información mediante las ecuaciones de Bellman.

## Resumen

En §2.1 se distinguen el valor de un estado, el valor de una acción y su ventaja respecto a la política. Las ecuaciones de Bellman descomponen un problema de largo plazo en recompensa inmediata y valor del siguiente estado. Evaluar una política fija usa un promedio sobre sus acciones; buscar la política óptima introduce un máximo.

Con modelo conocido, §2.2 presenta iteración de valor, iteración de políticas y RTDP. La primera aplica repetidamente el operador de optimalidad; la segunda alterna evaluación y mejora. RTDP concentra el trabajo en estados relevantes para el problema. En el caso finito descontado, la contracción de Bellman da una razón precisa para la convergencia. El espacio de estados puede seguir siendo demasiado grande para calcularlo todo.

Cuando solo tenemos muestras, §2.3 compara Monte Carlo y diferencias temporales. Monte Carlo espera al retorno observado; TD usa una estimación del futuro como parte de su objetivo, operación llamada bootstrap. TD puede actualizar antes y reducir varianza, pero incorpora errores de su estimación. Los retornos de varios pasos, TD($\lambda$) y las trazas de elegibilidad distribuyen crédito entre distintos horizontes.

SARSA (§2.4) actualiza usando la siguiente acción realmente elegida por la política de comportamiento. Q-learning (§2.5) usa el máximo estimado sobre acciones siguientes. Esta diferencia explica su carácter on-policy y off-policy, respectivamente: Q-learning puede aprender sobre una política greedy mientras recoge datos exploratorios.

La aproximación funcional permite reemplazar tablas enormes por redes. DQN combina aprendizaje de Q con replay y una red objetivo. Reutilizar experiencias mejora el aprovechamiento de datos; una red objetivo cambia más despacio que la red entrenada. Estas técnicas son útiles frente a la combinación problemática de aproximación funcional, bootstrap y aprendizaje off-policy, la llamada tríada mortal. Las garantías tabulares no se trasladan automáticamente al caso neuronal.

El máximo de estimaciones ruidosas introduce sobreestimación. Double Q separa selección y evaluación; Double DQN traslada la idea al caso neuronal. El capítulo también recorre replay priorizado, arquitecturas dueling, noisy nets, retornos de varios pasos, Rainbow y BBF. Son combinaciones de decisiones estadísticas y de ingeniería, no sustitutos del principio de Bellman. Finalmente, HER reutiliza trayectorias fallidas como ejemplos de metas que sí se alcanzaron, recalculando las recompensas correspondientes.

## Ecuaciones esenciales

$$
Q^\pi(s,a)=\mathbb{E}[r_t+\gamma V^\pi(s_{t+1})|s_t=s,a_t=a],
\qquad
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

Q-learning tabular:

$$
Q(s,a)\leftarrow Q(s,a)+\alpha
\left[r+\gamma(1-d)\max_b Q(s',b)-Q(s,a)\right].
$$

$d=1$ indica un terminal real. En SARSA el máximo se reemplaza por $Q(s',a')$, con $a'\sim\pi(\cdot|s')$. Un límite artificial de tiempo no equivale necesariamente a un terminal del problema: hay que definir qué retorno queremos estimar.

## Experimento 2 · Aprender Bellman mediante muestras

**Pregunta:** ¿puede Q-learning aproximarse al resultado de un planificador que conoce el modelo?

Usamos una cadena con siete estados: 0 y 6 son terminales, se empieza cada episodio en un estado interior aleatorio y se evalúa desde el estado 3. Hay acciones izquierda/derecha y una probabilidad de 0.15 de ejecutar la dirección contraria. Entrar en 6 da recompensa 1; las demás transiciones no terminales de origen dan −0.01. Después de terminar no se reciben más recompensas.

El planificador calcula $Q^*$ con el modelo exacto. El agente aprende únicamente de transiciones, con $\epsilon=0.25$, $\gamma=0.95$ y tasa por visita $N(s,a)^{-0.6}$. Se comparan 20 ejecuciones de 16.000 transiciones reales.

~~~bash
uv run python -m experimentos.ejecutar --capitulo 2
~~~

**Métricas:** RMSE entre Q y $Q^*$ sobre pares no terminales, y valor exacto de la política greedy aprendida desde el estado 3. El evaluador conoce el modelo; el aprendiz no lo consulta.

**Qué observar:** el error disminuye y la política puede llegar al valor óptimo antes de que Q sea muy preciso. Para elegir bien basta con ordenar correctamente las acciones relevantes. Los resultados guardados alcanzan el valor óptimo aproximado 0.8166, con error de Q pequeño pero no nulo.

**Modificación:** compara varias probabilidades de deslizamiento o reduce la exploración. Una única ejecución no permite concluir que un cambio de hiperparámetro sea mejor.

**Límite:** este ejemplo es tabular; no prueba estabilidad de DQN ni reproduce la tríada mortal. Tampoco justifica trasladar su tasa de aprendizaje sin cambios a una red.

![Resultados del capítulo 2](../resultados/cap02/figura.png)

## Para el grupo

1. ¿Por qué SARSA puede preferir una ruta segura mientras sigue explorando?
2. ¿Cuál es la diferencia entre error de Q y pérdida de rendimiento de la política?
3. ¿Por qué una red objetivo ayuda pero no ofrece por sí sola una garantía universal?

**Conexión:** el capítulo 3 usará estimaciones de valor para mejorar directamente una política parametrizada.
