# Capítulo 7 · Otros temas de RL

**Lectura:** pp. 173–198; §§7.1–7.8. [Capítulo original](https://arxiv.org/html/2412.05265v5#Ch7).

**Idea central:** el algoritmo depende también de cómo se obtiene información, de qué datos existen y de la escala temporal en la que se representa la tarea.

## Resumen por bloques

**§7.1 — Regret.** En lugar de mirar solo el rendimiento final, medimos lo perdido mientras aprendemos frente a un comparador. En un entorno estático puede ser la mejor política fija; en uno cambiante puede ser una secuencia de políticas óptimas. Las garantías de regret dinámico necesitan restricciones sobre el cambio del entorno. El criterio bayesiano promedia incertidumbre con un prior; el minimax protege frente a una clase de casos. No son objetivos intercambiables.

**§7.2 — Exploración.** Una formulación bayesiana aumenta el estado con creencias sobre el modelo; planificar en ese espacio incorpora el valor de aprender. Salvo casos especiales, es caro. Thompson sampling muestrea un modelo de su posterior y actúa como si fuera correcto; UCB usa una estimación optimista. PSRL extiende el muestreo posterior a MDP y mantiene una política coherente durante un episodio. Sus garantías dependen del modelo, del prior o de los intervalos utilizados.

**§7.3 — RL distribucional.** Aprender la distribución del retorno aporta más información que aprender solo su esperanza. Puede representarse mediante cuantiles o masas categóricas. No equivale a estimar incertidumbre epistémica sobre Q: una distribución amplia puede deberse al azar del entorno. Tampoco basta discretizar una media escalar para afirmar que se ha aprendido toda la distribución de retornos.

**§7.4 — Motivación intrínseca.** La curiosidad recompensa novedad o aprendizaje del mundo; otros métodos recompensan adquirir habilidades y alcanzar metas. Bonos de conteo, RND, ganancia de información, empowerment, currículos y Go-Explore atacan problemas distintos. El ejemplo de la “televisión ruidosa” muestra que error de predicción alto no siempre significa información útil: un agente puede quedarse mirando ruido irreducible.

**§7.5 — Jerarquía.** Una opción es una política temporalmente extendida, con conjunto de iniciación y regla de terminación. El nivel superior escoge habilidades; la ejecución puede durar varios pasos. Option-critic, MAXQ y otras variantes distribuyen decisiones y crédito. En jerarquías feudales, un gestor propone metas a un ejecutor. El descubrimiento de subtareas busca cuellos de botella o habilidades diversas. Una jerarquía impuesta puede acelerar aprendizaje y también restringir las soluciones posibles.

**§7.6 — Imitación.** Behavior cloning trata acciones expertas como etiquetas. Su problema secuencial es que pequeños errores pueden llevar a estados ausentes de las demostraciones. IRL busca una recompensa que explique al experto, pero generalmente no identifica una recompensa única. Los métodos de divergencia comparan distribuciones de ocupación estado-acción; GAIL y AIRL usan ideas adversarias.

**§7.7 — Offline RL.** Se aprende de un conjunto fijo sin poder probar nuevas acciones. La política puede elegir acciones sin soporte en los datos y explotar valores o modelos mal estimados. Restricciones de política, incertidumbre y penalizaciones conservadoras, como CQL, limitan este problema. Los modelos del mundo offline también requieren cautela ante predicciones fuera de distribución. Decision Transformer y modelos de trayectorias convierten partes del problema en modelado condicionado por retorno; pedir un retorno alto no garantiza que una acción lo cause en un mundo estocástico. Offline-to-online añade interacción posterior. DAgger corrige imitación pidiendo etiquetas expertas en estados visitados por el aprendiz: requiere interacción y acceso al experto.

**§7.8 — RL general.** AIXI combina una distribución sobre programas con planificación para definir un agente ideal. Es un referente teórico, no una receta ejecutable general: la mezcla universal no es computable en su forma ideal y la planificación es costosa. El cierre aborda agentes con capacidad y datos limitados.

## Ecuaciones esenciales

En nuestro bandit, con medias de recompensa $\mu_t(a)$,

$$
\mathcal{R}_T=\sum_{t=1}^T
\left[\max_a\mu_t(a)-\mu_t(a_t)\right].
$$

Es **pseudorregret**: usamos medias verdaderas para evaluar las acciones elegidas. El agente no conoce esas medias. En el escenario cambiante el comparador puede cambiar de brazo en cada instante.

Si una opción dura $\tau$ pasos, el objetivo temporal tiene la forma

$$
y=\sum_{k=0}^{\tau-1}\gamma^k r_{t+k}
+\gamma^\tau V(s_{t+\tau}).
$$

Usar solo $\gamma$ en el último término ignoraría su duración.

## Experimento 7 · Explorar y adaptarse

Comparamos $\epsilon$-greedy con media histórica, UCB1, Thompson Beta-Bernoulli y Thompson con olvido. Los cuatro brazos tienen probabilidades de éxito $(0.2,0.4,0.6,0.8)$. En el segundo escenario se invierte ese orden después de 2.000 de las 4.000 interacciones.

Cada método prueba los cuatro brazos al inicio. UCB1 usa $\hat\mu(a)+\sqrt{2\log(t)/N(a)}$. Thompson estándar empieza con Beta(1,1). En la variante con olvido, los pseudoconteos por encima del prior se multiplican por 0.98 antes de incorporar cada observación. Esta última es una heurística adaptativa, no el posterior exacto del modelo estacionario.

~~~bash
uv run python -m experimentos.ejecutar --capitulo 7
~~~

**Métrica:** pseudorregret acumulado, con 20 trayectorias independientes por método y escenario. El ruido de recompensas se comparte por réplica e instante entre métodos para facilitar la comparación; no se comparten las recompensas de acciones no elegidas.

**Resultado observado:** Thompson estándar funciona especialmente bien en el escenario fijo. Olvidar mejora su adaptación al cambio en este ejemplo, pero empeora mucho su rendimiento estacionario. UCB1 obtiene el menor regret cambiante entre las configuraciones probadas. El experimento no confirma un ganador universal ni que olvidar sea siempre mejor.

**Modificación:** cambia el instante de inversión, la separación entre brazos o el factor de olvido. Separa hiperparámetros de evaluación si quieres compararlos formalmente.

**Límite:** el laboratorio implementa §§7.1–7.2 y conecta con incertidumbre; los apartados de jerarquía, imitación y offline RL se resumen, pero no se reproducen aquí. Un bandit no prueba exploración profunda en MDP.

![Resultados del capítulo 7](../resultados/cap07/figura.png)

## Para el grupo

1. ¿Por qué una incertidumbre predictiva grande puede ser inútil para explorar?
2. ¿Qué no puedes aprender de un dataset que nunca probó cierta acción?
3. ¿Qué supuestos hacen útil una habilidad aprendida al cambiar de tarea?

**Conexión:** vuelve a la pregunta inicial: ¿qué sabe el agente y qué datos puede conseguir?
