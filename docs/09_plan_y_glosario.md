# Plan de lectura y glosario

Este calendario es una propuesta adaptable. Cada sesión puede durar 75–90 minutos: 10 de repaso, 25 de explicación, 25 de experimento y 15–30 de discusión. Los capítulos largos merecen más de una sesión.

## Propuesta de 14 sesiones

| Sesión | Lectura | Preparación concreta |
|---|---|---|
| 0 | Conocimientos previos | Resolver el autodiagnóstico y abrir el cuaderno 1 |
| 1 | Cap. 1 | Describir un problema propio como agente, observación, acción y recompensa |
| 2 | §§2.1–2.3 | Derivar Bellman para una cadena de tres estados |
| 3 | §§2.4–2.5 | Ejecutar Q-learning y explicar SARSA, replay y red objetivo |
| 4 | §§3.1–3.2 | Explicar la identidad del gradiente y ejecutar el baseline |
| 5 | §§3.3–3.6 | Comparar PPO, TRPO y SAC por objetivo, datos y restricciones |
| 6 | §§4.1–4.3 | Ejecutar Dyna y contar datos reales y actualizaciones |
| 7 | §§4.4–4.5 | Comparar reconstrucción, valor y autopredicción; discutir transferencia |
| 8 | Cap. 5 | Ejecutar el juego y diferenciar media temporal de última política |
| 9 | §§6.1–6.2 | Ejecutar el proxy; distinguir recompensa, preferencias y KL |
| 10 | §§6.3–6.4 | Dibujar un ciclo de LLM, herramienta, observación, recompensa y actualización |
| 11 | §§7.1–7.4 | Comparar regret fijo y dinámico; ejecutar los bandits |
| 12 | §§7.5–7.8 | Contrastar opciones, imitación, offline RL y límites computacionales |
| 13 | Cap. 8 y cierre | Ejecutar selección de semillas y presentar una variante propia |

Para una primera pasada corta, dedica una sesión a cada capítulo 1–7 y otra al cierre. Reserva las pruebas técnicas y las familias de algoritmos menos familiares para la segunda pasada.

## Ficha reutilizable para preparar una sesión

1. **Problema:** ¿qué asume este capítulo sobre entorno, acciones, observaciones y datos?
2. **Objeto aprendido:** ¿valor, política, modelo, recompensa o representación?
3. **Una ecuación:** explica qué representa cada término y de dónde se obtienen sus datos.
4. **Una comparación:** identifica qué cambió y qué presupuesto se mantuvo fijo.
5. **Una evidencia:** anota una métrica con variabilidad.
6. **Una limitación:** escribe qué no permite concluir el experimento.
7. **Una pregunta:** formula algo que otro miembro pueda discutir o comprobar.

## Glosario de consulta

| Término | Significado en esta guía |
|---|---|
| Política | Regla o distribución que elige acciones |
| Retorno | Suma de recompensas, posiblemente descontadas |
| Valor | Esperanza del retorno bajo una política |
| Ventaja | Diferencia entre el valor de una acción y el valor medio del estado |
| Bootstrap | Usar una estimación aprendida dentro del objetivo de otra actualización |
| On-policy | Aprender sobre la política que genera los datos usados |
| Off-policy | Aprender sobre una política distinta de la que generó los datos |
| Offline RL | Aprender de un dataset fijo sin nueva interacción |
| Replay buffer | Memoria de transiciones reutilizables |
| Red objetivo | Copia o promedio lento que define objetivos de aprendizaje |
| Actor / crítico | Política / estimador de valor |
| Modelo del mundo | Predictor de consecuencias de acciones |
| Planificación | Usar un modelo para comparar decisiones o mejorar una política |
| MCTS | Búsqueda de árbol guiada mediante simulaciones |
| MPC | Optimizar un plan, ejecutar su primera acción y replanificar |
| No estacionariedad | Cambios en la distribución o dinámica que enfrenta el aprendiz |
| Mejor respuesta | Estrategia óptima frente a oponentes fijos |
| NashConv | Suma de las mejoras unilaterales posibles |
| CTDE | Entrenamiento centralizado y ejecución descentralizada |
| RLHF / RLAIF | Aprendizaje con feedback humano / de otro modelo |
| RLVR | Aprendizaje con recompensas verificables |
| RLOO | REINFORCE con baseline que excluye la propia muestra |
| DPO | Optimización directa a partir de pares de preferencias |
| Regret | Pérdida frente a un comparador definido |
| Pseudorregret | Diferencia acumulada entre medias de recompensa |
| Opción | Habilidad con iniciación, política interna y terminación |
| Behavior cloning | Imitación por aprendizaje supervisado de acciones |
| IRL | Inferir recompensas a partir de comportamiento |
| CQL | Aprendizaje de Q con una penalización conservadora |
| AIXI | Definición teórica de un agente con mezcla universal y planificación |

## Pregunta transversal

En cada capítulo, identifica quién paga el coste de la incertidumbre: ¿se paga con más interacciones, más cómputo, supuestos más fuertes, datos expertos o restricciones sobre la política? Esa pregunta conecta casi todo el libro.

**Referencia:** organización pedagógica propia a partir del [índice de Murphy v5](https://arxiv.org/html/2412.05265v5).
