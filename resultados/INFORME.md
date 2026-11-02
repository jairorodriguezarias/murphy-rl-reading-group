# Resultados observados

Ejecución de referencia: semilla base 2026, 20 réplicas. Los valores con ± indican media y semianchura del IC normal aproximado del 95%. Las cifras se generan a partir de los JSON; no son resultados del libro.

| Cap. | Métrica | Resultado |
|---|---|---:|
| 1 | Acierto sin memoria | 0.4955 ± 0.0094 |
| 1 | Acierto con memoria | 1.0000 ± 0.0000 |
| 2 | RMSE final de Q | 0.0158 ± 0.0032 |
| 2 | Valor de política final | 0.8166 ± 0.0000 |
| 3 | Retorno sin baseline, restando 5 | 0.9680 ± 0.0018 |
| 3 | Retorno con baseline, restando 5 | 0.9673 ± 0.0006 |
| 4 | RMSE con 0 pasos de planificación | 0.0186 ± 0.0032 |
| 4 | RMSE con 5 pasos de planificación | 0.0000 ± 0.0000 |
| 5 | NashConv de políticas actuales | 1.6699 ± 0.1049 |
| 5 | NashConv de políticas medias | 0.0310 ± 0.0006 |
| 6 | Corrección, proxy sin KL | 0.0078 ± 0.0001 |
| 6 | Corrección, proxy con β=2 | 0.7358 ± 0.0002 |
| 6 | Corrección, verificador con β=0.1 | 0.9871 ± 0.0001 |

## Varianza del gradiente

Con política fija, la traza de covarianza baja de 17.067 a 0.392. La reducción de varianza no produce una mejora final clara en la recompensa de este ejemplo.

## Exploración y cambio del entorno

| Método | Regret estacionario | Regret cambiante |
|---|---:|---:|
| ε-greedy | 135.3900 ± 12.1126 | 1123.0300 ± 46.1016 |
| UCB1 | 98.5100 ± 5.4502 | 125.3400 ± 11.8200 |
| Thompson | 19.9600 ± 4.2416 | 491.0400 ± 66.8177 |
| Thompson con olvido | 298.5500 ± 8.5112 | 305.7400 ± 7.8340 |

Thompson estándar destaca en el entorno fijo. El olvido ayuda a Thompson en el cambiante, pero UCB1 logra aquí el menor regret cambiante. No hay un ganador universal.

## Sesgo de selección

Con algoritmos de igual calidad real, comparar medias produce 0.0014; seleccionar la mejor semilla de A produce 1.8640; evaluar de nuevo produce -0.0444. Son promedios de 2.000 estudios sintéticos.

## Tiempo y alcance

Los ocho módulos tardaron 13.28 segundos en esta máquina, sin incluir instalación, creación de cuadernos ni validación. El tiempo en otro equipo puede variar.

Dyna compara presupuestos iguales de interacción, con cómputo diferente. El laboratorio de LLM es categórico. Las curvas no establecen resultados generales ni reproducen benchmarks de redes profundas.

Consulta los JSON y arrays por capítulo para más métricas y precisión. No se ha elegido solo la mejor semilla.
