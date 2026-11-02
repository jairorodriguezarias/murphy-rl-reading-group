# Protocolo de reproducción e interpretación

## Qué se ejecutó

La referencia usa semilla base 2026 y 20 réplicas. Los módulos 1–6 repiten con semillas 2026–2045; el módulo 7 genera 20 trayectorias por lote a partir de flujos pseudoaleatorios separados para decisiones y recompensas. El módulo 8 genera 2.000 estudios sintéticos, cada uno con 20 semillas de A y B.

Los presupuestos, tasas y demás hiperparámetros están en el código y en cada resumen.json. El manifiesto de resultados registra versiones, tiempos por capítulo y hashes SHA-256 del código usado. uv.lock fija la resolución del entorno; requirements.txt fija las dependencias de ejecución para la alternativa pip.

Las réplicas miden azar en observaciones, acciones o inicializaciones según el experimento. No representan una muestra de todos los entornos posibles.

## Archivos que puedes inspeccionar

| Archivo por capítulo | Contenido |
|---|---|
| figura.png | Gráfica para lectura y presentación |
| figura.svg | Figura vectorial para exportar |
| datos.npz | Arrays por réplica y punto temporal; no solo promedios |
| metricas.csv | Métricas finales por réplica o por estudio |
| resumen.json | Parámetros principales y agregados |

Los datos se abren con numpy.load("resultados/cap02/datos.npz"). Usa .files para ver las claves. Las trayectorias completas de estados y acciones no se guardan; sí las series de métricas necesarias para reconstruir las figuras.

## Bandas y evaluación

Las bandas representan media ± 1.96 veces el error estándar entre ejecuciones. Son aproximaciones normales, con solo 20 réplicas por defecto, y se calculan punto a punto. No son desviaciones estándar individuales ni intervalos simultáneos. Un IC puede salir del rango de una métrica acotada; no lo truncamos para aparentar mayor precisión.

No suavizamos las curvas. Los ejemplos pequeños permiten evaluación exacta en los capítulos 2, 3, 5 y 6. Dyna se evalúa mediante error respecto al óptimo conocido y metas conseguidas durante entrenamiento. El regret de bandits usa las medias verdaderas solo para evaluar.

Las configuraciones comparten la tarea y el presupuesto dentro de cada comparación. Igual número de interacciones no equivale a igual cómputo: Dyna registra ambos. Compartir semillas o ruido no implica que las trayectorias sean iguales cuando las políticas divergen.

## Validación numérica

Las pruebas contrastan la implementación con propiedades independientes: solución cerrada de una cadena determinista, residuos de Bellman, valores terminales, invariancia de softmax, gradientes por diferencias finitas, solución analítica regularizada, esperanza exacta de RLOO, equilibrio del juego, reproducibilidad y cotas del pseudorregret.

La validación documental comprueba presencia de capítulos, cuadernos, resultados, fórmulas renderizadas y destinos locales de enlaces. Ejecutar los cuadernos de principio a fin comprueba además imports y rutas.

## Cómo proponer una variante

Antes de ejecutar, escribe una hipótesis y cambia un solo factor: descuento, exploración, ruido, planificación, penalización KL o cambio del entorno. Elige un directorio nuevo con --salida. Mantén un conjunto de semillas para explorar hiperparámetros y otro para evaluarlos si pretendes hacer una comparación formal. Informa resultados desfavorables y diferencias en cómputo.

Los laboratorios son demostraciones didácticas. No justifican conclusiones sobre Atari, robótica o LLM reales sin experimentos adicionales.
