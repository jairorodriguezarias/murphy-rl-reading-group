"""Cap. 2: Q-learning por muestras frente al óptimo de Bellman."""
import numpy as np
import matplotlib.pyplot as plt
from .comun import (cadena, iteracion_valor, evaluar_politica, guardar, curva,
                    greedy_aleatorio, filas_finales, resumen_muestra)

def ejecutar(out, seed=2026, replicas=20, pasos=16000):
    p, r = cadena()
    gamma = .95
    vstar, qstar = iteracion_valor(p, r, gamma)
    checkpoints = np.arange(0, pasos+1, 200)
    error = np.empty((replicas, len(checkpoints)))
    valor = np.empty_like(error)
    finales = []
    for k in range(replicas):
        rng = np.random.default_rng(seed+k)
        q = np.zeros_like(qstar)
        visits = np.zeros_like(q)
        s = int(rng.integers(1, 6))
        j = 0
        for t in range(pasos+1):
            if t % 200 == 0:
                error[k, j] = np.sqrt(np.mean((q[1:-1]-qstar[1:-1])**2))
                valor[k, j] = evaluar_politica(p, r, q.argmax(axis=1), gamma)[3]
                j += 1
            if t == pasos:
                break
            a = int(rng.integers(2)) if rng.random() < .25 else greedy_aleatorio(q[s], rng)
            ns = int(rng.choice(7, p=p[s, a]))
            done = ns in (0, 6)
            visits[s, a] += 1
            alpha = visits[s, a]**(-.6)
            target = r[s, a, ns] + gamma*(not done)*q[ns].max()
            q[s, a] += alpha*(target-q[s, a])
            s = int(rng.integers(1, 6)) if done else ns
        finales.append(q)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    curva(ax[0], checkpoints, error, "Q-learning")
    ax[0].set(xlabel="Transiciones reales", ylabel="RMSE frente a Q*", title="Error de estimación")
    curva(ax[1], checkpoints, valor, "Política greedy aprendida")
    ax[1].axhline(vstar[3], color="black", ls="--", label="Óptimo por programación dinámica")
    ax[1].set(xlabel="Transiciones reales", ylabel="V(s=3), evaluación exacta", title="Calidad de la política")
    ax[1].legend()
    metrics = {"gamma": gamma, "slip": .15, "epsilon": .25, "pasos": pasos,
               "valor_optimo_inicio": float(vstar[3]), "rmse_final": resumen_muestra(error[:, -1]),
               "valor_final": resumen_muestra(valor[:, -1])}
    return guardar(out, dict(pasos=checkpoints, rmse=error, valor=valor, qstar=qstar,
                            q_final=np.array(finales)), filas_finales("RMSE Q-learning", error[:, -1]),
                   metrics, fig)
