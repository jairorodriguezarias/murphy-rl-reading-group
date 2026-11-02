"""Cap. 5: regret matching en piedra, papel y tijera; política actual y media."""
import numpy as np
import matplotlib.pyplot as plt
from .comun import guardar, curva, filas_finales, resumen_muestra

A = np.array([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]])

def estrategia(regrets):
    pos = np.maximum(regrets, 0)
    return pos/pos.sum() if pos.sum() > 0 else np.ones(3)/3

def nashconv(p, q):
    # Suma de incentivos a desviarse; no dividimos por 2.
    return float(np.max(A@q)-np.min(p@A))

def entrenar(seed, pasos=4000):
    rng = np.random.default_rng(seed)
    regrets1, regrets2 = np.zeros(3), np.zeros(3)
    p, q = rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3))
    sum_p, sum_q = np.zeros(3), np.zeros(3)
    actual, media = [], []
    for t in range(1, pasos+1):
        sum_p += p
        sum_q += q
        actual.append(nashconv(p, q))
        media.append(nashconv(sum_p/t, sum_q/t))
        u1, u2 = A@q, -A.T@p
        # Actualizaciones simultáneas a partir de las políticas previas.
        regrets1 += u1 - p@u1
        regrets2 += u2 - q@u2
        p, q = estrategia(regrets1), estrategia(regrets2)
    return np.array(actual), np.array(media), sum_p/pasos, sum_q/pasos

def ejecutar(out, seed=2026, replicas=20):
    data = [entrenar(seed+k) for k in range(replicas)]
    actual, media = np.array([d[0] for d in data]), np.array([d[1] for d in data])
    p = np.array([d[2] for d in data])
    q = np.array([d[3] for d in data])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(1, actual.shape[1]+1)
    curva(ax[0], x, actual, "Políticas actuales")
    curva(ax[0], x, media, "Políticas medias")
    ax[0].set(xlabel="Iteración", ylabel="NashConv (suma de mejoras unilaterales)", title="No confundir última política con media")
    ax[0].legend()
    ax[1].bar(np.arange(3)-.18, p.mean(axis=0), width=.36, label="Jugador 1")
    ax[1].bar(np.arange(3)+.18, q.mean(axis=0), width=.36, label="Jugador 2")
    ax[1].axhline(1/3, ls="--", color="black", label="Equilibrio")
    ax[1].set(xticks=np.arange(3), xticklabels=["Piedra", "Papel", "Tijera"],
              ylabel="Probabilidad media", ylim=(0, .5))
    ax[1].legend()
    metrics = {"iteraciones": 4000, "nashconv_actual_final": resumen_muestra(actual[:, -1]),
               "nashconv_media_final": resumen_muestra(media[:, -1]),
               "politica_media_jugador_1": p.mean(axis=0).tolist(),
               "politica_media_jugador_2": q.mean(axis=0).tolist()}
    return guardar(out, dict(actual=actual, media=media, p=p, q=q), filas_finales("NashConv media", media[:, -1]),
                   metrics, fig)
