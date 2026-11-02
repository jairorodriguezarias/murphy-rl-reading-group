"""Cap. 6: respuestas categóricas, RLOO, KL y reward hacking; NO es un LLM."""
import numpy as np
import matplotlib.pyplot as plt
from .comun import softmax, guardar, curva, filas_finales, resumen_muestra

REFERENCIA = np.array([.5, .2, .1, .2])
REAL = np.array([1., 1., 0., 0.])
PROXY = np.array([1., 1.2, 1.8, -.2])

def objetivo(theta, reward, beta):
    p = softmax(theta)
    return float(p@reward - beta*np.sum(p*np.log(p/REFERENCIA)))

def gradiente_exacto(theta, reward, beta):
    p = softmax(theta)
    utility = reward - beta*(np.log(p/REFERENCIA)+1)
    return p*(utility-p@utility)

def entrenar(seed, beta, reward, pasos=800, batch=64):
    rng = np.random.default_rng(seed)
    theta = np.log(REFERENCIA).copy()
    history = []
    for t in range(pasos+1):
        p = softmax(theta)
        history.append([p@PROXY, p@REAL, np.sum(p*np.log(p/REFERENCIA))])
        if t == pasos:
            break
        a = rng.choice(4, batch, p=p)
        rewards = reward[a]
        # Cada baseline excluye la recompensa de la acción que multiplica.
        loo = (rewards.sum()-rewards)/(batch-1)
        signal = rewards-loo-beta*np.log(p[a]/REFERENCIA[a])
        grad = np.mean(signal[:, None]*(np.eye(4)[a]-p), axis=0)
        theta += .15*grad
    return np.array(history), softmax(theta)

def ejecutar(out, seed=2026, replicas=20):
    cases = [("Proxy β=0", 0., PROXY), ("Proxy β=0.1", .1, PROXY),
             ("Proxy β=0.5", .5, PROXY), ("Proxy β=2", 2., PROXY),
             ("Verificador β=0.1", .1, REAL)]
    arrays, rows, metrics = {}, [], {"iteraciones": 800, "batch": 64, "learning_rate": .15,
                                   "referencia": REFERENCIA.tolist(), "recompensa_real": REAL.tolist(),
                                   "recompensa_proxy": PROXY.tolist()}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, beta, reward in cases:
        data = [entrenar(seed+k, beta, reward) for k in range(replicas)]
        h = np.array([d[0] for d in data])
        final_p = np.array([d[1] for d in data])
        arrays[name], arrays[name+" probabilidades"] = h, final_p
        for j in range(3):
            curva(ax[j], np.arange(h.shape[1])*64, h[:, :, j], name)
        optimum = softmax(np.log(REFERENCIA)+reward/beta) if beta else np.eye(4)[reward.argmax()]
        metrics[name] = {"acierto_final": resumen_muestra(h[:, -1, 1]),
                         "proxy_final": resumen_muestra(h[:, -1, 0]),
                         "kl_final": resumen_muestra(h[:, -1, 2]),
                         "probabilidades_finales_medias": final_p.mean(axis=0).tolist(),
                         "optimo_analitico": optimum.tolist()}
        rows += filas_finales(name, h[:, -1, 1])
    for axis, ylabel in zip(ax, ["Recompensa proxy esperada", "Probabilidad de respuesta correcta", "KL(π || π referencia)"]):
        axis.set(xlabel="Respuestas muestreadas", ylabel=ylabel)
    ax[1].set_ylim(0, 1)
    ax[0].legend(fontsize=8)
    return guardar(out, arrays, rows, metrics, fig)
