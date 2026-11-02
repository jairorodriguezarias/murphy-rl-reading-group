"""Cap. 7: exploración en bandits Bernoulli estacionarios y cambiantes."""
import numpy as np
import matplotlib.pyplot as plt
from .comun import guardar, curva, filas_finales, resumen_muestra

METODOS = ["ε-greedy", "UCB1", "Thompson", "Thompson con olvido"]

def simular(method, cambiante, seed, replicas, pasos=4000):
    rng = np.random.default_rng(seed)
    n = np.zeros((replicas, 4))
    sums = np.zeros_like(n)
    alpha, beta = np.ones_like(n), np.ones_like(n)
    regrets = np.zeros((replicas, pasos))
    idx = np.arange(replicas)
    # Ruido de recompensa separado del RNG de decisiones.
    reward_rng = np.random.default_rng(seed+1000000)
    for t in range(pasos):
        mu = np.array([.2, .4, .6, .8])
        if cambiante and t >= pasos//2:
            mu = mu[::-1]
        mean = np.divide(sums, n, out=np.zeros_like(sums), where=n > 0)
        if t < 4:
            a = np.full(replicas, t)
        elif method == "ε-greedy":
            a = (mean + rng.uniform(0, 1e-10, mean.shape)).argmax(axis=1)
            explore = rng.random(replicas) < .1
            a[explore] = rng.integers(0, 4, explore.sum())
        elif method == "UCB1":
            a = (mean + np.sqrt(2*np.log(t+1)/n)).argmax(axis=1)
        else:
            a = rng.beta(alpha, beta).argmax(axis=1)
        reward = (reward_rng.random(replicas) < mu[a]).astype(float)
        n[idx, a] += 1
        sums[idx, a] += reward
        if method == "Thompson con olvido":
            alpha = 1+.98*(alpha-1)
            beta = 1+.98*(beta-1)
        alpha[idx, a] += reward
        beta[idx, a] += 1-reward
        regrets[:, t] = mu.max()-mu[a]
    return regrets.cumsum(axis=1)

def ejecutar(out, seed=2026, replicas=20):
    arrays, rows, metrics = {}, [], {"pasos": 4000, "cambio_en": 2000,
                                   "epsilon": .1, "factor_olvido": .98}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for col, cambiante in enumerate([False, True]):
        scenario = "Cambiante" if cambiante else "Estacionario"
        metrics[scenario] = {}
        for method in METODOS:
            regret = simular(method, cambiante, seed, replicas)
            name = scenario+" "+method
            arrays[name] = regret
            rows += filas_finales(name, regret[:, -1])
            curva(ax[col], np.arange(1, 4001), regret, method)
            metrics[scenario][method] = resumen_muestra(regret[:, -1])
        ax[col].set(xlabel="Interacción", ylabel="Pseudorregret acumulado", title=scenario)
        ax[col].legend()
        if cambiante:
            ax[col].axvline(2000, ls=":", color="black")
    return guardar(out, arrays, rows, metrics, fig)
