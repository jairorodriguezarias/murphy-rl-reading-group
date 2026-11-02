"""Cap. 3: REINFORCE y reducción de varianza mediante un baseline."""
import numpy as np
import matplotlib.pyplot as plt
from .comun import softmax, curva, guardar, filas_finales, resumen_muestra

MEDIAS = np.array([[1., 0., -1.], [-1., 0., 1.]])

def entrenar(seed, baseline, pasos=500, batch=32):
    rng = np.random.default_rng(seed)
    theta = np.zeros((2, 3))
    b = np.zeros(2)
    valores = [0.]
    for _ in range(pasos):
        s = rng.integers(0, 2, batch)
        probs = softmax(theta)
        a = (rng.random(batch)[:, None] > np.cumsum(probs[s], axis=1)).sum(axis=1)
        rewards = 5 + MEDIAS[s, a] + rng.normal(0, .5, batch)
        advantage = rewards - b[s] if baseline else rewards
        score = np.eye(3)[a] - probs[s]
        grad = np.zeros_like(theta)
        np.add.at(grad, s, advantage[:, None]*score / batch)
        theta += .1*grad
        # El baseline de ESTE lote se calculó antes de observar sus recompensas.
        for context in range(2):
            if np.any(s == context):
                b[context] = .9*b[context]+.1*rewards[s == context].mean()
        valores.append(float((softmax(theta)*MEDIAS).sum()/2))
    return np.array(valores)

def ejecutar(out, seed=2026, replicas=20):
    runs = {label: np.array([entrenar(seed+k, use) for k in range(replicas)])
            for label, use in [("Sin baseline", False), ("Baseline aprendido", True)]}
    # Medimos la varianza con la misma política fija, no durante políticas distintas.
    rng = np.random.default_rng(seed+100000)
    a = rng.integers(0, 3, 50000)
    r = 5 + MEDIAS[0, a] + rng.normal(0, .5, len(a))
    score = np.eye(3)[a] - 1/3
    g0, gb = r[:, None]*score, (r-5)[:, None]*score
    variances = np.array([g0.var(axis=0, ddof=1).sum(), gb.var(axis=0, ddof=1).sum()])
    exact = MEDIAS[0]/3
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, values in runs.items():
        curva(ax[0], np.arange(values.shape[1])*32, values, name)
    ax[0].axhline(1, color="black", ls="--", label="Óptimo")
    ax[0].set(xlabel="Interacciones", ylabel="Recompensa esperada menos 5", title="Aprendizaje de la política")
    ax[0].legend()
    ax[1].bar(["Sin baseline", "Baseline exacto 5"], variances)
    ax[1].set(ylabel="Traza de covarianza del gradiente", title="Varianza con política uniforme fija")
    metrics = {"batch": 32, "iteraciones": 500, "learning_rate": .1,
               "final": {n: resumen_muestra(v[:, -1]) for n, v in runs.items()},
               "varianza_sin_baseline": float(variances[0]), "varianza_con_baseline": float(variances[1]),
               "gradiente_exacto_contexto_0": exact.tolist(),
               "gradiente_estimado_sin_baseline": g0.mean(axis=0).tolist(),
               "gradiente_estimado_con_baseline": gb.mean(axis=0).tolist()}
    rows = [row for n, v in runs.items() for row in filas_finales(n, v[:, -1])]
    return guardar(out, {**runs, "varianzas": variances}, rows, metrics, fig)
