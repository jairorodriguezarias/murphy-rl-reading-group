"""Cap. 4: Dyna-Q, coste de planificación y eficiencia en datos reales."""
import numpy as np
import matplotlib.pyplot as plt
from .comun import (cadena, iteracion_valor, greedy_aleatorio, guardar, curva,
                    filas_finales, resumen_muestra)

def entrenar(seed, planning, pasos=1600):
    rng_act = np.random.default_rng(seed)
    rng_plan = np.random.default_rng(seed+100000)
    p, r = cadena(n=9, slip=0, coste=0)
    _, qstar = iteracion_valor(p, r)
    q = np.zeros((9, 2))
    model = {}
    keys = []
    s, goals = 1, 0
    errors, successes = [], []
    for t in range(pasos+1):
        if t % 20 == 0:
            errors.append(np.sqrt(np.mean((q[1:-1]-qstar[1:-1])**2)))
            successes.append(goals)
        if t == pasos:
            break
        a = int(rng_act.integers(2)) if rng_act.random() < .2 else greedy_aleatorio(q[s], rng_act)
        ns = s+(1 if a else -1)
        reward, done = float(ns == 8), ns in (0, 8)
        q[s, a] += .2*(reward+.95*(not done)*q[ns].max()-q[s, a])
        if (s, a) not in model:
            keys.append((s, a))
        model[s, a] = (ns, reward, done)
        for _ in range(planning):
            ms, ma = keys[int(rng_plan.integers(len(keys)))]
            mn, mr, md = model[ms, ma]
            q[ms, ma] += .2*(mr+.95*(not md)*q[mn].max()-q[ms, ma])
        goals += ns == 8
        s = 1 if done else ns
    return np.array(errors), np.array(successes)

def ejecutar(out, seed=2026, replicas=20):
    arrays, rows, metrics = {}, [], {"pasos_reales": 1600, "epsilon": .2, "alpha": .2, "gamma": .95}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for n in [0, 5, 20]:
        data = [entrenar(seed+k, n) for k in range(replicas)]
        err, goals = np.array([x[0] for x in data]), np.array([x[1] for x in data])
        name = f"{n} pasos de planificación"
        x = np.arange(err.shape[1])*20
        curva(ax[0], x, err, name)
        curva(ax[1], x, goals, name)
        arrays[f"rmse_{n}"], arrays[f"metas_{n}"] = err, goals
        rows += filas_finales(name, err[:, -1])
        metrics[str(n)] = {"rmse_final": resumen_muestra(err[:, -1]),
                          "metas_alcanzadas": resumen_muestra(goals[:, -1]),
                          "backups_totales_por_replica": 1600*(1+n)}
    ax[0].set(xlabel="Transiciones reales", ylabel="RMSE frente a Q*", title="Propagar información mediante el modelo")
    ax[1].set(xlabel="Transiciones reales", ylabel="Metas acumuladas durante entrenamiento", title="Más planificación requiere más cómputo")
    ax[0].legend()
    return guardar(out, arrays, rows, metrics, fig)
