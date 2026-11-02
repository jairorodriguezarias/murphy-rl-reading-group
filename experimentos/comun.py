"""Utilidades numéricas y escritura de resultados; no hay descargas al ejecutar."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": .2,
                     "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

def softmax(logits):
    z = np.asarray(logits, dtype=float)
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def intervalo(x):
    """Media e IC normal aproximado del 95%, entre ejecuciones independientes."""
    x = np.asarray(x)
    mean = x.mean(axis=0)
    half = 1.96 * x.std(axis=0, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.zeros_like(mean)
    return mean, half

def curva(ax, x, runs, label):
    mean, half = intervalo(runs)
    line, = ax.plot(x, mean, label=label)
    ax.fill_between(x, mean-half, mean+half, color=line.get_color(), alpha=.16)

def guardar(out, arrays, rows, metrics, fig):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "datos.npz", **arrays)
    with (out / "metricas.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "resumen.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    fig.tight_layout()
    fig.savefig(out / "figura.png", bbox_inches="tight")
    fig.savefig(out / "figura.svg", bbox_inches="tight")
    plt.close(fig)
    return metrics

def filas_finales(method, values):
    return [{"metodo": method, "replica": i, "valor_final": float(v)}
            for i, v in enumerate(values)]

def resumen_muestra(values):
    mean, half = intervalo(np.asarray(values))
    return {"media": float(mean), "ic95_aprox_semianchura": float(half)}

def cadena(n=7, slip=.15, coste=-.01):
    """Estados 0 y n-1 terminales; izquierda/derecha, recompensa al entrar en meta."""
    p = np.zeros((n, 2, n))
    r = np.zeros((n, 2, n))
    for s in range(1, n-1):
        for a in range(2):
            for actual, prob in ((a, 1-slip), (1-a, slip)):
                ns = s + (1 if actual else -1)
                p[s, a, ns] += prob
                r[s, a, ns] = 1.0 if ns == n-1 else coste
    p[0, :, 0] = 1
    p[-1, :, -1] = 1
    return p, r

def iteracion_valor(p, r, gamma=.95, tol=1e-12):
    v = np.zeros(len(p))
    for _ in range(10000):
        q = (p*(r + gamma*v[None, None, :])).sum(axis=2)
        nv = q.max(axis=1)
        if np.max(np.abs(nv-v)) < tol:
            return nv, q
        v = nv
    raise RuntimeError("La iteración de valor no convergió.")

def evaluar_politica(p, r, policy, gamma=.95):
    """Evaluación exacta; no altera el aprendizaje ni consume su RNG."""
    ps = p[np.arange(len(p)), policy]
    rs = (ps*r[np.arange(len(p)), policy]).sum(axis=1)
    return np.linalg.solve(np.eye(len(p))-gamma*ps, rs)

def greedy_aleatorio(q, rng):
    return int(rng.choice(np.flatnonzero(np.isclose(q, q.max(), atol=1e-12, rtol=0))))
