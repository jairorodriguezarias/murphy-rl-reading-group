"""Cierre metodológico añadido a los agradecimientos; no es un método del cap. 8."""
import numpy as np
import matplotlib.pyplot as plt
from .comun import guardar

def ejecutar(out, seed=2026, replicas=20):
    rng = np.random.default_rng(seed)
    # Dos algoritmos idénticos N(0,1). Cada estudio tiene 20 semillas por algoritmo.
    estudios = max(2000, replicas*100)
    a = rng.normal(size=(estudios, 20))
    b = rng.normal(size=(estudios, 20))
    honest = a.mean(axis=1)-b.mean(axis=1)
    selected = a.max(axis=1)-b.mean(axis=1)
    # Evaluación independiente del resultado elegido: no reutiliza su ruido.
    heldout = rng.normal(size=estudios)-rng.normal(size=(estudios, 20)).mean(axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(honest, bins=45, density=True, alpha=.6, label="Media A − media B")
    ax[0].hist(selected, bins=45, density=True, alpha=.6, label="Mejor semilla A − media B")
    ax[0].axvline(0, color="black", ls="--")
    ax[0].set(xlabel="Ventaja aparente", ylabel="Densidad", title="A y B tienen idéntico rendimiento real")
    ax[0].legend(fontsize=8)
    ax[1].bar(["Todas las\nsemillas", "Selección de\nla mejor", "Evaluación\nindependiente"],
              [honest.mean(), selected.mean(), heldout.mean()])
    ax[1].set(ylabel="Ventaja media observada", title="La selección fabrica una mejora")
    rows = [{"estudio": i, "diferencia_medias": float(honest[i]),
             "ventaja_seleccionada": float(selected[i]), "ventaja_independiente": float(heldout[i])}
            for i in range(estudios)]
    metrics = {"estudios": estudios, "semillas_por_algoritmo": 20,
               "diferencia_real": 0, "diferencia_media_observada": float(honest.mean()),
               "ventaja_seleccionando_mejor_semilla": float(selected.mean()),
               "ventaja_evaluacion_independiente": float(heldout.mean()),
               "nota": "Simulación estadística añadida por esta guía; vinculada a §1.3.7."}
    return guardar(out, dict(honest=honest, selected=selected, heldout=heldout), rows, metrics, fig)
