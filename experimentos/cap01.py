"""Cap. 1: descuento temporal y observación parcial (políticas prescritas)."""
import numpy as np
import matplotlib.pyplot as plt
from .comun import guardar, intervalo, resumen_muestra

def ejecutar(out, seed=2026, replicas=20):
    gamma = np.linspace(0, 1, 201)
    # A paga 1 ahora. B paga 3 en t=4: hay cuatro factores de descuento.
    inmediato = np.ones_like(gamma)
    demorado = 3*gamma**4
    umbral = (1/3)**.25
    resultados = np.empty((replicas, 2))
    for k in range(replicas):
        rng = np.random.default_rng(seed+k)
        # Una señal revela el bit al empezar; al decidir solo se observa "?".
        bit = rng.integers(0, 2, 1000)
        sin_memoria = rng.integers(0, 2, len(bit))
        con_memoria = bit.copy()
        resultados[k] = [(sin_memoria == bit).mean(), (con_memoria == bit).mean()]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(gamma, inmediato, label="1 ahora")
    ax[0].plot(gamma, demorado, label="3 en t=4")
    ax[0].axvline(umbral, ls=":", color="black", label=f"Umbral {umbral:.3f}")
    ax[0].set(xlabel="Descuento γ", ylabel="Retorno desde t=0", title="El objetivo cambia con γ")
    ax[0].legend()
    mean, half = intervalo(resultados)
    ax[1].bar(["Sin memoria", "Con memoria"], mean, yerr=half, capsize=5)
    ax[1].set(ylabel="Proporción de aciertos", ylim=(0, 1.08), title="La observación actual es insuficiente")
    rows = [{"replica": k, "sin_memoria": float(v[0]), "con_memoria": float(v[1])}
            for k, v in enumerate(resultados)]
    metrics = {"umbral_gamma": umbral, "episodios_por_replica": 1000,
               "sin_memoria": resumen_muestra(resultados[:, 0]),
               "con_memoria": resumen_muestra(resultados[:, 1])}
    return guardar(out, dict(gamma=gamma, inmediato=inmediato, demorado=demorado,
                            aciertos=resultados), rows, metrics, fig)
