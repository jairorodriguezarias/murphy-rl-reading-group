"""Comprobaciones con soluciones analíticas, diferencias finitas e invariantes."""
import unittest
import numpy as np
from experimentos.comun import cadena, iteracion_valor, evaluar_politica, softmax
from experimentos.cap03 import entrenar as reinforce
from experimentos.cap05 import nashconv
from experimentos.cap06 import objetivo, gradiente_exacto, REFERENCIA, PROXY
from experimentos.cap07 import simular

class Numerica(unittest.TestCase):
    def test_cadena_solucion_cerrada_y_terminales(self):
        p, r = cadena(n=7, slip=0, coste=0)
        v, q = iteracion_valor(p, r, .9)
        esperado = np.array([0]+[.9**(5-s) for s in range(1, 6)]+[0])
        np.testing.assert_allclose(v, esperado, atol=1e-10)
        np.testing.assert_allclose(q[[0, 6]], 0)
        np.testing.assert_allclose(evaluar_politica(p, r, np.ones(7, int), .9), esperado)

    def test_residuo_bellman_y_probabilidades(self):
        p, r = cadena()
        np.testing.assert_allclose(p.sum(axis=-1), 1)
        self.assertTrue(np.all(p >= 0))
        v, q = iteracion_valor(p, r)
        backup = (p*(r+.95*v[None, None, :])).sum(axis=2).max(axis=1)
        np.testing.assert_allclose(v, backup, atol=1e-10)
        np.testing.assert_allclose(evaluar_politica(p, r, q.argmax(axis=1)), v, atol=1e-10)

    def test_softmax_y_baseline(self):
        theta = np.array([.2, -.4, 1.])
        p = softmax(theta)
        np.testing.assert_allclose(softmax(theta+10000), p, atol=1e-12)
        score = np.eye(3)-p
        r = np.array([1., 0., -1.])
        g0 = (p[:, None]*r[:, None]*score).sum(axis=0)
        gb = (p[:, None]*(r[:, None]-7)*score).sum(axis=0)
        np.testing.assert_allclose(g0, gb, atol=1e-14)

    def test_gradiente_regularizado_diferencias_finitas(self):
        theta = np.array([.1, -.5, .2, .7])
        eps, beta = 1e-6, .6
        numeric = np.array([(objetivo(theta+eps*e, PROXY, beta)-objetivo(theta-eps*e, PROXY, beta))/(2*eps)
                            for e in np.eye(4)])
        np.testing.assert_allclose(gradiente_exacto(theta, PROXY, beta), numeric, atol=1e-8)

    def test_optimo_regularizado_analitico(self):
        beta = .5
        theta = np.log(REFERENCIA)+PROXY/beta
        np.testing.assert_allclose(gradiente_exacto(theta, PROXY, beta), 0, atol=1e-12)
        rng = np.random.default_rng(7)
        best = objetivo(theta, PROXY, beta)
        self.assertTrue(all(objetivo(rng.normal(size=4), PROXY, beta) <= best+1e-12 for _ in range(100)))

    def test_esperanza_rloo_por_enumeracion(self):
        # Enumera las 16 parejas de acciones de un lote de tamaño 2.
        theta, beta = np.array([.2, -.1, .6, -.3]), .4
        p = softmax(theta)
        expected = np.zeros(4)
        for a in range(4):
            for b in range(4):
                actions = np.array([a, b])
                rewards = PROXY[actions]
                baselines = rewards[::-1]
                signal = rewards-baselines-beta*np.log(p[actions]/REFERENCIA[actions])
                estimate = np.mean(signal[:, None]*(np.eye(4)[actions]-p), axis=0)
                expected += p[a]*p[b]*estimate
        np.testing.assert_allclose(expected, gradiente_exacto(theta, PROXY, beta), atol=1e-12)

    def test_nashconv_equilibrio_y_pura(self):
        uniform = np.ones(3)/3
        self.assertAlmostEqual(nashconv(uniform, uniform), 0)
        self.assertAlmostEqual(nashconv(np.array([1,0,0]), np.array([1,0,0])), 2)

    def test_reproducibilidad_y_cotas_regret(self):
        np.testing.assert_array_equal(reinforce(7, True, pasos=20), reinforce(7, True, pasos=20))
        for changing in [False, True]:
            v = simular("Thompson", changing, 7, 3, pasos=200)
            np.testing.assert_array_equal(v, simular("Thompson", changing, 7, 3, pasos=200))
            self.assertTrue(np.all(np.diff(v, axis=1) >= -1e-12))
            self.assertTrue(np.all(v[:, -1] <= .6*200+1e-8))

if __name__ == "__main__":
    unittest.main()
