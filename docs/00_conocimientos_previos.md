# Conocimientos previos

Esta guía asume Python básico y repasa las matemáticas necesarias. No hace falta conocer RL antes de empezar. La prioridad es comprender las expectativas, el condicionamiento y los gradientes; las demostraciones de convergencia pueden quedar para una segunda lectura.

## 1. Probabilidad: pensar en resultados posibles

Una variable aleatoria representa un resultado incierto. Para una recompensa discreta, su esperanza es

$$
\mathbb{E}[R]=\sum_r p(r)r.
$$

Una acción que entrega 10 con probabilidad 0.2 y 0 en el resto de casos tiene recompensa esperada 2. Esto no significa que entregue 2 en una ejecución. La varianza mide la dispersión alrededor de la media:

$$
\operatorname{Var}(R)=\mathbb{E}[(R-\mathbb{E}[R])^2].
$$

La distribución condicional $p(s'|s,a)$ describe qué puede pasar después de actuar. La ley de la esperanza total permite separar decisiones y consecuencias:

$$
\mathbb{E}[R|s]=\sum_a \pi(a|s)\sum_{s'}p(s'|s,a)\mathbb{E}[R|s,a,s'].
$$

**Ejercicio:** una política elige A el 75% de las veces y B el 25%. Sus recompensas esperadas son 2 y 6. La recompensa esperada de la política es $0.75(2)+0.25(6)=3$.

**Monte Carlo** significa aproximar una esperanza mediante una media de muestras. Con muestras independientes y varianza finita, el error estándar de la media disminuye como $1/\sqrt{n}$. Multiplicar por cuatro las muestras reduce ese error aproximadamente a la mitad. Las transiciones consecutivas de una misma trayectoria suelen estar correlacionadas: no deben contarse como ejecuciones independientes.

## 2. Bayes y dos tipos de incertidumbre

La regla de Bayes actualiza una creencia previa:

$$
p(\theta|D)=\frac{p(D|\theta)p(\theta)}{p(D)}.
$$

En un bandit Bernoulli, una acción da éxito o fracaso. Si su probabilidad desconocida de éxito tiene distribución $\operatorname{Beta}(\alpha,\beta)$, observar un éxito incrementa $\alpha$ y observar un fracaso incrementa $\beta$. La media posterior es $\alpha/(\alpha+\beta)$.

Con prior Beta(1,1), tres éxitos y un fracaso producen Beta(4,2), cuya media es 2/3. Esta será la base de Thompson sampling en el capítulo 7.

- **Incertidumbre epistémica:** desconocemos la probabilidad de éxito; los datos pueden reducirla.
- **Incertidumbre aleatoria:** incluso con probabilidad conocida, el siguiente resultado puede seguir siendo impredecible.

Confundir ambas puede hacer que un agente busque ruido en vez de información.

## 3. Álgebra lineal: tablas, vectores y transformaciones

Debes reconocer un vector, un producto escalar y una multiplicación matriz-vector. Si $w=(2,-1)$ y $\phi(s)=(3,4)$, entonces $w^\top\phi(s)=2$. Una función de valor lineal usa precisamente $V_w(s)=w^\top\phi(s)$.

Una matriz de transición contiene probabilidades: cada fila suma 1. Con una política fija, $P_\pi$ describe cómo cambia el estado. La ecuación de evaluación es

$$
V^\pi=r_\pi+\gamma P_\pi V^\pi,
\qquad
(I-\gamma P_\pi)V^\pi=r_\pi.
$$

Para un MDP finito con $\gamma<1$, podemos resolver este sistema. En código se usa un solucionador lineal; no hace falta calcular explícitamente la inversa.

**Ejemplo:** un único estado da recompensa 2 cada paso y vuelve a sí mismo. Con $\gamma=0.9$, $V=2+0.9V$, de donde $V=20$.

## 4. Cálculo y optimización

El gradiente indica cómo cambia una función al modificar sus parámetros. Para minimizar una pérdida usamos descenso; para maximizar retorno usamos ascenso:

$$
\theta_{\mathrm{nuevo}}=\theta+\eta\nabla_\theta J(\theta).
$$

$\eta$ es la tasa de aprendizaje: demasiado grande puede causar oscilaciones; demasiado pequeña puede hacer el aprendizaje lento.

La regla de la cadena permite derivar una composición de funciones, como una red neuronal. La identidad más importante para el capítulo 3 es

$$
\nabla_\theta p_\theta(x)=p_\theta(x)\nabla_\theta\log p_\theta(x).
$$

Permite estimar el gradiente de una recompensa mediante muestras, aunque el entorno no sea diferenciable. Para políticas softmax,

$$
\pi_\theta(a)=\frac{e^{\theta_a}}{\sum_b e^{\theta_b}},
\qquad
\frac{\partial\log\pi_\theta(a)}{\partial\theta_j}
=\mathbf{1}[a=j]-\pi_\theta(j).
$$

El laboratorio 3 implementa esta expresión directamente con NumPy. Restar el máximo de los logits antes de exponenciar evita desbordamientos sin cambiar las probabilidades.

## 5. Aprendizaje automático

Conviene distinguir:

| Concepto | Significado | Dónde aparece |
|---|---|---|
| Regresión | Predecir una cantidad numérica | Valores, recompensas |
| Clasificación | Predecir una distribución sobre categorías | Acciones discretas, tokens |
| Máxima verosimilitud | Aumentar la probabilidad de datos observados | Modelos del mundo, imitación |
| Sobreajuste | Mejorar en los datos disponibles sin generalizar | Todos los capítulos |
| Regularización | Restringir o penalizar soluciones | PPO, SAC, RLHF, offline RL |
| Representación | Transformar observaciones en variables útiles | DQN, Dreamer, JEPA |
| Cambio de distribución | Evaluar en estados distintos a los de entrenamiento | Modelos aprendidos, offline RL |

Una red neuronal es una función parametrizada que transforma entradas en salidas mediante capas. El entrenamiento ajusta parámetros a partir de una pérdida. En RL los datos cambian cuando cambia la política, y los objetivos pueden depender de estimaciones aprendidas. Esto complica el problema respecto a una regresión con datos fijos.

**Entrenamiento y evaluación deben separarse.** En los laboratorios 2, 3 y 6 calculamos algunas métricas con el modelo conocido del juguete; el agente no utiliza ese conocimiento para actualizarse. Ese acceso es una herramienta de evaluación, no una capacidad que debamos suponer en un problema real.

## 6. Entropía y divergencia KL

La entropía mide la dispersión de una distribución:

$$
H(\pi)=-\sum_a\pi(a)\log\pi(a).
$$

Una política determinista tiene entropía 0; una uniforme sobre $K$ acciones tiene entropía $\log K$. Entropía alta significa variedad de acciones, pero no garantiza exploración útil.

La divergencia KL compara distribuciones:

$$
D_{\mathrm{KL}}(p\|q)=\sum_a p(a)\log\frac{p(a)}{q(a)}.
$$

Es no negativa y, en general, no simétrica. Si $q$ asigna probabilidad cero a un resultado al que $p$ asigna probabilidad positiva, la KL es infinita. En el laboratorio 6 la referencia tiene soporte completo.

Al maximizar $\mathbb{E}_\pi[r]-\beta D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})$, el primer término favorece respuestas bien puntuadas y el segundo penaliza alejarse de la referencia. Una $\beta$ grande refuerza esa penalización.

## 7. Procesos de Markov y notación mínima

La propiedad de Markov afirma que el estado actual contiene la información necesaria para predecir el siguiente estado, dada la acción. No afirma que el mundo sea determinista ni que la observación revele el estado.

| Símbolo | Lectura |
|---|---|
| $s_t$, $o_t$ | Estado y observación |
| $a_t$, $r_t$ | Acción y recompensa de esa transición |
| $\pi(a|s)$ | Política: distribución de acciones |
| $p(s'|s,a)$ | Modelo de transición |
| $\gamma$ | Descuento temporal |
| $G_t$ | Retorno de una trayectoria desde $t$ |
| $V^\pi(s)$ | Retorno esperado desde $s$ siguiendo $\pi$ |
| $Q^\pi(s,a)$ | Retorno esperado al empezar con $a$ |
| $A^\pi(s,a)$ | Ventaja: $Q^\pi(s,a)-V^\pi(s)$ |
| $\alpha$, $\eta$ | Tasas de aprendizaje |
| $\theta$, $w$ | Parámetros aprendidos |

Usaremos $r_t$ para la recompensa producida por $a_t$, como en gran parte de Murphy. Otros textos la llaman $r_{t+1}$; cambia el índice, no el concepto.

## 8. Python necesario para los laboratorios

Necesitas ejecutar un módulo, importar funciones y trabajar con arrays:

~~~python
import numpy as np
recompensas = np.array([0.0, 0.0, 1.0])
gamma = 0.9
retorno = np.sum(gamma ** np.arange(len(recompensas)) * recompensas)
print(retorno)  # 0.81
~~~

Un generador aleatorio con semilla permite repetir una ejecución. Mantener la semilla no basta para reproducir cualquier entorno: también importan versiones, código y configuración. Aquí quedan registrados en el manifiesto de resultados.

## 9. Lo mínimo de LLM y teoría de juegos

Para el capítulo 5, piensa en una matriz: las filas son las acciones de un jugador, las columnas las del otro y cada casilla indica las recompensas. Una estrategia mixta es una distribución sobre acciones. Una mejor respuesta maximiza la recompensa frente a una estrategia rival fija.

Para el capítulo 6, un token es una unidad de texto. Un modelo autorregresivo asigna

$$
p(y|x)=\prod_{t=1}^{T}p(y_t|x,y_{<t}).
$$

Una respuesta completa es una acción grande; cada token también puede verse como una acción dentro de un episodio. SFT aprende de respuestas demostradas; RLFT aprende a partir de recompensas. Para la primera lectura no necesitas implementar atención, LoRA ni entrenamiento distribuido.

## 10. Autodiagnóstico con soluciones

Intenta resolver estas preguntas antes de mirar las respuestas.

1. ¿Cuánto vale una secuencia de recompensas $(1,0,2)$ con $\gamma=0.5$?
2. ¿Un valor esperado de 3 implica observar recompensa 3?
3. ¿Qué posterior resulta de Beta(1,1), dos éxitos y tres fracasos?
4. Si $Q(s,a)=4$ y $V(s)=3$, ¿cuál es la ventaja?
5. ¿Qué diferencia hay entre observación y estado?
6. ¿Qué hace el signo positivo en el ascenso por gradiente?
7. ¿Una política de alta entropía siempre es mejor?
8. ¿Podemos concluir que A supera a B porque su mejor semilla obtuvo más recompensa?

**Soluciones:** 1) $1+0.5^2(2)=1.5$. 2) No: es una media ponderada. 3) Beta(3,4), media 3/7. 4) 1. 5) La observación puede ocultar información relevante del estado. 6) Intenta aumentar el objetivo localmente. 7) No: depende de recompensa, incertidumbre y objetivo. 8) No: compara distribuciones de ejecuciones bajo un protocolo común.

Si fallas 1–4, dedica una sesión a probabilidad y notación. Si fallas 5–8, empieza con los laboratorios 1 y 8 antes de interpretar curvas de aprendizaje.

## Ruta de preparación

Una propuesta flexible: 60–90 minutos para probabilidad y Bayes, 45–60 para álgebra y cálculo, 45 para ML y entropía, y 30–45 para ejecutar el primer cuaderno. Son tiempos de estudio orientativos, no requisitos del libro.

**Fuente y alcance:** material preparatorio elaborado para esta guía, conectado con [Murphy v5, capítulos 1–3](https://arxiv.org/html/2412.05265v5#Ch1). Las explicaciones y ejercicios son propios.
