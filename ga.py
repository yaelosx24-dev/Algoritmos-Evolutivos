"""
ga.py — Núcleo del Algoritmo Genético (AG)
===========================================
Materia : Algoritmos Evolutivos (I.S.C.) — TESI
Problema: Maximizar f(x) = x · sin(10·π·x) + 1,  x ∈ [-1, 2]

Este módulo contiene TODOS los componentes del AG, separados en funciones
para que cada uno pueda estudiarse, probarse y reemplazarse por separado:

    1. Representación  -> binaria estándar, código Gray o real (punto flotante)
    2. Evaluación      -> función objetivo f(x) + aptitud escalada (para ruleta)
    3. Selección       -> ruleta (proporcional) o torneo
    4. Cruce           -> un punto, dos puntos, uniforme (binaria)
                          aritmético / BLX-alfa (real)
    5. Mutación        -> inversión de bit (binaria)
                          gaussiana, uniforme, no uniforme, polinomial (real)
    + Elitismo opcional y registro de estadísticas por generación.

Referencias de los operadores:
    Eiben & Smith (2015), cap. 4-5; Simon (2013), cap. 3 y 8;
    Deb & Agrawal (1995) para la mutación polinomial;
    Michalewicz (1996) para la mutación no uniforme y este problema de prueba.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. PROBLEMA: función objetivo y dominio
# ---------------------------------------------------------------------------
X_MIN, X_MAX = -1.0, 2.0


def funcion_objetivo(x: float) -> float:
    """f(x) = x·sin(10πx) + 1. Es multimodal: tiene 15 máximos locales en [-1, 2]."""
    return x * math.sin(10.0 * math.pi * x) + 1.0


# ---------------------------------------------------------------------------
# 2. REPRESENTACIÓN
# ---------------------------------------------------------------------------
def bits_necesarios(precision_decimales: int) -> int:
    """Número mínimo de bits L tal que 2^L - 1 >= (b - a)·10^p.
    Con p = 6 decimales: (2-(-1))·10^6 = 3 000 000  ->  L = 22  (2^22 = 4 194 304)."""
    intervalos = (X_MAX - X_MIN) * 10 ** precision_decimales
    return math.ceil(math.log2(intervalos + 1))


def binario_a_entero(bits: list[int]) -> int:
    valor = 0
    for b in bits:
        valor = (valor << 1) | b
    return valor


def gray_a_binario(gray: list[int]) -> list[int]:
    """Decodificación Gray: b1 = g1 ;  b_i = b_{i-1} XOR g_i."""
    binario = [gray[0]]
    for g in gray[1:]:
        binario.append(binario[-1] ^ g)
    return binario


def binario_a_gray(binario: list[int]) -> list[int]:
    """Codificación Gray: g1 = b1 ;  g_i = b_{i-1} XOR b_i."""
    return [binario[0]] + [binario[i - 1] ^ binario[i] for i in range(1, len(binario))]


def decodificar(cromosoma, representacion: str) -> float:
    """Convierte el genotipo en el fenotipo x ∈ [X_MIN, X_MAX].
    Binaria: x = a + entero(c) · (b - a) / (2^L - 1)."""
    if representacion == "real":
        return cromosoma[0]
    bits = gray_a_binario(cromosoma) if representacion == "gray" else cromosoma
    L = len(bits)
    return X_MIN + binario_a_entero(bits) * (X_MAX - X_MIN) / (2 ** L - 1)


def individuo_aleatorio(rng: random.Random, representacion: str, L: int):
    if representacion == "real":
        return [rng.uniform(X_MIN, X_MAX)]
    return [rng.randint(0, 1) for _ in range(L)]


# ---------------------------------------------------------------------------
# 3. EVALUACIÓN DE APTITUD
# ---------------------------------------------------------------------------
def aptitud_escalada(valores_f: list[float]) -> list[float]:
    """La ruleta exige aptitudes positivas, pero f(x) puede ser negativa
    (mínimo ≈ -0.95). Se usa escalamiento por ventana (windowing):
        F_i = f_i - min(f) + ε,
    que conserva el orden y hace a todos los individuos seleccionables."""
    fmin = min(valores_f)
    eps = 1e-6
    return [v - fmin + eps for v in valores_f]


# ---------------------------------------------------------------------------
# 4. SELECCIÓN
# ---------------------------------------------------------------------------
def seleccion_ruleta(rng, poblacion, aptitudes, n):
    """Selección proporcional: p_i = F_i / Σ F_j. Se gira la ruleta n veces
    (muestreo estocástico con reemplazo) usando las probabilidades acumuladas q_i."""
    total = sum(aptitudes)
    acumuladas, q = [], 0.0
    for a in aptitudes:
        q += a / total
        acumuladas.append(q)
    elegidos = []
    for _ in range(n):
        r = rng.random()
        for i, qi in enumerate(acumuladas):
            if r <= qi:
                elegidos.append(list(poblacion[i]))
                break
        else:  # protección ante error de redondeo (q_n ≈ 0.99999…)
            elegidos.append(list(poblacion[-1]))
    return elegidos


def seleccion_torneo(rng, poblacion, valores_f, n, k=2):
    """Torneo determinista de tamaño k: se toman k individuos al azar y gana
    el de mayor f(x). No necesita escalamiento y la presión selectiva crece con k."""
    elegidos = []
    for _ in range(n):
        competidores = rng.sample(range(len(poblacion)), k)
        ganador = max(competidores, key=lambda i: valores_f[i])
        elegidos.append(list(poblacion[ganador]))
    return elegidos


# ---------------------------------------------------------------------------
# 5. CRUCE (recombinación)
# ---------------------------------------------------------------------------
def cruce_un_punto(rng, p1, p2):
    """Se elige c ∈ {1,…,L-1}; h1 = p1[0:c] + p2[c:],  h2 = p2[0:c] + p1[c:]."""
    c = rng.randint(1, len(p1) - 1)
    return p1[:c] + p2[c:], p2[:c] + p1[c:]


def cruce_dos_puntos(rng, p1, p2):
    """Se eligen c1 < c2; se intercambia el segmento central [c1, c2)."""
    c1, c2 = sorted(rng.sample(range(1, len(p1)), 2))
    return (p1[:c1] + p2[c1:c2] + p1[c2:],
            p2[:c1] + p1[c1:c2] + p2[c2:])


def cruce_uniforme(rng, p1, p2, p_intercambio=0.5):
    """Para cada gen i se lanza una moneda: si r < 0.5 se intercambian los genes."""
    h1, h2 = list(p1), list(p2)
    for i in range(len(p1)):
        if rng.random() < p_intercambio:
            h1[i], h2[i] = h2[i], h1[i]
    return h1, h2


def cruce_blx(rng, p1, p2, alfa=0.5):
    """BLX-α (real): h ~ U[min - α·d, max + α·d], d = |p1 - p2|, recortado al dominio."""
    h1, h2 = [], []
    for a, b in zip(p1, p2):
        lo, hi = min(a, b), max(a, b)
        d = hi - lo
        for h in (h1, h2):
            h.append(min(X_MAX, max(X_MIN, rng.uniform(lo - alfa * d, hi + alfa * d))))
    return h1, h2


def cruce_aritmetico(rng, p1, p2):
    """Aritmético (real): h1 = λ·p1 + (1-λ)·p2 ; h2 = (1-λ)·p1 + λ·p2,  λ ~ U(0,1)."""
    lam = rng.random()
    return ([lam * a + (1 - lam) * b for a, b in zip(p1, p2)],
            [(1 - lam) * a + lam * b for a, b in zip(p1, p2)])


CRUCES = {
    "un_punto": cruce_un_punto,
    "dos_puntos": cruce_dos_puntos,
    "uniforme": cruce_uniforme,
    "blx": cruce_blx,
    "aritmetico": cruce_aritmetico,
}


# ---------------------------------------------------------------------------
# 6. MUTACIÓN
# ---------------------------------------------------------------------------
def mutacion_bit_flip(rng, crom, pm, **_):
    """Cada bit se invierte (0↔1) con probabilidad pm."""
    return [1 - b if rng.random() < pm else b for b in crom]


def mutacion_gaussiana(rng, crom, pm, sigma=0.1, **_):
    """x' = x + N(0, σ²), recortado a [a, b]."""
    return [min(X_MAX, max(X_MIN, x + rng.gauss(0, sigma))) if rng.random() < pm else x
            for x in crom]


def mutacion_uniforme(rng, crom, pm, **_):
    """x' ~ U(a, b): el gen se reemplaza por un valor aleatorio del dominio."""
    return [rng.uniform(X_MIN, X_MAX) if rng.random() < pm else x for x in crom]


def mutacion_no_uniforme(rng, crom, pm, gen=0, max_gen=1, b=5.0, **_):
    """Michalewicz: x' = x + Δ(t, b_sup - x) ó x - Δ(t, x - b_inf),
    Δ(t, y) = y·(1 - r^((1 - t/T)^b)). El paso se reduce conforme t -> T."""
    salida = []
    for x in crom:
        if rng.random() < pm:
            r = rng.random()
            factor = 1 - r ** ((1 - gen / max_gen) ** b)
            if rng.random() < 0.5:
                x = x + (X_MAX - x) * factor
            else:
                x = x - (x - X_MIN) * factor
        salida.append(x)
    return salida


def mutacion_polinomial(rng, crom, pm, eta=20.0, **_):
    """Deb: u ~ U(0,1); δ = (2u)^(1/(η+1)) - 1 si u<0.5, si no 1 - (2(1-u))^(1/(η+1));
    x' = x + δ·(b - a). η grande -> perturbaciones pequeñas."""
    salida = []
    for x in crom:
        if rng.random() < pm:
            u = rng.random()
            if u < 0.5:
                delta = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            x = min(X_MAX, max(X_MIN, x + delta * (X_MAX - X_MIN)))
        salida.append(x)
    return salida


MUTACIONES = {
    "bit_flip": mutacion_bit_flip,
    "gaussiana": mutacion_gaussiana,
    "uniforme": mutacion_uniforme,
    "no_uniforme": mutacion_no_uniforme,
    "polinomial": mutacion_polinomial,
}


# ---------------------------------------------------------------------------
# 7. CICLO EVOLUTIVO
# ---------------------------------------------------------------------------
@dataclass
class Resultado:
    mejor_x: float
    mejor_f: float
    mejor_cromosoma: list
    historial_mejor: list = field(default_factory=list)      # mejor histórico por generación
    historial_promedio: list = field(default_factory=list)   # f promedio de la población
    historial_diversidad: list = field(default_factory=list) # desviación estándar de x
    generacion_estable: int = 0
    ultima_mejora: int = 0
    tiempo_s: float = 0.0
    evaluaciones: int = 0


def validar_config(cfg: dict) -> dict:
    """Completa valores por omisión y verifica combinaciones válidas."""
    base = {
        "tam_poblacion": 50, "generaciones": 100, "prob_cruce": 0.8,
        "prob_mutacion": 0.01, "semilla": 42, "representacion": "binaria",
        "seleccion": "torneo", "tam_torneo": 2, "cruce": "un_punto",
        "mutacion": "bit_flip", "elitismo": 1, "precision_decimales": 6,
        "sigma": 0.1, "eta": 20.0, "b_no_uniforme": 5.0, "alfa_blx": 0.5,
    }
    base.update({k: v for k, v in cfg.items() if not k.startswith("_")})
    if base["representacion"] not in ("binaria", "gray", "real"):
        raise ValueError("representacion debe ser 'binaria', 'gray' o 'real'")
    binarios = ("un_punto", "dos_puntos", "uniforme")
    if base["representacion"] == "real":
        if base["cruce"] in binarios:
            # con un solo gen real el cruce por puntos no tiene sentido
            base["cruce"] = "blx"
        if base["mutacion"] == "bit_flip":
            base["mutacion"] = "gaussiana"
    else:
        if base["cruce"] not in binarios:
            raise ValueError("Con representación binaria/gray use un_punto, dos_puntos o uniforme")
        base["mutacion"] = "bit_flip"
    if base["tam_poblacion"] < 2 or base["generaciones"] < 1:
        raise ValueError("tam_poblacion >= 2 y generaciones >= 1")
    for p in ("prob_cruce", "prob_mutacion"):
        if not 0 <= base[p] <= 1:
            raise ValueError(f"{p} debe estar en [0, 1]")
    return base


def ejecutar_ag(cfg: dict, verbose: bool = False) -> Resultado:
    cfg = validar_config(cfg)
    rng = random.Random(cfg["semilla"])
    rep = cfg["representacion"]
    L = bits_necesarios(cfg["precision_decimales"])
    N, T = cfg["tam_poblacion"], cfg["generaciones"]
    pc, pm = cfg["prob_cruce"], cfg["prob_mutacion"]
    cruzar = CRUCES[cfg["cruce"]]
    mutar = MUTACIONES[cfg["mutacion"]]
    params_mut = {"sigma": cfg["sigma"], "eta": cfg["eta"],
                  "b": cfg["b_no_uniforme"], "max_gen": T}

    t0 = time.perf_counter()
    poblacion = [individuo_aleatorio(rng, rep, L) for _ in range(N)]
    evaluaciones = 0

    mejor_crom, mejor_f = None, -math.inf
    hist_mejor, hist_prom, hist_div = [], [], []

    for gen in range(T + 1):
        # --- Evaluación ---
        xs = [decodificar(c, rep) for c in poblacion]
        fs = [funcion_objetivo(x) for x in xs]
        evaluaciones += N
        i_mejor = max(range(N), key=lambda i: fs[i])
        if fs[i_mejor] > mejor_f:
            mejor_f, mejor_crom = fs[i_mejor], list(poblacion[i_mejor])
        media_x = sum(xs) / N
        hist_mejor.append(mejor_f)
        hist_prom.append(sum(fs) / N)
        hist_div.append(math.sqrt(sum((x - media_x) ** 2 for x in xs) / N))
        if verbose and (gen % max(1, T // 10) == 0 or gen == T):
            print(f"  gen {gen:4d} | mejor f = {mejor_f:.6f} | f prom = {hist_prom[-1]:.4f}"
                  f" | div(x) = {hist_div[-1]:.4f}")
        if gen == T:
            break

        # --- Elitismo: se guardan los e mejores sin modificar ---
        orden = sorted(range(N), key=lambda i: fs[i], reverse=True)
        elite = [list(poblacion[i]) for i in orden[:cfg["elitismo"]]]

        # --- Selección ---
        n_hijos = N - len(elite)
        if cfg["seleccion"] == "ruleta":
            padres = seleccion_ruleta(rng, poblacion, aptitud_escalada(fs), n_hijos)
        else:
            padres = seleccion_torneo(rng, poblacion, fs, n_hijos, cfg["tam_torneo"])

        # --- Cruce (por parejas, con probabilidad pc) ---
        hijos = []
        for i in range(0, n_hijos, 2):
            p1 = padres[i]
            p2 = padres[i + 1] if i + 1 < n_hijos else padres[0]
            if rng.random() < pc:
                if cfg["cruce"] == "blx":
                    h1, h2 = cruzar(rng, p1, p2, cfg["alfa_blx"])
                else:
                    h1, h2 = cruzar(rng, p1, p2)
            else:
                h1, h2 = list(p1), list(p2)
            hijos.extend([h1, h2])
        hijos = hijos[:n_hijos]

        # --- Mutación ---
        hijos = [mutar(rng, h, pm, gen=gen, **params_mut) for h in hijos]

        # --- Reemplazo generacional con elitismo ---
        poblacion = elite + hijos

    tiempo = time.perf_counter() - t0
    # Generación de estabilización: primera generación en que el mejor histórico
    # queda a menos de 1e-4 de su valor final (las mejoras posteriores son
    # ajustes de la 5a-6a cifra decimal). Se guarda también la última mejora.
    gen_estable = next(g for g, v in enumerate(hist_mejor) if v >= mejor_f - 1e-4)
    ultima_mejora = next(g for g, v in enumerate(hist_mejor) if v >= mejor_f - 1e-12)

    return Resultado(
        mejor_x=decodificar(mejor_crom, rep), mejor_f=mejor_f, mejor_cromosoma=mejor_crom,
        historial_mejor=hist_mejor, historial_promedio=hist_prom,
        historial_diversidad=hist_div, generacion_estable=gen_estable,
        ultima_mejora=ultima_mejora,
        tiempo_s=tiempo, evaluaciones=evaluaciones,
    )


# ---------------------------------------------------------------------------
# 8. VALIDACIÓN: fuerza bruta + refinamiento analítico (Newton sobre f'(x)=0)
# ---------------------------------------------------------------------------
def optimo_fuerza_bruta(paso: float = 1e-6):
    """Evalúa f en una malla de paso 1e-6 (3 000 001 puntos) y regresa el máximo."""
    import numpy as np
    x = np.arange(X_MIN, X_MAX + paso / 2, paso)
    y = x * np.sin(10 * np.pi * x) + 1
    i = int(np.argmax(y))
    return float(x[i]), float(y[i])


def optimo_analitico(x0: float, tol: float = 1e-14, max_iter: int = 50):
    """Condición de primer orden: f'(x) = sin(10πx) + 10πx·cos(10πx) = 0.
    Se resuelve con Newton-Raphson partiendo de la estimación de fuerza bruta.
    f''(x) = 20π·cos(10πx) - 100π²x·sin(10πx)  (< 0 confirma un máximo)."""
    k = 10 * math.pi
    x = x0
    for _ in range(max_iter):
        d1 = math.sin(k * x) + k * x * math.cos(k * x)
        d2 = 2 * k * math.cos(k * x) - k * k * x * math.sin(k * x)
        paso = d1 / d2
        x -= paso
        if abs(paso) < tol:
            break
    d2 = 2 * k * math.cos(k * x) - k * k * x * math.sin(k * x)
    return x, funcion_objetivo(x), d2
