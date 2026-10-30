"""
experimentos.py — Diseño experimental (Actividad 4)
===================================================
Ejecuta cada configuración con 10 semillas distintas (el mínimo pedido es 5),
calcula promedio y desviación estándar del mejor valor, generación de
estabilización, tiempo y diversidad, y genera tablas CSV y gráficas.

Uso:  python experimentos.py
"""

import csv
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ga import ejecutar_ag, optimo_fuerza_bruta

SALIDA = "resultados"
SEMILLAS = list(range(1, 11))
UMBRAL_EXITO = 1e-3        # |f* - f| < 1e-3  => se alcanzó el pico global
UMBRAL_PREMATURA = 2.80    # f < 2.80 => quedó atrapado en un óptimo local

BASE = {"generaciones": 100, "representacion": "binaria", "seleccion": "torneo",
        "tam_torneo": 2, "cruce": "un_punto", "mutacion": "bit_flip", "elitismo": 1}

# Experimento A: variación de N, pc y pm (diseño principal pedido)
CONFIGS_A = {
    "C1": {"tam_poblacion": 10,  "prob_cruce": 0.8, "prob_mutacion": 0.01},
    "C2": {"tam_poblacion": 50,  "prob_cruce": 0.8, "prob_mutacion": 0.01},
    "C3": {"tam_poblacion": 100, "prob_cruce": 0.8, "prob_mutacion": 0.01},
    "C4": {"tam_poblacion": 50,  "prob_cruce": 0.5, "prob_mutacion": 0.01},
    "C5": {"tam_poblacion": 50,  "prob_cruce": 0.8, "prob_mutacion": 0.001},
    "C6": {"tam_poblacion": 50,  "prob_cruce": 0.8, "prob_mutacion": 0.10},
}

# Experimento B: operadores (contraste teórico-práctico)
CONFIGS_B = {
    "Binaria + ruleta + 1 punto":    {"seleccion": "ruleta"},
    "Binaria + torneo + 1 punto":    {},
    "Binaria + torneo + 2 puntos":   {"cruce": "dos_puntos"},
    "Binaria + torneo + uniforme":   {"cruce": "uniforme"},
    "Gray + torneo + 1 punto":       {"representacion": "gray"},
    "Real + torneo + BLX + gauss":   {"representacion": "real", "cruce": "blx",
                                      "mutacion": "gaussiana", "prob_mutacion": 0.1},
    "Real + torneo + BLX + polin.":  {"representacion": "real", "cruce": "blx",
                                      "mutacion": "polinomial", "prob_mutacion": 0.1},
}


def correr(nombre, extra, f_opt):
    filas, curvas, divs = [], [], []
    for s in SEMILLAS:
        cfg = {**BASE, "tam_poblacion": 50, "prob_cruce": 0.8, "prob_mutacion": 0.01,
               **extra, "semilla": s}
        r = ejecutar_ag(cfg)
        filas.append({"config": nombre, "semilla": s, "mejor_x": r.mejor_x, "mejor_f": r.mejor_f,
                      "gen_estable": r.generacion_estable, "ultima_mejora": r.ultima_mejora, "tiempo_s": r.tiempo_s,
                      "div_final": r.historial_diversidad[-1],
                      "exito": int(abs(f_opt - r.mejor_f) < UMBRAL_EXITO)})
        curvas.append(r.historial_mejor)
        divs.append(r.historial_diversidad)
    return filas, np.array(curvas), np.array(divs)


def resumen(nombre, cfg, filas):
    fs = [f["mejor_f"] for f in filas]
    return {
        "config": nombre, **{k: cfg.get(k) for k in ("tam_poblacion", "prob_cruce", "prob_mutacion")},
        "f_prom": st.mean(fs), "f_desv": st.stdev(fs), "f_max": max(fs), "f_min": min(fs),
        "gen_prom": st.mean(f["gen_estable"] for f in filas),
        "gen_desv": st.stdev(f["gen_estable"] for f in filas),
        "t_prom_ms": 1000 * st.mean(f["tiempo_s"] for f in filas),
        "div_final": st.mean(f["div_final"] for f in filas),
        "exitos": sum(f["exito"] for f in filas),
        "prematuras": sum(1 for v in fs if v < UMBRAL_PREMATURA),
    }


def guardar_csv(ruta, filas):
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)


def main():
    os.makedirs(SALIDA, exist_ok=True)
    x_opt, f_opt = optimo_fuerza_bruta()
    print(f"Óptimo de referencia: x={x_opt:.6f}, f={f_opt:.6f}")

    # ---------------- Experimento A ----------------
    todas, resumenes, curvas_A, divs_A = [], [], {}, {}
    for nombre, extra in CONFIGS_A.items():
        filas, curvas, divs = correr(nombre, extra, f_opt)
        cfg = {"tam_poblacion": 50, "prob_cruce": 0.8, "prob_mutacion": 0.01, **extra}
        todas += filas
        resumenes.append(resumen(nombre, cfg, filas))
        curvas_A[nombre], divs_A[nombre] = curvas, divs
    guardar_csv(f"{SALIDA}/experimento_A_corridas.csv", todas)
    guardar_csv(f"{SALIDA}/experimento_A_resumen.csv", resumenes)

    # ---------------- Experimento B ----------------
    todas_B, res_B, curvas_B = [], [], {}
    for nombre, extra in CONFIGS_B.items():
        filas, curvas, _ = correr(nombre, extra, f_opt)
        todas_B += filas
        cfg = {"tam_poblacion": 50, "prob_cruce": 0.8, "prob_mutacion": 0.01, **extra}
        res_B.append(resumen(nombre, cfg, filas))
        curvas_B[nombre] = curvas
    guardar_csv(f"{SALIDA}/experimento_B_corridas.csv", todas_B)
    guardar_csv(f"{SALIDA}/experimento_B_resumen.csv", res_B)

    with open(f"{SALIDA}/resumen.json", "w", encoding="utf-8") as fh:
        json.dump({"optimo": {"x": x_opt, "f": f_opt}, "A": resumenes, "B": res_B,
                   "A_corridas": todas}, fh, indent=2, ensure_ascii=False)

    # ---------------- Gráficas ----------------
    colores = ["#1f5fa8", "#e08a1e", "#2e7d32", "#8e24aa", "#c62828", "#00838f", "#6d4c41"]
    gens = np.arange(curvas_A["C1"].shape[1])

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for c, (n, cur) in zip(colores, curvas_A.items()):
        p = CONFIGS_A[n]
        ax.plot(gens, cur.mean(0), color=c, lw=1.8,
                label=f"{n}: N={p['tam_poblacion']}, pc={p['prob_cruce']}, pm={p['prob_mutacion']}")
    ax.axhline(f_opt, ls="--", color="k", lw=.8, label=f"Óptimo = {f_opt:.6f}")
    ax.set_ylim(2.0, 2.88); ax.set_xlabel("Generación"); ax.set_ylabel("Mejor f(x) (promedio de 10 corridas)")
    ax.set_title("Convergencia promedio por configuración"); ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    fig.savefig(f"{SALIDA}/convergencia_configuraciones.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for c, (n, d) in zip(colores, divs_A.items()):
        ax.plot(gens, d.mean(0), color=c, lw=1.5, label=n)
    ax.set_xlabel("Generación"); ax.set_ylabel("Desv. estándar de x en la población")
    ax.set_title("Diversidad de la población (promedio de 10 corridas)"); ax.grid(alpha=.3)
    ax.legend(fontsize=8, ncol=3); fig.tight_layout()
    fig.savefig(f"{SALIDA}/diversidad_configuraciones.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    datos = [[f["mejor_f"] for f in todas if f["config"] == n] for n in CONFIGS_A]
    ax.boxplot(datos, tick_labels=list(CONFIGS_A), showmeans=True)
    ax.axhline(f_opt, ls="--", color="k", lw=.8)
    ax.set_ylabel("Mejor f(x) final"); ax.set_title("Distribución del mejor valor (10 semillas por configuración)")
    ax.grid(alpha=.3, axis="y"); fig.tight_layout()
    fig.savefig(f"{SALIDA}/boxplot_configuraciones.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for c, (n, cur) in zip(colores, curvas_B.items()):
        ax.plot(gens, cur.mean(0), color=c, lw=1.6, label=n)
    ax.axhline(f_opt, ls="--", color="k", lw=.8)
    ax.set_ylim(2.55, 2.87); ax.set_xlabel("Generación"); ax.set_ylabel("Mejor f(x) (promedio)")
    ax.set_title("Comparación de operadores (N=50, pc=0.8)"); ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
    fig.savefig(f"{SALIDA}/convergencia_operadores.png", dpi=170); plt.close(fig)

    for r in resumenes + res_B:
        print(f"{r['config']:<30} f={r['f_prom']:.6f}±{r['f_desv']:.6f}  gen={r['gen_prom']:.1f}"
              f"  t={r['t_prom_ms']:.1f}ms  div={r['div_final']:.4f}  éxitos={r['exitos']}/10"
              f"  prematuras={r['prematuras']}")


if __name__ == "__main__":
    main()
