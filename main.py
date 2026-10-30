"""
main.py — Menú interactivo del Algoritmo Genético
=================================================
Uso:
    python main.py                      # abre el menú interactivo (lee config.json)
    python main.py --config otra.json   # usa otro archivo de configuración
    python main.py --run                # ejecuta directo con config.json, sin menú

Todos los parámetros se modifican desde el menú o editando config.json;
NO es necesario tocar el código fuente.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # permite guardar gráficas sin entorno gráfico
import matplotlib.pyplot as plt

from ga import (bits_necesarios, ejecutar_ag, optimo_analitico,
                optimo_fuerza_bruta, validar_config, X_MIN, X_MAX, funcion_objetivo)

PARAMETROS_EDITABLES = [
    ("tam_poblacion", int, "Tamaño de población"),
    ("generaciones", int, "Número de generaciones"),
    ("prob_cruce", float, "Probabilidad de cruce (0-1)"),
    ("prob_mutacion", float, "Probabilidad de mutación por gen (0-1)"),
    ("semilla", int, "Semilla aleatoria"),
    ("representacion", str, "Representación [binaria | gray | real]"),
    ("seleccion", str, "Selección [ruleta | torneo]"),
    ("tam_torneo", int, "Tamaño del torneo"),
    ("cruce", str, "Cruce [un_punto | dos_puntos | uniforme | blx | aritmetico]"),
    ("mutacion", str, "Mutación [bit_flip | gaussiana | uniforme | no_uniforme | polinomial]"),
    ("elitismo", int, "Individuos élite"),
]


def cargar_config(ruta):
    with open(ruta, encoding="utf-8") as fh:
        return json.load(fh)


def guardar_config(cfg, ruta):
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    print(f"  Configuración guardada en {ruta}")


def graficar_convergencia(res, cfg, ruta_png, f_opt=None):
    gens = range(len(res.historial_mejor))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gens, res.historial_mejor, lw=2, color="#1f5fa8", label="Mejor aptitud (histórico)")
    ax.plot(gens, res.historial_promedio, lw=1.2, color="#e08a1e", alpha=.85, label="Aptitud promedio")
    if f_opt is not None:
        ax.axhline(f_opt, ls="--", color="#2e7d32", lw=1, label=f"Óptimo (fuerza bruta) = {f_opt:.6f}")
    ax.set_xlabel("Generación")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Convergencia del AG — N={cfg['tam_poblacion']}, pc={cfg['prob_cruce']}, "
                 f"pm={cfg['prob_mutacion']}, {cfg['seleccion']}, {cfg['cruce']}")
    ax.grid(alpha=.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=150)
    plt.close(fig)


def reporte(res, cfg, f_opt=None, x_opt=None):
    print("\n================ REPORTE DEL MEJOR INDIVIDUO ================")
    if cfg["representacion"] != "real":
        print(f"  Cromosoma ({cfg['representacion']}, {len(res.mejor_cromosoma)} bits): "
              f"{''.join(map(str, res.mejor_cromosoma))}")
    print(f"  x*            = {res.mejor_x:.6f}")
    print(f"  f(x*)         = {res.mejor_f:.6f}")
    print(f"  Gen. estable  = {res.generacion_estable} (última mejora: {res.ultima_mejora})")
    print(f"  Evaluaciones  = {res.evaluaciones}")
    print(f"  Tiempo        = {res.tiempo_s:.4f} s")
    if f_opt is not None:
        err = abs(f_opt - res.mejor_f)
        print(f"  Óptimo real   = x={x_opt:.6f}, f={f_opt:.6f}")
        print(f"  Error absoluto= {err:.2e}  ({100*err/f_opt:.5f} %)")
    print("==============================================================\n")


def ejecutar(cfg, salida="resultados"):
    cfg_v = validar_config(cfg)
    print(f"\nEjecutando AG (L = {bits_necesarios(cfg_v['precision_decimales'])} bits si es binaria)…")
    res = ejecutar_ag(cfg_v, verbose=True)
    x_bf, f_bf = optimo_fuerza_bruta()
    os.makedirs(salida, exist_ok=True)
    png = os.path.join(salida, f"convergencia_semilla{cfg_v['semilla']}.png")
    graficar_convergencia(res, cfg_v, png, f_bf)
    reporte(res, cfg_v, f_bf, x_bf)
    print(f"  Gráfica de convergencia: {png}")
    return res


def validar():
    print("\nValidación del óptimo global en [-1, 2]:")
    x_bf, f_bf = optimo_fuerza_bruta()
    print(f"  Fuerza bruta (paso 1e-6, 3 000 001 evaluaciones): x = {x_bf:.6f}, f = {f_bf:.6f}")
    x_an, f_an, d2 = optimo_analitico(x_bf)
    print(f"  Newton sobre f'(x)=0:  x = {x_an:.10f}, f = {f_an:.10f}, f''(x) = {d2:.2f} (<0 => máximo)")


def editar(cfg):
    for i, (clave, _, desc) in enumerate(PARAMETROS_EDITABLES, 1):
        print(f"  {i:2d}. {desc:<62} = {cfg.get(clave)}")
    op = input("  Número del parámetro a modificar (Enter = regresar): ").strip()
    if not op.isdigit() or not 1 <= int(op) <= len(PARAMETROS_EDITABLES):
        return
    clave, tipo, desc = PARAMETROS_EDITABLES[int(op) - 1]
    nuevo = input(f"  Nuevo valor para {desc}: ").strip()
    try:
        cfg[clave] = tipo(nuevo)
        validar_config(cfg)
        print("  ✔ Parámetro actualizado.")
    except (ValueError, TypeError) as e:
        print(f"  ✘ Valor inválido: {e}")


def menu(ruta_cfg):
    cfg = cargar_config(ruta_cfg)
    while True:
        print("\n========== ALGORITMO GENÉTICO — f(x) = x·sin(10πx) + 1 ==========")
        print("  1. Ver configuración actual")
        print("  2. Modificar un parámetro")
        print("  3. Ejecutar el AG")
        print("  4. Validar óptimo (fuerza bruta + analítico)")
        print("  5. Guardar configuración en", ruta_cfg)
        print("  6. Graficar la función objetivo")
        print("  0. Salir")
        op = input("Opción: ").strip()
        if op == "1":
            print(json.dumps(cfg, indent=2, ensure_ascii=False))
        elif op == "2":
            editar(cfg)
        elif op == "3":
            ejecutar(cfg)
        elif op == "4":
            validar()
        elif op == "5":
            guardar_config(cfg, ruta_cfg)
        elif op == "6":
            graficar_funcion()
        elif op == "0":
            break


def graficar_funcion(ruta="resultados/funcion_objetivo.png"):
    import numpy as np
    x = np.linspace(X_MIN, X_MAX, 4000)
    y = x * np.sin(10 * np.pi * x) + 1
    x_bf, f_bf = optimo_fuerza_bruta()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(x, y, color="#1f5fa8", lw=1.2)
    ax.plot([x_bf], [f_bf], "o", color="#c62828", ms=7, label=f"Óptimo global ({x_bf:.4f}, {f_bf:.4f})")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.set_title("f(x) = x·sin(10πx) + 1 en [-1, 2]")
    ax.grid(alpha=.3); ax.legend(loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"  Gráfica guardada en {ruta}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AG para maximizar x·sin(10πx)+1")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--run", action="store_true", help="ejecutar sin menú")
    args = ap.parse_args()
    if args.run:
        ejecutar(cargar_config(args.config))
    else:
        menu(args.config)
