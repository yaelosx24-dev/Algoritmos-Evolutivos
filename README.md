# Algoritmo Genético — Maximización de f(x) = x·sin(10πx) + 1 en [-1, 2]

Proyecto de la **Evaluación del 2.º parcial de Algoritmos Evolutivos** (I.S.C., TESI).
Actividad 3: implementación práctica de un AG. Actividad 4: experimentación y convergencia.

## Estructura

| Archivo | Contenido |
|---|---|
| `ga.py` | Núcleo del AG: representación (binaria, Gray, real), selección (ruleta, torneo), cruce (un punto, dos puntos, uniforme, BLX-α, aritmético), mutación (bit-flip, gaussiana, uniforme, no uniforme, polinomial), elitismo y validación (fuerza bruta + Newton). |
| `main.py` | Menú interactivo, lectura de `config.json`, gráfica de convergencia y reporte del mejor individuo. |
| `config.json` | Parámetros configurables sin editar el código. |
| `experimentos.py` | Diseño experimental: 6 configuraciones × 10 semillas + comparación de operadores. Genera CSV y gráficas en `resultados/`. |

## Instalación y uso

```bash
pip install -r requirements.txt
python main.py            # menú interactivo
python main.py --run      # ejecución directa con config.json
python experimentos.py    # reproduce todas las tablas y gráficas del reporte
```

## Parámetros de `config.json`

| Clave | Significado | Valores |
|---|---|---|
| `tam_poblacion` | Tamaño de la población N | entero ≥ 2 |
| `generaciones` | Número de generaciones T | entero ≥ 1 |
| `prob_cruce` | Probabilidad de cruce pc | 0–1 |
| `prob_mutacion` | Probabilidad de mutación por gen pm | 0–1 |
| `semilla` | Semilla del generador aleatorio | entero |
| `representacion` | Codificación | `binaria`, `gray`, `real` |
| `seleccion` | Operador de selección | `ruleta`, `torneo` |
| `tam_torneo` | Competidores por torneo | entero ≥ 2 |
| `cruce` | Operador de cruce | binaria: `un_punto`, `dos_puntos`, `uniforme`; real: `blx`, `aritmetico` |
| `mutacion` | Operador de mutación | binaria: `bit_flip`; real: `gaussiana`, `uniforme`, `no_uniforme`, `polinomial` |
| `elitismo` | Individuos que pasan intactos | entero ≥ 0 |
| `precision_decimales` | Decimales de precisión (define L) | 6 → 22 bits |

## Resultado de referencia

- Óptimo por fuerza bruta (paso 10⁻⁶): **x = 1.850547, f = 2.850274**
- Newton sobre f'(x) = 0: x = 1.8505474661, f = 2.8502737668, f''(x) = −1827.23 < 0 (máximo)
- AG con `config.json` por omisión (semilla 42): x = 1.850548, f = 2.850274, error 5.6 × 10⁻¹¹

## Trabajo colaborativo

Cada integrante debe hacer *commits* de la parte que le corresponde (por ejemplo:
operadores de selección/cruce, mutación y validación, menú y configuración,
experimentos y gráficas) para que el historial evidencie la colaboración.
