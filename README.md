# 🎲 Bondad de Ajuste χ² con Computer Vision (`supervision` + OpenCV)

Pipeline *end-to-end* que conecta **Computer Vision en tiempo real** con **Inferencia Estadística paramétrica y no paramétrica** (Análisis Estadístico II). El sistema detecta y contabiliza tiradas de 5 dados mediante cámara o simulación sintética utilizando [`roboflow/supervision`](https://github.com/roboflow/supervision), almacena las observaciones en una base de datos transaccional **SQLite** y ejecuta pruebas rigurosas de **Bondad de Ajuste Chi-Cuadrado (χ²)** con nivel de significación **α = 0.05**.

---

## 📊 Dashboard de Resultados (n = 422 tiradas)

![Dashboard de Bondad de Ajuste](assets/dashboard_bondad_ajuste.jpg)

---

## 🎯 Objetivo del Proyecto

Demostrar empíricamente dos pilares fundamentales de la teoría de probabilidades y la estadística inferencial a partir de datos capturados por visión artificial:

1. **Prueba 1 — Ley Uniforme Discreta (Dado Individual d₁):**
   Verificar si un dado individual es equilibrado (*justo*), contrastando las frecuencias observadas frente a una distribución uniforme discreta teórica donde no se estiman parámetros muestrales (**k = 0**).

2. **Prueba 2 — Aproximación Normal vía Teorema Central del Límite (Suma de 5 Dados):**
   Comprobar que la suma de 5 variables aleatorias independientes e idénticamente distribuidas converge a una distribución **Normal(μ̂, σ̂)**, estimando sus parámetros a partir de datos agrupados (**k = 2**) y aplicando el criterio estricto de agrupación de colas cuando **Eᵢ < 5**.

---

## 👁️ Sistema Dual de Computer Vision (`roboflow/supervision`)

El motor visual procesa el stream de video en vivo (cámara cenital / Iriun Webcam) o alterna en caliente con un generador sintético de alta precisión. Todas las detecciones se empaquetan en estructuras nativas `sv.Detections` y se renderizan mediante `sv.RoundBoxAnnotator`, `sv.LabelAnnotator` y telemetría HUD en tiempo real.

| Captura Real con Cámara Cenital (`Mode: CV`) | Simulador Sintético Integrado (`Mode: Sim`) |
| :---: | :---: |
| ![Detección Cámara Real](assets/deteccion_camara_real.jpg) | ![Detección Modo Sintético](assets/deteccion_modo_sintetico.jpg) |
| *Aislamiento de ROIs, filtrado morfológico y conteo de pips en la cara superior.* | *Generación procedural y validación automática del pipeline con `supervision`.* |

---

## 📐 Metodología Estadística y Rigor Académico

Para un tamaño muestral consolidado de **n = 422 tiradas** (**α = 0.05**, confianza acumulada del **95%**), el motor estadístico ejecuta el siguiente procedimiento formal:

### 1. Prueba de Uniformidad Discreta (Dado d₁)

* **Hipótesis:**
  * **H₀:** El dado d₁ es justo (**Pₑ = 1/6** para cada cara **xᵢ ∈ {1, 2, 3, 4, 5, 6}**).
  * **H₁:** El dado d₁ presenta sesgo (**Pₑ ≠ 1/6** en al menos una cara).
* **Frecuencia Esperada:** `Eᵢ = n × Pₑ = 422 × (1/6) = 70.33` por categoría (**Eᵢ ≥ 5** en todas las clases).
* **Grados de Libertad:** `gl = r - k - 1 = 6 - 0 - 1 = 5`.
* **Resultado Experimental:**
  * **χ² calculado:** `5.9621`
  * **χ² crítico (0.95; gl = 5):** `11.0705`
  * **p-valor:** `0.3099 ≥ 0.05`

> **Conclusión Prueba 1:** Como `χ²_calc (5.9621) ≤ χ²_crit (11.0705)`, **NO SE RECHAZA H₀**. No existe evidencia estadística significativa para afirmar que el dado d₁ esté sesgado.

---

### 2. Prueba de Normalidad Continua por TCL (Suma de 5 Dados)

* **Hipótesis:**
  * **H₀:** La suma de los 5 dados (**S = d₁ + d₂ + d₃ + d₄ + d₅**) se ajusta a una distribución **Normal(μ̂, σ̂)**.
  * **H₁:** La suma de los 5 dados no sigue una distribución Normal.
* **Construcción de Intervalos y Estimación en Datos Agrupados (k = 2):**
  * Se construyen intervalos semiabiertos **[L_inf, L_sup)** de amplitud entera redondeada hacia arriba **h = 4**.
  * A partir de las marcas de clase (**xᵢ**) y frecuencias observadas (**Oᵢ**), se estiman los parámetros poblacionales de datos agrupados:
    * **Media estimada (μ̂):** `Σ(Oᵢ · xᵢ) / n = 18.1374`
    * **Desvío estándar estimado (σ̂):** `√[ Σ Oᵢ · (xᵢ - μ̂)² / (n - 1) ] = 3.7996`
* **Probabilidades Esperadas Exactas y Pooling (Eᵢ ≥ 5):**
  * Se evalúa la función de distribución acumulada normal estándar (**Φ**) en los límites exactos de cada intervalo (sin extender colas a infinito):
    * `Pₑ = Φ(Z_sup) - Φ(Z_inf)`  donde  `Z = (L - μ̂) / σ̂`
    * `Eᵢ = n · Pₑ`
  * Los intervalos extremos con frecuencias esperadas menores a 5 se agrupan automáticamente hacia adentro hasta cumplir la condición de validez de Pearson (**Eᵢ ≥ 5**), resultando en **r = 5 categorías finales**: `[7, 11)`, `[11, 15)`, `[15, 19)`, `[19, 23)` y `[23, 27)`.
* **Grados de Libertad:** Al haber estimado **μ̂** y **σ̂** a partir de la muestra (**k = 2**), se descuentan dos grados de libertad adicionales:
  * `gl = r - k - 1 = 5 - 2 - 1 = 2`
* **Resultado Experimental:**
  * **χ² calculado:** `1.4864`
  * **χ² crítico (0.95; gl = 2):** `5.9915`
  * **p-valor:** `0.4756 ≥ 0.05`

> **Conclusión Prueba 2:** Como `χ²_calc (1.4864) ≤ χ²_crit (5.9915)`, **NO SE RECHAZA H₀**. La distribución empírica de la suma de los 5 dados es estadísticamente compatible con una ley **Normal(μ̂ = 18.14, σ̂ = 3.80)**, validando en la práctica el Teorema Central del Límite.

---

## 🚀 Instalación y Uso Rápido

### 1. Clonar el repositorio e instalar dependencias

```bash
git clone https://github.com/matiasbellidor/bondad-ajuste-dado.git
cd bondad-ajuste-dado
python -m venv .venv

# Activar en Windows:
.\.venv\Scripts\activate
# Activar en Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Ejecutar el pipeline interactivo

```bash
# Modo OpenCV con cámara principal (o simulador automático si no hay cámara)
python main.py

# Especificar índice de cámara externa (ej. Iriun Webcam / DroidCam)
python main.py --camera 1
```

### 3. Controles en tiempo real (Ventana de Supervisión)

| Tecla | Acción en el Pipeline |
| :---: | :--- |
| `ESPACIO` | Registra la tirada actual de 5 dados en SQLite (`dice_experiments.db`) y exporta el frame anotado en `linkedin_captures/`. |
| `T` | Alterna en vivo entre la cámara física (`Mode: CV`) y el generador sintético (`Mode: Sim`). |
| `S` | **Fast-Forward Seed:** Inyecta tiradas simuladas en SQLite y guarda capturas de muestra anotadas. |
| `A` | Cierra la captura de video, ejecuta el motor estadístico χ², imprime las tablas completas en consola y genera `dashboard_bondad_ajuste.png`. |
| `Q` / `ESC` | Cierra la aplicación de forma segura. |

---

## 📁 Estructura del Repositorio

```text
bondad-ajuste-dado/
├── assets/
│   ├── dashboard_bondad_ajuste.jpg      # Gráfico principal con resultados (n = 422)
│   ├── deteccion_camara_real.jpg        # Captura del detector en cámara cenital
│   └── deteccion_modo_sintetico.jpg     # Captura del detector en modo simulador
├── main.py                              # Código fuente completo (CV + SQLite + Motor Chi2)
├── requirements.txt                     # Dependencias del proyecto
└── README.md                            # Documentación técnica y estadística
```

---

## 👤 Autor

**Matías Bellido**  
Proyecto desarrollado para la cátedra de **Análisis Estadístico II**, integrando herramientas modernas de Visión por Computadora con pruebas de hipótesis de Bondad de Ajuste.
