# Documentación de Testeo — Modelos Binarios (Informe Final)

Este documento describe **cómo se obtuvieron los resultados de Precision y Recall** reportados en la
tabla de validación de modelos binarios del informe de memoria (Tabla `resumen_validacion_binaria`).

---

## Scripts de testeo utilizados

| Fila en el informe | Notebook de testeo | Sección dentro del notebook |
|---|---|---|
| `1 Cohorte (4D Base)` | `1cohorte_4d.ipynb` | Celda "Sin balancear" y "Balanceados" |
| `1 Cohorte (5D Bal.)` | `1cohorte_5d.ipynb` | Celda "Probar modelos: red sin balancear" y "red balanceada" |
| `2 Cohortes (4D Base / 5D)` | `2_cohortes.ipynb` / `2cohortes_5d.ipynb` | Secciones de evaluación final |
| `3 Cohortes (4D Bal.)` | `3_cohortes.ipynb` | Celda de evaluación con rdim 1-16 |

---

## Datos de testeo utilizados

### Modelos de 1 Cohorte

Los modelos TuckER fueron **entrenados** con la cohorte **2019** (S1→S2) y **evaluados/testeados**
sobre la cohorte **2020**:

| Archivo CSV | Descripción |
|---|---|
| `mis_scripts/dataframes_por_semestre/df_20201.csv` | Input (notas Semestre 1, cohorte 2020) |
| `mis_scripts/dataframes_por_semestre/df_20202.csv` | Target (resultados Semestre 2, cohorte 2020) |
| `mis_scripts/new_df_por_sem/base_puntaje_estudiante.csv` | Puntaje PSU/PAES (solo para modelos 5D) |

> **Nota importante**: aunque en el informe se menciona "Cohorte 2022" como conjunto de prueba
> general, los notebooks de 1 cohorte evalúan sobre la **Cohorte 2020** (primeros alumnos no vistos
> durante el entrenamiento). La evaluación sobre Cohorte 2022 corresponde a los experimentos de
> 3 cohortes (`3_cohortes.ipynb`).

---

## Vocabulario y dataset de entrenamiento TuckER

| Config | Dataset de tripletas (vocabulario) |
|---|---|
| 4D Base | `data/dataset_20192_fundamentales/` |
| 5D Bal. | `data/dataset_20192_fundamentales/` |

Archivos dentro del dataset: `train.txt`, `train_balanceado.txt`, `valid.txt`, `test.txt`

---

## Criterio de evaluación (Precision y Recall)

Las métricas reportadas son **Precision y Recall de la clase "Reprueba" (label=0)**,
calculadas desde la matriz de confusión:

```
Recall_reprueba    = TN / (TN + FP)   ← de todos los que realmente reprobaron, ¿cuántos detectamos?
Precision_reprueba = TN / (TN + FN)   ← de todos los que predijimos como reprobados, ¿cuántos eran reales?
```

**Criterio de predicción**: si `sigmoid(score_aprueba) >= sigmoid(score_reprueba)` → predice Aprueba,
si no → predice Reprueba.

---

## Modelos entrenados utilizados (archivos `.pt`)

> **¿Por qué no están en el repositorio?**
> Los archivos de modelos (`.pt`) están excluidos por `.gitignore` porque son artefactos binarios
> generados durante el entrenamiento. La práctica recomendada en ML es no subirlos a Git sino
> reproducirlos con los scripts de entrenamiento. En este caso son ~12 MB en 317 archivos, por lo que
> en caso de necesitarlos se pueden regenerar ejecutando los notebooks de generación.

### Predictores (redes neuronales) para 1 Cohorte

| Config | Carpeta | Nombre de archivo |
|---|---|---|
| 4D Base (sin balancear) | `predictores_sem2_2019/` | `best_predictor_model_rdim{rdim}_normalizado_2019.pt` |
| 4D Base (balanceado) | `predictores_sem1_balanceados/` | `best_predictor_dim4_rdim{rdim}_2019_balanceado.pt` |
| 5D Standard | `predictores_5d_standard/` | `best_predictor_dim5_rdim{rdim}_standard_5d.pt` |
| 5D Balanceado | `predictores_sem1_balanceados_5d/` | `best_predictor_dim5_rdim{rdim}_2019_balanceado_5d.pt` |

### Modelos TuckER (pesos `best_model.pt`) para 1 Cohorte

Los pesos TuckER se ubican en `results/` (también excluido por `.gitignore`):

| Config | Prefijo de carpeta en `results/` |
|---|---|
| 4D Base | `Experimento_warm_start_rdim{rdim}_1000epochs_earlystopping_2019_patience300/` |
| 4D Balanceado | `Experimento_warm_start_rdim{rdim}_1000epochs_earlystopping_2019patience400_balanceado/` |
| 5D Balanceado | `Experimento_warm_start_rdim{rdim}_1000epochs_earlystopping_2019patience400_balanceado_5d/` |

Dimensiones evaluadas: `rdim` ∈ {1, 2, ..., 16}; `edim` = 4 (4D) o 5 (5D)

---

## Cursos involucrados

```python
CURSOS_PRIMER  = ['MA1101', 'MA1001', 'FI1000', 'BT1211']   # Semestre 1 (input del predictor)
CURSOS_SEGUNDO = ['MA1002', 'MA1102', 'FI1100', 'CC1002']    # Semestre 2 (predicción objetivo)
CURSOS_EVAL    = CURSOS_PRIMER + CURSOS_SEGUNDO               # Todos los cursos evaluados
```

Criterio de selección de alumnos: deben haber inscrito los **4 cursos de CURSOS_PRIMER** en el
Semestre 1 de la cohorte.

---

## Reproducibilidad

Para reproducir los resultados:

1. Ejecutar el notebook de generación de datos y embeddings:
   `notebooks/Experimento_warm_start/1cohorte_4d.ipynb` (sección "Generar dataset") o
   `1cohorte_5d.ipynb` (sección "Generar embeddings 5D")

2. Lanzar entrenamiento TuckER con el script `.bat` incluido dentro del notebook
   (buscar celda con `@echo off`), que usa `main_original_warm_start_gemini_2_earlystopping_gemini.py`

3. Ejecutar las celdas de testeo del mismo notebook (secciones "Probar modelos") sobre los datos
   del semestre 1 y 2 de la cohorte siguiente (2020 para 1 cohorte, 2022 para 3 cohortes)
