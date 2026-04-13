# Documentación de Experimentos

Este documento describe cómo se obtuvo cada resultado experimental, qué scripts
se ejecutaron, en qué orden y con qué parámetros.

---

## Índice

1. [Preparación de datos (S3 binario)](#1-preparación-de-datos-s3-binario)
2. [Generación de embeddings 8D](#2-generación-de-embeddings-8d)
3. [Entrenamiento de TuckER (relation ranking)](#3-entrenamiento-de-tucker-relation-ranking)
4. [Validación inductiva de la cohorte OOT](#4-validación-inductiva-de-la-cohorte-oot)
5. [Análisis de drift de embeddings](#5-análisis-de-drift-de-embeddings)
6. [Comparación de proyecciones (relation ranking)](#6-comparación-de-proyecciones-relation-ranking)
7. [Entrenamiento de TuckER (entity ranking)](#7-entrenamiento-de-tucker-entity-ranking)
8. [Comparación de proyecciones (entity ranking)](#8-comparación-de-proyecciones-entity-ranking)
9. [XGBoost: validación temporal S3](#9-xgboost-validación-temporal-s3)
10. [XGBoost: tuning con grid search S3](#10-xgboost-tuning-con-grid-search-s3)
11. [XGBoost: walk-forward S3](#11-xgboost-walk-forward-s3)
12. [XGBoost: validación y tuning S2](#12-xgboost-validación-y-tuning-s2)
13. [Comparación de modelos de árboles](#13-comparación-de-modelos-de-árboles)
14. [Ablación de flags de missing (S2)](#14-ablación-de-flags-de-missing-s2)
15. [Análisis de falsos positivos](#15-análisis-de-falsos-positivos)

---

## 1. Preparación de datos (S3 binario)

**Script:** `scripts/generar_tripletas_tercer_semestre.py`

**Propósito:** Generar tripletas `(alumno, relación, curso)` con relaciones
binarias `{aprueba, reprueba}` para la predicción de tercer semestre.

**Datos de entrada:**
- CSVs en `data/dataframes_por_semestre/df_XXXXX.csv` (separador `;`)

**Cohortes usadas:**
| Semestre de entrada (S1) | Semestre objetivo (S3) |
|---|---|
| 20191 | 20201 |
| 20201 | 20211 |
| 20211 | 20221 |

**Criterios de elegibilidad:**
1. El alumno debe haber cursado los **4 cursos fundamentales de S1**: `{MA1101, MA1001, FI1000, BT1211}`
2. El alumno debe tener al menos 1 curso de S1/S2/S3 en el semestre objetivo

**Cursos incluidos en las tripletas:**
- S1: `{MA1101, MA1001, FI1000, BT1211}`
- S2: `{MA1002, MA1102, FI1100, CC1002}`
- S3: `{MA2001, MA2601, FI2001, FI2003, IQ2211}`

**Regla de clasificación:**
- `Aprobado` → `aprueba`
- `Reprobado` o `Eliminado` → `reprueba`
- Si no hay estado claro: nota ≥ 4.0 → `aprueba`, nota < 4.0 → `reprueba`

**Split:** 80/10/10 **por alumno** (no por tripleta), semilla `SEED=42`.
Se genera también un `train_balanceado.txt` con sobremuestreo de `reprueba`.

**Ejecución:**
```bash
cd tucker_pull
python scripts/generar_tripletas_tercer_semestre.py
```

**Salida:** `data/dataset_2019_2020_2021_hacia_s3_binario/`
- `train.txt`, `valid.txt`, `test.txt`
- `train_balanceado.txt`
- `ids_train.txt`, `ids_valid.txt`, `ids_test.txt`

---

## 2. Generación de embeddings 8D

**Script:** `scripts/generar_embeddings_8d_s3.py`

**Propósito:** Crear embeddings iniciales de 8 dimensiones para cada alumno,
basados en sus notas de S1 y S2, para hacer warm start en TuckER.

**Vector de 8 dimensiones (orden):**
```
[MA1101, MA1001, FI1000, BT1211, MA1002, MA1102, FI1100, CC1002]
  dim 0   dim 1   dim 2   dim 3   dim 4   dim 5   dim 6   dim 7
  ─── S1 ───────────────────────  ─── S2 ───────────────────────
```

**Reglas para la nota efectiva:**
- Reintento del curso: se toma la nota **máxima** obtenida
- `Aprobado (T)` (transferencia): se asigna `4.0`
- No cursó el ramo en S2: se asigna `1.0` (`FAIL_GRADE`)

**Normalización:** `(nota - 4.0) / 3.0` → mapea `[1, 7]` a `[-1, 1]`

**Ejecución:**
```bash
python scripts/generar_embeddings_8d_s3.py
```

**Salida:** `notebooks/Experimento_warm_start/embeddings_8d_binario_s3/`
- `embeddings_inicializados_binarios_8d.pt` — tensor `(N_entidades, 8)`
- `vocabulario_binario_8d.json` — mapeo de nombres a índices

---

## 3. Entrenamiento de TuckER (relation ranking)

**Script:** `scripts/main/main_original_warm_start_gemini_2_earlystopping_gemini_definitivo.py`

**Propósito:** Entrenar TuckER con warm start (embeddings de notas) y early
stopping usando **ranking de relaciones** como métrica de validación.

**Evaluación (relation ranking):** Para cada par `(alumno, curso)`, se rankean
solo las relaciones forward `{aprueba, reprueba}` (excluyendo `_reverse`).
Se calcula el score para cada relación `r` como:
`s = e_h · M_r · e_t`, donde `M_r = W ×_1 w_r`.

**Parámetros de ejecución:**
```
--dataset     dataset_2019_2020_2021_hacia_s3_binario
--edim        8
--rdim        11
--lr          0.003
--batch_size  128
--num_iterations 1000
--patience    400
--input_dropout    0.3
--hidden_dropout1  0.4
--hidden_dropout2  0.5
--label_smoothing  0.1
--dr          1.0            # sin decay
--cuda        True
--init_embeddings  notebooks/Experimento_warm_start/embeddings_8d_binario_s3/embeddings_inicializados_binarios_8d.pt
--init_vocab       notebooks/Experimento_warm_start/embeddings_8d_binario_s3/vocabulario_binario_8d.json
--output_prefix    prueba_s3_8d_rdim11
```

**Semilla:** `seed=20` (hardcoded en el script)

**Ejecución:**
```bash
cd tucker_pull
python scripts/main/main_original_warm_start_gemini_2_earlystopping_gemini_definitivo.py \
  --dataset dataset_2019_2020_2021_hacia_s3_binario \
  --edim 8 --rdim 11 --lr 0.003 --batch_size 128 \
  --num_iterations 1000 --patience 400 \
  --input_dropout 0.3 --hidden_dropout1 0.4 --hidden_dropout2 0.5 \
  --label_smoothing 0.1 --dr 1.0 --cuda True \
  --init_embeddings notebooks/Experimento_warm_start/embeddings_8d_binario_s3/embeddings_inicializados_binarios_8d.pt \
  --init_vocab notebooks/Experimento_warm_start/embeddings_8d_binario_s3/vocabulario_binario_8d.json \
  --output_prefix prueba_s3_8d_rdim11
```

**Resultado:** Best epoch **92**, val_MRR = **0.6604**. Early stopping en epoch 492.

**Salida:** `results/prueba_s3_8d_rdim11/`
- `best_model.pt` — state dict del modelo en el mejor epoch
- `training_metrics.csv` — loss y métricas (val/test) por época
- `embeddings_history.csv` — snapshots de embeddings de 10 alumnos muestra

---

## 4. Validación inductiva de la cohorte OOT

**Script:** `scripts/validar_cohorte_nueva_s3.py`

**Propósito:** Evaluar el modelo entrenado sobre alumnos de la cohorte 2022
que **nunca** estuvieron en el entrenamiento de TuckER. Usa scoring manual
(no model.forward) porque los alumnos OOT no están en el vocabulario.

**Cohorte OOT:**
- S1 (para embeddings): semestre `20221`
- S2 (para embeddings): semestre `20222`
- S3 (para evaluación): semestre `20231`
- Resultado: 811 alumnos, 3164 tripletas

**Pipeline interno:**
1. Construye embeddings 8D para cada alumno OOT (misma normalización)
2. Carga `best_model.pt` y extrae `W`, `BN0`, `BN1`, `E.weight`
3. Para cada `(alumno*, curso)`:
   - Aplica `BN0(embedding)`, contrae con `W_r`, aplica `BN1`, dot con `e_curso`
4. Evalúa por ranking de relaciones (same as training)

**Ejecución:**
```bash
python scripts/validar_cohorte_nueva_s3.py
```

**Salida:** `results/validacion_cohorte_20221_20231/`
- `metricas_cohorte_20221_20231.json`
- `predicciones_por_tripleta.csv`

---

## 5. Análisis de drift de embeddings

**Script:** `scripts/_analizar_drift.py`

**Propósito:** Comparar embeddings iniciales (notas normalizadas) vs. aprendidos
por TuckER para cuantificar cuánto modifica el modelo las representaciones.

**Ejecución:**
```bash
python scripts/_analizar_drift.py
```

**Resultado (relation ranking model):**
| Estadístico | Valor |
|---|---|
| Norma L2 media del drift | 1.5153 |
| Norma L2 media del vector inicial | 1.7483 |
| Norma L2 media del vector aprendido | 2.4478 |
| Ratio drift/inicial | 0.8667 |

---

## 6. Comparación de proyecciones (relation ranking)

**Script:** `scripts/comparar_proyecciones_extendido.py`

**Propósito:** Evaluar 12 métodos de proyección para mapear notas 8D de
alumnos OOT al espacio latente de TuckER, usando **ranking de relaciones**.

**Constantes del script:**
```python
EDIM, RDIM = 8, 11
SEED = 42
SEM_S1_OOT, SEM_S2_OOT, SEM_OBJ_OOT = "20221", "20222", "20231"
MODEL_PATH = "results/prueba_s3_8d_rdim11/best_model.pt"
```

**Métodos evaluados:**
1. Identidad (sin proyección)
2. Lineal (MSE)
3. MLP 8→64→128→8 (MSE)
4. MLP profundo 8→64→128→64→8 (MSE)
5. Residual MLP (MSE): `f(x) = x + g(x)`
6. MLP 8→64→128→8 (coseno)
7. Residual + LayerNorm (MSE)
8. KNN ponderado k=5 (L2)
9. KNN ponderado k=10 (L2)
10. KNN ponderado k=20 (L2)
11. KNN ponderado k=10 (coseno)
12. KNN ponderado k=20 (coseno)

Los métodos con optimización usan Adam lr=1e-3 y early stopping (paciencia 200).

**Scoring (relation ranking):** Para cada `(alumno*, curso)`, se calcula
el score para `aprueba` y `reprueba`, la relación con mayor score es la
predicción. Métricas: Hits@1, MRR, Accuracy, Precision/Recall/F1 de `reprueba`.

**Ejecución:**
```bash
python scripts/comparar_proyecciones_extendido.py
```

**Resultado:** MLP coseno gana con Hits@1=0.4889, MRR=0.7445.

**Salida:** `results/comparacion_proyecciones_8d/`
- `comparacion_extendida.csv` — tabla de resultados de los 12 métodos

---

## 7. Entrenamiento de TuckER (entity ranking)

**Script:** `scripts/main/main_entity_ranking_definitivo.py`

**Propósito:** Reentrenar TuckER con idéntica configuración pero usando
**entity ranking filtrado** (protocolo estándar de KGE) como criterio de
validación y early stopping.

**Evaluación (entity ranking):** Para cada `(e1, r)`, se rankean **todas
las 2262 entidades** como posible cola. Se aplica filtered setting: se
ponen a 0 las predicciones de colas válidas conocidas (de train+valid+test)
excepto la target, y se busca la posición del target en el ranking.

**Parámetros:** Idénticos al entrenamiento con relation ranking (sección 3),
excepto `--output_prefix prueba_s3_8d_rdim11_entity_ranking`.

**Ejecución:**
```bash
python scripts/main/main_entity_ranking_definitivo.py \
  --dataset dataset_2019_2020_2021_hacia_s3_binario \
  --edim 8 --rdim 11 --lr 0.003 --batch_size 128 \
  --num_iterations 1000 --patience 400 \
  --input_dropout 0.3 --hidden_dropout1 0.4 --hidden_dropout2 0.5 \
  --label_smoothing 0.1 --dr 1.0 --cuda True \
  --init_embeddings notebooks/Experimento_warm_start/embeddings_8d_binario_s3/embeddings_inicializados_binarios_8d.pt \
  --init_vocab notebooks/Experimento_warm_start/embeddings_8d_binario_s3/vocabulario_binario_8d.json \
  --output_prefix prueba_s3_8d_rdim11_entity_ranking
```

**Resultado:** Best epoch **258**, val_MRR = **0.4444**. Early stopping en epoch 658.

**Drift del modelo entity ranking:**
| Estadístico | Valor |
|---|---|
| Norma L2 media del drift | 1.7873 |
| Ratio drift/inicial | 1.0223 |

**Salida:** `results/prueba_s3_8d_rdim11_entity_ranking/`
- `best_model.pt`, `training_metrics.csv`, `embeddings_history.csv`

---

## 8. Comparación de proyecciones (entity ranking)

**Script:** `scripts/comparar_proyecciones_entity_ranking.py`

**Propósito:** Evaluar 10 métodos de proyección usando **entity ranking
filtrado** sobre la cohorte OOT.

**Constantes del script:**
```python
EDIM, RDIM = 8, 11
SEED = 42
MODEL_PATH = "results/prueba_s3_8d_rdim11_entity_ranking/best_model.pt"
```

**Métodos evaluados:**
1. Identidad
2. MLP MSE (64-128)
3. Residual MSE (64-128)
4. MLP coseno (64-128)
5. Residual + LayerNorm (MSE)
6. KNN k=5 (L2)
7. KNN k=10 (L2)
8. KNN k=20 (L2)
9. KNN k=10 (coseno)
10. KNN k=20 (coseno)

**Scoring (entity ranking):** Para cada `(alumno*, r)`, se replica
`model.forward()` manualmente: `BN0 → W×r → BN1 → dot(all entities) → sigmoid`.
Se rankean las 2262 entidades, filtered setting (zerean tails válidas excepto target).

**Ejecución:**
```bash
python scripts/comparar_proyecciones_entity_ranking.py
```

**Resultado:** 9/10 métodos convergen a Hits@1=0.4972. Solo identidad
(0.4608) y MLP coseno (0.4883) difieren.

**Conclusión:** Entity ranking satura porque distinguir 13 cursos de 2249
alumnos es trivial; relation ranking discrimina mejor la calidad del proyector.

**Salida:** `results/comparacion_proyecciones_8d_entity_ranking/`
- `comparacion_entity_ranking.csv`

---

## 9. XGBoost: validación temporal S3

**Script:** `scripts/validacion_temporal.py`

**Propósito:** Baseline de XGBoost para predecir reprobación en S3.
Dos modos: (1) mezcla aleatoria 80/20 con todas las generaciones, y
(2) out-of-time: entrena con 2019-2022, testea con 2023.

**Features:** Notas de S1 + S2 (8 cursos) como valores numéricos.
Misma regla: `FAIL_GRADE=1.0` para no cursado, `APPROVED_TRANSFER=4.0`.

**Configuración XGBoost baseline:**
```python
n_estimators=150, max_depth=4, learning_rate=0.05,
scale_pos_weight=auto, random_state=42
```

**Generaciones:**
- Train: `[20191, 20201, 20211, 20221]`
- Test OOT: `20231`

**Ejecución:**
```bash
python scripts/validacion_temporal.py
```

**Salida:** `results/validacion_temporal.csv`

---

## 10. XGBoost: tuning con grid search S3

**Script:** `scripts/tuning_temporal.py`

**Propósito:** Grid search de hiperparámetros de XGBoost para S3
con configuraciones predefinidas (handcrafted grid).

**Grid de búsqueda (9 configuraciones):** baseline, conservador_d3,
regularizado_d3, profundo_d5, profundo_regularizado, agresivo,
conservador_deep, minimalista, balanced.

**Parámetros variados:**
- `n_estimators`: 100-500
- `max_depth`: 2-5
- `learning_rate`: 0.01-0.1
- `min_child_weight`: 1-5
- `subsample`: 0.7-1.0
- `colsample_bytree`: 0.7-1.0
- `gamma`: 0.0-1.0
- `reg_lambda`: 1.0-5.0
- `reg_alpha`: 0.0-0.5

**Ejecución:**
```bash
python scripts/tuning_temporal.py
```

**Salida:**
- `results/tuning_temporal_todos.csv` — todas las corridas
- `results/tuning_temporal_mejores_oot.csv` — mejores por F1(R) en test 2023
- `results/tuning_temporal_mejores_validacion.csv`

---

## 11. XGBoost: walk-forward S3

**Script:** `scripts/tuning_temporal_walkforward.py`

**Propósito:** Validación walk-forward estricta: entrena con 2019-2021,
valida con 2022 (selección de umbral), testea con 2023.

**Esquema temporal:**
```
Train: [20191, 20201, 20211]
Valid: 20221  → selección de umbral
Test:  20231  → evaluación final
```

**Configuración base:** misma que baseline + variantes del grid.
Se evalúa por curso individual en S3.

**Ejecución:**
```bash
python scripts/tuning_temporal_walkforward.py
```

**Salida:**
- `results/tuning_walkforward_todos.csv`
- `results/tuning_walkforward_mejores_2023.csv` — mejores en test OOT
- `results/tuning_walkforward_mejores_validacion.csv`
- `results/tuning_walkforward_resumen.csv`

---

## 12. XGBoost: validación y tuning S2

**Scripts:**
- `scripts/validacion_temporal_s2.py` — baseline XGBoost para S2 (solo features de S1)
- `scripts/tuning_temporal_s2.py` — grid search para S2
- `scripts/tuning_temporal_walkforward_s2.py` — walk-forward para S2

**Propósito:** Misma estructura que S3 pero prediciendo reprobación en
**segundo semestre** usando solo notas de S1 (4 features).

**Features:** Solo notas de S1: `[MA1101, MA1001, FI1000, BT1211]`

**Ejecución:**
```bash
python scripts/validacion_temporal_s2.py
python scripts/tuning_temporal_s2.py
python scripts/tuning_temporal_walkforward_s2.py
```

**Salida:**
- `results/validacion_temporal_s2.csv`
- `results/tuning_temporal_s2_todos.csv`, `tuning_temporal_s2_mejores_oot.csv`
- `results/tuning_walkforward_s2_todos.csv`, `tuning_walkforward_s2_mejores_2023.csv`

---

## 13. Comparación de modelos de árboles

**Script:** `scripts/gemini_tree.py`

**Propósito:** Comparar Random Forest, XGBoost y LightGBM en la misma
tarea de predicción de reprobación en S3.

**Modelos comparados:**
1. Random Forest (sklearn)
2. XGBoost (xgboost)
3. LightGBM (lightgbm)

**Ejecución:**
```bash
python scripts/gemini_tree.py
```

**Salida:**
- `results/gemini_tree_model_comparison.csv`
- `results/gemini_tree_summary.csv`
- `results/gemini_tree_tercer_semestre.csv`

---

## 14. Ablación de flags de missing (S2)

**Script:** `scripts/compare_missing_flags_s2.py`

**Propósito:** Evaluar si agregar flags binarias indicando qué cursos de S2
no fueron cursados mejora la predicción. Compara features "solo notas" vs.
"notas + flags de missing".

**Ejecución:**
```bash
python scripts/compare_missing_flags_s2.py
```

**Salida:**
- `results/ablation_missing_flags_s2_comparison.csv`
- `results/ablation_missing_flags_s2_raw.csv`

---

## 15. Análisis de falsos positivos

**Scripts:**
- `scripts/analyze_false_positives_walkforward.py` — FP por curso en S3
- `scripts/analyze_false_positives_walkforward_s2.py` — FP por curso en S2

**Propósito:** Desglosar los falsos positivos (alumnos predichos como
reprobados que realmente aprueban) por curso, para identificar en qué
cursos el modelo falla más.

**Ejecución:**
```bash
python scripts/analyze_false_positives_walkforward.py
python scripts/analyze_false_positives_walkforward_s2.py
```

**Salida:**
- `results/falsos_positivos_tercer_semestre.csv`
- `results/tuning_walkforward_falsos_positivos_*.csv`
- `results/tuning_walkforward_s2_falsos_positivos_*.csv`

---

## Estructura de archivos de resultados

```
results/
├── prueba_s3_8d_rdim11/                    # TuckER relation ranking
│   ├── best_model.pt
│   ├── training_metrics.csv
│   └── embeddings_history.csv
├── prueba_s3_8d_rdim11_entity_ranking/     # TuckER entity ranking
│   ├── best_model.pt
│   ├── training_metrics.csv
│   └── embeddings_history.csv
├── validacion_cohorte_20221_20231/         # Validación inductiva OOT
│   ├── metricas_cohorte_20221_20231.json
│   └── predicciones_por_tripleta.csv
├── comparacion_proyecciones_8d/            # Proyecciones relation ranking
│   ├── comparacion_extendida.csv
│   └── comparacion_proyecciones.csv
├── comparacion_proyecciones_8d_entity_ranking/  # Proyecciones entity ranking
│   └── comparacion_entity_ranking.csv
├── validacion_temporal.csv                 # XGBoost baseline S3
├── validacion_temporal_s2.csv              # XGBoost baseline S2
├── tuning_temporal_todos.csv               # Grid search S3 (all)
├── tuning_temporal_mejores_oot.csv         # Grid search S3 (best)
├── tuning_walkforward_todos.csv            # Walk-forward S3 (all)
├── tuning_walkforward_mejores_2023.csv     # Walk-forward S3 (best)
├── tuning_walkforward_s2_todos.csv         # Walk-forward S2 (all)
├── tuning_walkforward_s2_mejores_2023.csv  # Walk-forward S2 (best)
├── gemini_tree_model_comparison.csv        # RF vs XGB vs LGBM
└── ablation_missing_flags_s2_comparison.csv # Ablación flags missing
```

---

## Orden de ejecución recomendado

```
# 1. Datos y embeddings
python scripts/generar_tripletas_tercer_semestre.py
python scripts/generar_embeddings_8d_s3.py

# 2. TuckER con relation ranking
python scripts/main/main_original_warm_start_gemini_2_earlystopping_gemini_definitivo.py \
  --dataset dataset_2019_2020_2021_hacia_s3_binario \
  --edim 8 --rdim 11 --lr 0.003 --batch_size 128 \
  --num_iterations 1000 --patience 400 \
  --input_dropout 0.3 --hidden_dropout1 0.4 --hidden_dropout2 0.5 \
  --label_smoothing 0.1 --output_prefix prueba_s3_8d_rdim11 \
  --init_embeddings notebooks/Experimento_warm_start/embeddings_8d_binario_s3/embeddings_inicializados_binarios_8d.pt \
  --init_vocab notebooks/Experimento_warm_start/embeddings_8d_binario_s3/vocabulario_binario_8d.json

# 3. Validación inductiva
python scripts/validar_cohorte_nueva_s3.py

# 4. Drift
python scripts/_analizar_drift.py

# 5. Comparación de proyecciones (relation ranking)
python scripts/comparar_proyecciones_extendido.py

# 6. TuckER con entity ranking
python scripts/main/main_entity_ranking_definitivo.py \
  --dataset dataset_2019_2020_2021_hacia_s3_binario \
  --edim 8 --rdim 11 --lr 0.003 --batch_size 128 \
  --num_iterations 1000 --patience 400 \
  --input_dropout 0.3 --hidden_dropout1 0.4 --hidden_dropout2 0.5 \
  --label_smoothing 0.1 --output_prefix prueba_s3_8d_rdim11_entity_ranking \
  --init_embeddings notebooks/Experimento_warm_start/embeddings_8d_binario_s3/embeddings_inicializados_binarios_8d.pt \
  --init_vocab notebooks/Experimento_warm_start/embeddings_8d_binario_s3/vocabulario_binario_8d.json

# 7. Comparación de proyecciones (entity ranking)
python scripts/comparar_proyecciones_entity_ranking.py

# 8. XGBoost baselines y tuning
python scripts/validacion_temporal.py
python scripts/tuning_temporal.py
python scripts/tuning_temporal_walkforward.py
python scripts/validacion_temporal_s2.py
python scripts/tuning_temporal_s2.py
python scripts/tuning_temporal_walkforward_s2.py

# 9. Comparación de modelos y análisis
python scripts/gemini_tree.py
python scripts/compare_missing_flags_s2.py
python scripts/analyze_false_positives_walkforward.py
python scripts/analyze_false_positives_walkforward_s2.py
```

---

## Entorno

- **Python:** 3.x (conda env `tucker-run`)
- **PyTorch:** 2.2.2
- **XGBoost / LightGBM / scikit-learn:** vía pip en el mismo entorno
- **Hardware:** GPU (CUDA) para TuckER, CPU para XGBoost
- **Semillas:** TuckER usa `seed=20`, XGBoost/splits usan `RANDOM_STATE=42`
