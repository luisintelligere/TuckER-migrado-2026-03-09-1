# -*- coding: utf-8 -*-
"""
Genera embeddings 8D (notas S1 + S2) para inicializar TuckER
en la predicción de tercer semestre.

Vector: [MA1101, MA1001, FI1000, BT1211,  MA1002, MA1102, FI1100, CC1002]
Reglas:
  - Reintento curso: nota máxima obtenida.
  - Aprobado (T): 4.0
  - No cursó S2: 1.0  (=FAIL_GRADE)
  - Normalización: (nota − 4.0) / 3.0

Salida: embeddings_8d_binario_s3/
  embeddings_inicializados_binarios_8d.pt
  vocabulario_binario_8d.json
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

# ── Repo root ─────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model import TuckER

# ── Rutas ─────────────────────────────────────────────────────────────────
DF_BASE          = os.path.join(REPO_ROOT, "data", "dataframes_por_semestre")
DATASET_DIR_S3   = os.path.join(REPO_ROOT, "data", "dataset_2019_2020_2021_hacia_s3_binario")
OUTPUT_EMB_8D    = os.path.join(REPO_ROOT, "notebooks", "Experimento_warm_start", "embeddings_8d_binario_s3")
RESUMEN_CSV_S3   = os.path.join(DATASET_DIR_S3, "resumen_alumnos_generacion_split.csv")

# ── Cursos y orden del vector ─────────────────────────────────────────────
CURSOS_S1 = ["MA1101", "MA1001", "FI1000", "BT1211"]
CURSOS_S2 = ["MA1002", "MA1102", "FI1100", "CC1002"]
CURSOS_8D = CURSOS_S1 + CURSOS_S2   # índices 0-3: S1 | 4-7: S2

# ── Constantes ────────────────────────────────────────────────────────────
FAIL_GRADE         = 1.0   # no cursó / sin nota válida
APPROVED_TRANSFER  = 4.0   # Aprobado (T)
EDIM               = 8
RDIM_DUMMY         = 2

# ── Mapeo cohorte → (semestre_s1, semestre_s2) ───────────────────────────
COHORTE_SEMESTRES = {
    "20191": ("20191", "20192"),
    "20201": ("20201", "20202"),
    "20211": ("20211", "20212"),
}


# ============================================================================
# UTILIDADES
# ============================================================================

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ID"]           = df["ID"].astype(str).str.strip().str.upper()
    df["CURSO"]        = df["CURSO"].astype(str).str.strip().str.upper()
    df["ESTADO_CURSO"] = df["ESTADO_CURSO"].astype(str)
    return df


def _nota_efectiva(estado: str, nota_raw) -> float:
    """
    Devuelve la nota numérica aplicando:
      Aprobado (T) → 4.0 | valor numérico → float | fallback → 1.0
    """
    if "Aprobado (T)" in str(estado) or "Aprobado(T)" in str(estado):
        return APPROVED_TRANSFER
    try:
        return float(str(nota_raw).replace(",", "."))
    except Exception:
        if "Aprobado" in str(estado):
            return APPROVED_TRANSFER
        return FAIL_GRADE


def _normalizar(n) -> float:
    """(nota − 4) / 3  →  rango ≈ [−1, 1];  ausencia → −1.0"""
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return (FAIL_GRADE - 4.0) / 3.0
    return (float(n) - 4.0) / 3.0


def _construir_pivot(ruta_csv: str, cursos: list) -> pd.DataFrame:
    """
    Carga un CSV semestral, aplica _nota_efectiva y devuelve pivot
    ID × cursos con aggfunc=max (nota más alta si reintentó el curso).
    """
    if not os.path.exists(ruta_csv):
        print(f"  ⚠️  No encontrado: {ruta_csv}")
        return pd.DataFrame(columns=cursos)
    df = _norm_cols(pd.read_csv(ruta_csv, sep=";"))
    df = df[df["CURSO"].isin(cursos)].copy()
    df["NOTA_EF"] = [_nota_efectiva(e, n) for e, n in zip(df["ESTADO_CURSO"], df["NOTA"])]
    pivot = df.pivot_table(index="ID", columns="CURSO", values="NOTA_EF", aggfunc="max")
    return pivot.reindex(columns=cursos)


def _get_vocab_s3() -> SimpleNamespace:
    """Construye vocabulario entidades/relaciones desde train/valid/test S3."""
    entities, relations = set(), set()
    for fname in ("train.txt", "valid.txt", "test.txt"):
        path = os.path.join(DATASET_DIR_S3, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                h, r, t = parts
                entities.add(h); entities.add(t); relations.add(r)
    d = SimpleNamespace()
    d.entities      = sorted(entities)
    d.relations     = sorted(relations) + [r + "_reverse" for r in sorted(relations)]
    d.entity_idxs   = {e: i for i, e in enumerate(d.entities)}
    d.relation_idxs = {r: i for i, r in enumerate(d.relations)}
    return d


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n--- Generando embeddings 8D (notas S1 + S2) ---")
    os.makedirs(OUTPUT_EMB_8D, exist_ok=True)

    # 1) Leer resumen para mapear alumno → cohorte
    resumen = pd.read_csv(RESUMEN_CSV_S3)
    resumen["ID"] = resumen["ID"].astype(str).str.strip().str.upper()
    id_a_cohorte  = dict(zip(resumen["ID"], resumen["generacion_entrada"].astype(str)))
    ids_validos   = set(id_a_cohorte)
    print(f"   Alumnos válidos: {len(ids_validos)}")

    # 2) Construir tabla 8D por cohorte
    pivots_s1, pivots_s2 = [], []
    for cohorte, (sem_s1, sem_s2) in COHORTE_SEMESTRES.items():
        ids_c = {k for k, v in id_a_cohorte.items() if v == cohorte}
        p1 = _construir_pivot(os.path.join(DF_BASE, f"df_{sem_s1}.csv"), CURSOS_S1)
        p2 = _construir_pivot(os.path.join(DF_BASE, f"df_{sem_s2}.csv"), CURSOS_S2)
        p1 = p1[p1.index.isin(ids_c)]
        p2 = p2[p2.index.isin(ids_c)]
        print(f"   Cohorte {cohorte}: {len(p1)} alumnos S1 | {len(p2)} alumnos S2")
        pivots_s1.append(p1)
        pivots_s2.append(p2)

    tabla_s1 = pd.concat(pivots_s1, axis=0).fillna(FAIL_GRADE)
    tabla_s2 = pd.concat(pivots_s2, axis=0).fillna(FAIL_GRADE)  # no cursó → 1.0
    tabla_8d = tabla_s1.join(tabla_s2, how="outer").fillna(FAIL_GRADE)
    tabla_8d = tabla_8d.reindex(columns=CURSOS_8D)
    print(f"   Tabla 8D: {len(tabla_8d)} alumnos × {len(CURSOS_8D)} cursos")

    # 3) Vocabulario y modelo
    d      = _get_vocab_s3()
    modelo = TuckER(d, EDIM, RDIM_DUMMY, input_dropout=0, hidden_dropout1=0, hidden_dropout2=0)
    print(f"   Vocabulario: {len(d.entities)} entidades | {len(d.relations)} relaciones")

    # 4) Inicializar pesos de entidades alumno con vectores 8D
    count, sin_datos = 0, 0
    with torch.no_grad():
        for entity, idx in d.entity_idxs.items():
            if entity not in ids_validos:
                continue
            if entity not in tabla_8d.index:
                sin_datos += 1
                continue
            row = tabla_8d.loc[entity]
            vec = np.array([_normalizar(row[c]) for c in CURSOS_8D], dtype=np.float32)
            modelo.E.weight[idx] = torch.tensor(vec)
            count += 1

    print(f"   ✅ Pesos inicializados con datos reales: {count} alumnos")
    if sin_datos:
        print(f"   ⚠️  Sin datos en tabla_8d: {sin_datos}")

    # 5) Guardar
    path_pt   = os.path.join(OUTPUT_EMB_8D, "embeddings_inicializados_binarios_8d.pt")
    path_json = os.path.join(OUTPUT_EMB_8D, "vocabulario_binario_8d.json")
    torch.save(modelo.E.weight.data, path_pt)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump({"entities": d.entities, "relations": d.relations}, f, indent=2)

    print(f"\n   💾 Pesos    → {path_pt}")
    print(f"   💾 Vocab    → {path_json}")

    # 6) Estadísticas rápidas
    print("\nEstadísticas tabla 8D (nota efectiva antes de normalizar):")
    print(tabla_8d.describe().round(2).to_string())


if __name__ == "__main__":
    main()
