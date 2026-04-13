# -*- coding: utf-8 -*-
"""
Validación inductiva sobre la cohorte 20221 → 20231 (generación siguiente).

Los alumnos de 2022 NO estuvieron en el entrenamiento. Se usan sus notas
de S1 (20221) y S2 (20222) para construir sus embeddings 8D y se evalúa
el modelo entrenado en las cohortes 20191-20201-20211.

Pipeline
--------
1. Genera tripletas (alumno, relación, curso) para 20221 → 20231.
2. Construye embeddings 8D para los alumnos nuevos.
3. Carga best_model.pt y aplica scoring de Tucker.
4. Evalúa con ranking de relaciones: Hits@1, Hits@2, MR, MRR.
5. Calcula Precision, Recall y F1 para la clase "reprueba".
"""

import sys, os, json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace
from collections import defaultdict
from model import TuckER

# ── Rutas ────────────────────────────────────────────────────────────────────
DF_BASE      = REPO_ROOT / "data" / "dataframes_por_semestre"
MODEL_PATH   = REPO_ROOT / "results" / "prueba_s3_8d_rdim11" / "best_model.pt"
VOCAB_PATH   = (REPO_ROOT / "notebooks" / "Experimento_warm_start"
                / "embeddings_8d_binario_s3" / "vocabulario_binario_8d.json")
OUTPUT_DIR   = REPO_ROOT / "results" / "validacion_cohorte_20221_20231"

# ── Cohorte nueva ────────────────────────────────────────────────────────────
SEM_S1  = "20221"    # notas S1 para embeddings
SEM_S2  = "20222"    # notas S2 para embeddings
SEM_OBJ = "20231"    # semestre objetivo (S3)

# ── Cursos ───────────────────────────────────────────────────────────────────
CURSOS_S1 = ["MA1101", "MA1001", "FI1000", "BT1211"]
CURSOS_S2 = ["MA1002", "MA1102", "FI1100", "CC1002"]
CURSOS_S3 = ["MA2001", "MA2601", "FI2001", "FI2003", "IQ2211"]
CURSOS_8D = CURSOS_S1 + CURSOS_S2
CURSOS_PERMITIDOS = set(CURSOS_S1 + CURSOS_S2 + CURSOS_S3)

FAIL_GRADE        = 1.0
APPROVED_TRANSFER = 4.0
EDIM, RDIM        = 8, 11    # deben coincidir con el modelo entrenado

# ═════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═════════════════════════════════════════════════════════════════════════════

def _norm_cols(df):
    df = df.copy()
    df["ID"]           = df["ID"].astype(str).str.strip().str.upper()
    df["CURSO"]        = df["CURSO"].astype(str).str.strip().str.upper()
    df["ESTADO_CURSO"] = df["ESTADO_CURSO"].astype(str)
    return df


def _nota_efectiva(estado, nota_raw):
    if "Aprobado (T)" in str(estado) or "Aprobado(T)" in str(estado):
        return APPROVED_TRANSFER
    try:
        return float(str(nota_raw).replace(",", "."))
    except Exception:
        if "Aprobado" in str(estado):
            return APPROVED_TRANSFER
        return FAIL_GRADE


def _normalizar(n):
    """(nota − 4) / 3  →  rango ≈ [−1, 1];  ausencia → −1.0"""
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return (FAIL_GRADE - 4.0) / 3.0
    return (float(n) - 4.0) / 3.0


def _construir_pivot(sem, cursos):
    ruta = DF_BASE / f"df_{sem}.csv"
    if not ruta.exists():
        print(f"  ⚠️  No existe: {ruta}")
        return pd.DataFrame(columns=cursos)
    df = _norm_cols(pd.read_csv(ruta, sep=";"))
    df = df[df["CURSO"].isin(cursos)].copy()
    df["NOTA_EF"] = [_nota_efectiva(e, n)
                     for e, n in zip(df["ESTADO_CURSO"], df["NOTA"])]
    pivot = df.pivot_table(index="ID", columns="CURSO",
                           values="NOTA_EF", aggfunc="max")
    return pivot.reindex(columns=cursos)


def _determinar_relacion(estado, nota_raw):
    """Devuelve 'aprueba' o 'reprueba'. None si no aplica."""
    e = str(estado)
    if "Aprobado" in e and "Eliminado" not in e and "Reprobado" not in e:
        return "aprueba"
    if "Reprobado" in e or "Eliminado" in e:
        return "reprueba"
    try:
        nota = float(str(nota_raw).replace(",", "."))
        return "reprueba" if nota < 4.0 else "aprueba"
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# 1. GENERAR TRIPLETAS PARA LA COHORTE NUEVA
# ═════════════════════════════════════════════════════════════════════════════

def generar_tripletas_cohorte_nueva():
    print(f"\n[1] Generando tripletas: {SEM_S1} → {SEM_OBJ}")

    # Alumnos que cursaron los 4 fundamentales de S1
    df_s1 = _norm_cols(pd.read_csv(DF_BASE / f"df_{SEM_S1}.csv", sep=";"))
    alumnos_con_4 = (
        df_s1[df_s1["CURSO"].isin(CURSOS_S1)]
        .groupby("ID")["CURSO"]
        .nunique()
    )
    ids_validos = set(alumnos_con_4[alumnos_con_4 == 4].index)
    print(f"   Alumnos con los 4 fundamentales de S1: {len(ids_validos)}")

    # Carga semestre objetivo
    df_obj = _norm_cols(pd.read_csv(DF_BASE / f"df_{SEM_OBJ}.csv", sep=";"))
    df_obj = df_obj[
        df_obj["ID"].isin(ids_validos) &
        df_obj["CURSO"].isin(CURSOS_PERMITIDOS)
    ].copy()
    df_obj["relacion"] = [
        _determinar_relacion(e, n)
        for e, n in zip(df_obj["ESTADO_CURSO"], df_obj["NOTA"])
    ]
    df_obj = df_obj.dropna(subset=["relacion"])

    tripletas = list(zip(df_obj["ID"], df_obj["relacion"], df_obj["CURSO"]))
    print(f"   Tripletas generadas: {len(tripletas)}")
    print(f"   Alumnos únicos en tripletas: {df_obj['ID'].nunique()}")
    print(f"   Distribución relaciones: {df_obj['relacion'].value_counts().to_dict()}")
    return tripletas, df_obj["ID"].unique().tolist()


# ═════════════════════════════════════════════════════════════════════════════
# 2. EMBEDDINGS 8D PARA ALUMNOS NUEVOS
# ═════════════════════════════════════════════════════════════════════════════

def construir_embeddings_nuevos(ids_alumnos):
    print(f"\n[2] Construyendo embeddings 8D para {len(ids_alumnos)} alumnos")

    p1 = _construir_pivot(SEM_S1, CURSOS_S1).fillna(FAIL_GRADE)
    p2 = _construir_pivot(SEM_S2, CURSOS_S2).fillna(FAIL_GRADE)

    p1 = p1[p1.index.isin(ids_alumnos)]
    p2 = p2[p2.index.isin(ids_alumnos)]
    print(f"   Con datos S1: {len(p1)} | Con datos S2: {len(p2)}")

    tabla = p1.join(p2, how="outer").fillna(FAIL_GRADE)
    tabla = tabla.reindex(columns=CURSOS_8D)

    # Vector normalizado por alumno
    emb_dict = {}
    for alumno in ids_alumnos:
        if alumno in tabla.index:
            row = tabla.loc[alumno]
            vec = np.array([_normalizar(row[c]) for c in CURSOS_8D], dtype=np.float32)
        else:
            vec = np.array([_normalizar(FAIL_GRADE)] * 8, dtype=np.float32)
        emb_dict[alumno] = vec

    print(f"   Embeddings construidos: {len(emb_dict)}")
    return emb_dict


# ═════════════════════════════════════════════════════════════════════════════
# 3. CARGAR MODELO Y VOCAB DE ENTRENAMIENTO
# ═════════════════════════════════════════════════════════════════════════════

def cargar_modelo_entrenado():
    print(f"\n[3] Cargando modelo desde:\n    {MODEL_PATH}")

    with open(VOCAB_PATH, encoding="utf-8") as f:
        vocab = json.load(f)

    d = SimpleNamespace()
    d.entities      = vocab["entities"]
    d.relations     = vocab["relations"]
    d.entity_idxs   = {e: i for i, e in enumerate(d.entities)}
    d.relation_idxs = {r: i for i, r in enumerate(d.relations)}

    model = TuckER(d, EDIM, RDIM, input_dropout=0.3,
                   hidden_dropout1=0.4, hidden_dropout2=0.5)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    print(f"   Entidades en vocab: {len(d.entities)}")
    print(f"   Relaciones en vocab: {d.relations}")
    return model, d


# ═════════════════════════════════════════════════════════════════════════════
# 4. SCORING INDUCTIVO (alumnos nuevos + cursos entrenados)
# ═════════════════════════════════════════════════════════════════════════════

def score_inductive(model, d, emb_nuevos, tripletas):
    """
    Para cada tripleta (alumno, relacion_verdadera, curso):
    - Usa el embedding 8D del alumno (no estaba en entrenamiento).
    - Usa el embedding entrenado del curso.
    - Rankea las relaciones y obtiene el rango de la verdadera.
    Devuelve lista de dicts con resultados por tripleta.
    """
    print(f"\n[4] Evaluando {len(tripletas)} tripletas (scoring inductivo)...")

    # Relaciones forward (sin _reverse)
    rels_fwd = [r for r in d.relations if not r.endswith("_reverse")]
    rel_idxs_fwd = [d.relation_idxs[r] for r in rels_fwd]
    print(f"   Relaciones forward: {rels_fwd}")

    # Pre-calcular W_r para cada relación forward
    # W shape: (rdim, edim, edim)
    # W_mat_r = e_r @ W.view(rdim, edim*edim) → reshape a (edim, edim)
    W_flat = model.W.data.view(RDIM, EDIM * EDIM)  # (11, 64)
    W_mats = {}
    with torch.no_grad():
        for r_name, r_idx in zip(rels_fwd, rel_idxs_fwd):
            e_r = model.R.weight[r_idx]          # (11,)
            W_mats[r_name] = (e_r @ W_flat).view(EDIM, EDIM)  # (8, 8)

    resultados = []
    sin_curso = 0

    with torch.no_grad():
        for alumno, rel_true, curso in tripletas:
            # Curso debe estar en vocab (entidades entrenadas)
            if curso not in d.entity_idxs:
                sin_curso += 1
                continue

            t_idx = d.entity_idxs[curso]
            e_t   = model.E.weight[t_idx]  # (8,)

            # Embedding alumno (8D grades)
            e_h = torch.tensor(emb_nuevos[alumno], dtype=torch.float32)  # (8,)

            # Aplicar bn0 a e_h (usar stats del entrenamiento, model.eval())
            e_h_bn = model.bn0(e_h.unsqueeze(0)).squeeze(0)  # (8,)
            # Dropout está desactivado en eval()

            # Score para cada relación forward
            scores = {}
            for r_name in rels_fwd:
                W_r = W_mats[r_name]             # (8, 8)
                x   = e_h_bn @ W_r               # (8,)
                x   = model.bn1(x.unsqueeze(0)).squeeze(0)  # bn1
                scores[r_name] = (x @ e_t).item()           # scalar

            # Rank de la relación verdadera
            orden = sorted(rels_fwd, key=lambda r: scores[r], reverse=True)
            if rel_true in orden:
                rank = orden.index(rel_true) + 1
            else:
                rank = len(rels_fwd) + 1  # relación desconocida

            pred_rel = orden[0]  # relación predicha (mayor score)

            resultados.append({
                "alumno":   alumno,
                "rel_true": rel_true,
                "rel_pred": pred_rel,
                "curso":    curso,
                "rank":     rank,
                "score_aprueba": scores.get("aprueba", float("nan")),
                "score_reprueba": scores.get("reprueba", float("nan")),
            })

    if sin_curso:
        print(f"   ⚠️  Tripletas omitidas (curso no en vocab): {sin_curso}")
    print(f"   Tripletas evaluadas: {len(resultados)}")
    return resultados


# ═════════════════════════════════════════════════════════════════════════════
# 5. MÉTRICAS
# ═════════════════════════════════════════════════════════════════════════════

def calcular_metricas(resultados):
    print("\n" + "=" * 60)
    print("MÉTRICAS DE EVALUACIÓN — COHORTE 2022→2023")
    print("=" * 60)

    ranks      = np.array([r["rank"] for r in resultados], dtype=float)
    rel_true   = [r["rel_true"] for r in resultados]
    rel_pred   = [r["rel_pred"] for r in resultados]
    n          = len(ranks)

    # ── Relation-ranking metrics ──────────────────────────────────────────
    hits1 = float(np.mean(ranks == 1))
    hits2 = float(np.mean(ranks <= 2))
    mr    = float(np.mean(ranks))
    mrr   = float(np.mean(1.0 / ranks))

    print(f"\n── Ranking de Relaciones ({n} pares alumno-curso) ──────────────")
    print(f"   Hits@1 : {hits1:.4f}  ({int(hits1*n)}/{n})")
    print(f"   Hits@2 : {hits2:.4f}  ({int(hits2*n)}/{n})")
    print(f"   MR     : {mr:.4f}")
    print(f"   MRR    : {mrr:.4f}")

    # ── Precision / Recall / F1 para "reprueba" ───────────────────────────
    TP = sum(1 for t, p in zip(rel_true, rel_pred) if t == "reprueba" and p == "reprueba")
    FP = sum(1 for t, p in zip(rel_true, rel_pred) if t == "aprueba"  and p == "reprueba")
    FN = sum(1 for t, p in zip(rel_true, rel_pred) if t == "reprueba" and p == "aprueba")
    TN = sum(1 for t, p in zip(rel_true, rel_pred) if t == "aprueba"  and p == "aprueba")

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (TP + TN) / n if n > 0 else 0.0

    n_reprueba = TP + FN
    n_aprueba  = FP + TN
    print(f"\n── Precision / Recall — clase REPRUEBA ─────────────────────")
    print(f"   Total reprueba (real) : {n_reprueba}  ({n_reprueba/n*100:.1f}%)")
    print(f"   Total aprueba  (real) : {n_aprueba}  ({n_aprueba/n*100:.1f}%)")
    print(f"   TP={TP}  FP={FP}  FN={FN}  TN={TN}")
    print(f"   Precision : {precision:.4f}")
    print(f"   Recall    : {recall:.4f}")
    print(f"   F1-score  : {f1:.4f}")
    print(f"   Accuracy  : {accuracy:.4f}")

    # ── Distribución de predicciones ────────────────────────────────────
    n_pred_rep = sum(1 for p in rel_pred if p == "reprueba")
    n_pred_apr = sum(1 for p in rel_pred if p == "aprueba")
    print(f"\n── Distribución predicciones ────────────────────────────────")
    print(f"   Predijo aprueba  : {n_pred_apr}  ({n_pred_apr/n*100:.1f}%)")
    print(f"   Predijo reprueba : {n_pred_rep}  ({n_pred_rep/n*100:.1f}%)")

    return {
        "hits@1": hits1, "hits@2": hits2, "mr": mr, "mrr": mrr,
        "precision_reprueba": precision,
        "recall_reprueba": recall,
        "f1_reprueba": f1,
        "accuracy": accuracy,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "n_total": n,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 6. GUARDAR RESULTADOS
# ═════════════════════════════════════════════════════════════════════════════

def guardar_resultados(resultados, metricas):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_res = pd.DataFrame(resultados)
    path_det  = OUTPUT_DIR / "predicciones_por_tripleta.csv"
    path_met  = OUTPUT_DIR / "metricas_cohorte_20221_20231.json"

    df_res.to_csv(path_det, index=False, encoding="utf-8-sig")
    with open(path_met, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    print(f"\n[6] Guardado:")
    print(f"   {path_det}")
    print(f"   {path_met}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("VALIDACIÓN INDUCTIVA — COHORTE 20221 → 20231")
    print("=" * 60)

    tripletas, ids_alumnos   = generar_tripletas_cohorte_nueva()
    emb_nuevos               = construir_embeddings_nuevos(ids_alumnos)
    model, d                 = cargar_modelo_entrenado()
    resultados               = score_inductive(model, d, emb_nuevos, tripletas)
    metricas                 = calcular_metricas(resultados)
    guardar_resultados(resultados, metricas)

    print("\n✅ Validación completada.")


if __name__ == "__main__":
    main()
