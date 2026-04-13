"""
Validación temporal del modelo XGBoost para predicción de reprobación en S3.

Experimento 1:
    Entrena con mezcla aleatoria 80/20 usando generaciones completas 2019-2023.

Experimento 2 (out-of-time):
    Entrena exclusivamente con generaciones 2019-2022 y testea con generación 2023,
    que el modelo nunca vio durante el ajuste de umbral ni el entrenamiento.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ── Configuración ─────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "dataframes_por_semestre"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

CURSOS_S1 = ["MA1101", "MA1001", "FI1000", "BT1211"]
CURSOS_S2 = ["CC1002", "MA1002", "MA1102", "FI1100"]
CURSOS_S3 = ["MA2001", "MA2601", "FI2001", "FI2003", "IQ2211"]

# Generaciones completas (>80% de alumnos llegan a S3) — solo 2019 en adelante
GENS_COMPLETAS = ["20191", "20201", "20211", "20221", "20231"]
GENS_TRAIN    = ["20191", "20201", "20211", "20221"]
GEN_TEST_OOT  = "20231"  # Out-of-time test: generación completamente separada

FAIL_GRADE = 1.0
APPROVED_TRANSFER_GRADE = 4.0
RANDOM_STATE = 42


# ── Utilidades ────────────────────────────────────────────────

def load_all(data_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "df_*.csv")))
    frames = [
        pd.read_csv(f, sep=";", encoding="utf-8", on_bad_lines="skip")
        for f in files
    ]
    return pd.concat(frames, ignore_index=True)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ESTADO_CURSO"] = df["ESTADO_CURSO"].astype(str).str.strip()
    df = df[df["ESTADO_CURSO"].str.contains("Aprobado|Reprobado", na=False)].copy()
    df["NOTA"] = pd.to_numeric(
        df["NOTA"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    mask = df["NOTA"].isna()
    aprobado_t_mask = mask & df["ESTADO_CURSO"].str.contains(r"Aprobado \(T\)", na=False)
    missing_mask = mask & ~aprobado_t_mask
    df.loc[aprobado_t_mask, "NOTA"] = APPROVED_TRANSFER_GRADE
    df.loc[missing_mask, "NOTA"] = FAIL_GRADE
    df["NOTA_FALTANTE"] = missing_mask.astype(int)
    df["SEMESTRE"] = df["SEMESTRE"].astype(str)
    return df


def asignar_generacion(df: pd.DataFrame) -> pd.DataFrame:
    """Asigna a cada alumno la generación = primer semestre en que cursó S1."""
    primer = (
        df[df["CURSO"].isin(CURSOS_S1)]
        .groupby("ID")["SEMESTRE"]
        .min()
        .reset_index()
        .rename(columns={"SEMESTRE": "generacion"})
    )
    return df.merge(primer, on="ID", how="left")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_courses = CURSOS_S1 + CURSOS_S2
    rows = df[df["CURSO"].isin(feature_courses)].copy()

    notas = rows.pivot_table(index="ID", columns="CURSO", values="NOTA", aggfunc="max").reset_index()
    missing = rows.pivot_table(index="ID", columns="CURSO", values="NOTA_FALTANTE", aggfunc="max").reset_index()

    for c in feature_courses:
        if c not in notas.columns:
            notas[c] = np.nan
        if c not in missing.columns:
            missing[c] = np.nan

    notas[feature_courses] = notas[feature_courses].fillna(FAIL_GRADE)
    missing = missing.rename(columns={c: f"{c}_faltante" for c in feature_courses})
    missing_cols = [f"{c}_faltante" for c in feature_courses]
    missing[missing_cols] = missing[missing_cols].fillna(1).astype(int)

    return notas[["ID"] + feature_courses].merge(missing[["ID"] + missing_cols], on="ID", how="left")


def build_target(df: pd.DataFrame, curso: str) -> pd.DataFrame:
    sub = df[df["CURSO"] == curso].sort_values("SEMESTRE").groupby("ID").first().reset_index()
    sub["TARGET"] = np.where(sub["ESTADO_CURSO"].str.contains("Reprobado", na=False), 1, 0)
    return sub[["ID", "TARGET", "generacion"]]


def choose_threshold(y_true, proba) -> tuple[float, float]:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        score = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, float(t)
    return best_t, best_f1


def metricas(y_true, y_pred, proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "reprobados": int(y_true.sum()),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
        "roc_auc": round(roc_auc_score(y_true, proba), 3),
        "balanced_acc": round(balanced_accuracy_score(y_true, y_pred), 3),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ── Experimentos ──────────────────────────────────────────────

def experimento_expandido(df_features: pd.DataFrame, df: pd.DataFrame) -> list[dict]:
    """Generaciones 2019-2023, split aleatorio 80/20."""
    resultados = []
    for curso in CURSOS_S3:
        df_target = build_target(df, curso)
        # Solo alumnos de generaciones completas
        df_target = df_target[df_target["generacion"].isin(GENS_COMPLETAS)]
        dm = pd.merge(df_features, df_target, on="ID", how="inner")
        if len(dm) < 30 or dm["TARGET"].nunique() < 2:
            continue

        feat_cols = [c for c in dm.columns if c not in ("ID", "TARGET", "generacion")]
        X, y = dm[feat_cols], dm["TARGET"]

        X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.25, random_state=RANDOM_STATE, stratify=y_tv)

        spw = (y_train == 0).sum() / max(y_train.sum(), 1)
        model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                              scale_pos_weight=spw, random_state=RANDOM_STATE,
                              eval_metric="logloss", tree_method="hist", n_jobs=1)
        model.fit(X_train, y_train)

        tau, val_f1 = choose_threshold(y_val, model.predict_proba(X_val)[:, 1])
        proba_test = model.predict_proba(X_test)[:, 1]
        y_pred = (proba_test >= tau).astype(int)

        row = {"experimento": "expandido", "curso": curso, "threshold": round(tau, 2), "val_f1": round(val_f1, 3)}
        row.update(metricas(y_test, y_pred, proba_test))
        resultados.append(row)
        print(f"[Expandido] {curso}: n={row['n']} F1={row['f1']} ROC-AUC={row['roc_auc']} tau={tau:.2f}")
    return resultados


def experimento_out_of_time(df_features: pd.DataFrame, df: pd.DataFrame) -> list[dict]:
    """Entrena en 2019-2022, testea en 2023."""
    resultados = []
    for curso in CURSOS_S3:
        df_target = build_target(df, curso)
        df_train_t = df_target[df_target["generacion"].isin(GENS_TRAIN)]
        df_test_t  = df_target[df_target["generacion"] == GEN_TEST_OOT]

        dm_train = pd.merge(df_features, df_train_t, on="ID", how="inner")
        dm_test  = pd.merge(df_features, df_test_t,  on="ID", how="inner")

        if len(dm_train) < 30 or len(dm_test) < 10:
            print(f"[OOT] {curso}: muestra insuficiente (train={len(dm_train)}, test={len(dm_test)}). Saltando.")
            continue
        if dm_train["TARGET"].nunique() < 2 or dm_test["TARGET"].nunique() < 2:
            print(f"[OOT] {curso}: clases insuficientes. Saltando.")
            continue

        feat_cols = [c for c in dm_train.columns if c not in ("ID", "TARGET", "generacion")]
        X_train_full = dm_train[feat_cols]
        y_train_full = dm_train["TARGET"]

        # Partición interna solo para elegir umbral
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_full
        )

        spw = (y_train == 0).sum() / max(y_train.sum(), 1)
        model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                              scale_pos_weight=spw, random_state=RANDOM_STATE,
                              eval_metric="logloss", tree_method="hist", n_jobs=1)
        model.fit(X_train, y_train)

        tau, val_f1 = choose_threshold(y_val, model.predict_proba(X_val)[:, 1])

        X_test = dm_test[feat_cols]
        y_test = dm_test["TARGET"]
        proba_test = model.predict_proba(X_test)[:, 1]
        y_pred = (proba_test >= tau).astype(int)

        row = {"experimento": "out_of_time", "curso": curso, "threshold": round(tau, 2), "val_f1": round(val_f1, 3)}
        row.update(metricas(y_test, y_pred, proba_test))
        resultados.append(row)
        print(f"[OOT]      {curso}: n={row['n']} F1={row['f1']} ROC-AUC={row['roc_auc']} tau={tau:.2f}")
    return resultados


# ── Main ──────────────────────────────────────────────────────

def main():
    print("Cargando todos los archivos...")
    df_raw = load_all(DATA_DIR)
    df = limpiar(df_raw)
    df = asignar_generacion(df)
    print(f"Registros totales: {len(df):,}  |  Alumnos únicos: {df['ID'].nunique():,}")

    df_features = build_features(df)

    print("\n=== EXPERIMENTO 1: Generaciones 2019-2023 (split aleatorio) ===")
    res_exp1 = experimento_expandido(df_features, df)

    print("\n=== EXPERIMENTO 2: Out-of-time — train 2019-2022, test 2023 ===")
    res_exp2 = experimento_out_of_time(df_features, df)

    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = pd.DataFrame(res_exp1 + res_exp2)
    out_path = RESULTS_DIR / "validacion_temporal.csv"
    all_results.to_csv(out_path, index=False)
    print(f"\nResultados guardados en {out_path}")

    print("\n=== RESUMEN COMPARATIVO ===")
    for exp in ["expandido", "out_of_time"]:
        sub = all_results[all_results["experimento"] == exp]
        label = "Exp1-Expandido" if exp == "expandido" else "Exp2-OOT-2023"
        print(f"\n{label}:")
        print(sub[["curso","n","reprobados","threshold","f1","roc_auc","balanced_acc","tp","fp","fn","tn"]].to_string(index=False))


if __name__ == "__main__":
    main()
