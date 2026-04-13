"""
Validación temporal del modelo XGBoost para predicción de reprobación en S2
a partir de resultados de S1.

Experimento 1:
    Entrena con mezcla aleatoria 80/20 usando generaciones completas 2019-2023.

Experimento 2 (out-of-time):
    Entrena exclusivamente con generaciones 2019-2022 y testea con generación 2023,
    que el modelo nunca vio durante el ajuste de umbral ni el entrenamiento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from validacion_temporal import (
    CURSOS_S1,
    CURSOS_S2,
    DATA_DIR,
    FAIL_GRADE,
    GENS_COMPLETAS,
    GENS_TRAIN,
    GEN_TEST_OOT,
    RANDOM_STATE,
    RESULTS_DIR,
    asignar_generacion,
    build_target,
    choose_threshold,
    load_all,
    limpiar,
    metricas,
)


# ── Features: solo cursos de S1 ──────────────────────────────

def build_features_s1(df: pd.DataFrame) -> pd.DataFrame:
    """Construye features usando SOLO cursos de primer semestre."""
    feature_courses = CURSOS_S1
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


# ── Experimentos ──────────────────────────────────────────────

def experimento_expandido(df_features: pd.DataFrame, df: pd.DataFrame) -> list[dict]:
    """Generaciones 2019-2023, split aleatorio 80/20."""
    resultados = []
    for curso in CURSOS_S2:
        df_target = build_target(df, curso)
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
    for curso in CURSOS_S2:
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
    print("=== Prediccion S1 -> S2: Validacion temporal ===")
    print("Cargando todos los archivos...")
    df_raw = load_all(DATA_DIR)
    df = limpiar(df_raw)
    df = asignar_generacion(df)
    print(f"Registros totales: {len(df):,}  |  Alumnos únicos: {df['ID'].nunique():,}")

    df_features = build_features_s1(df)
    print(f"Features S1: {[c for c in df_features.columns if c != 'ID']}")

    print("\n=== EXPERIMENTO 1: Generaciones 2019-2023 (split aleatorio) ===")
    res_exp1 = experimento_expandido(df_features, df)

    print("\n=== EXPERIMENTO 2: Out-of-time — train 2019-2022, test 2023 ===")
    res_exp2 = experimento_out_of_time(df_features, df)

    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = pd.DataFrame(res_exp1 + res_exp2)
    out_path = RESULTS_DIR / "validacion_temporal_s2.csv"
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
