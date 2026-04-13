from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from validacion_temporal import (
    CURSOS_S3,
    DATA_DIR,
    GEN_TEST_OOT,
    GENS_TRAIN,
    RANDOM_STATE,
    RESULTS_DIR,
    asignar_generacion,
    build_features,
    build_target,
    choose_threshold,
    load_all,
    limpiar,
    metricas,
)


HANDCRAFTED_GRID = [
    {
        "label": "baseline",
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    },
    {
        "label": "conservador_d3",
        "n_estimators": 250,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    },
    {
        "label": "regularizado_d3",
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 0.0,
        "reg_lambda": 3.0,
        "reg_alpha": 0.0,
    },
    {
        "label": "profundo_d5",
        "n_estimators": 150,
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    },
    {
        "label": "profundo_reg",
        "n_estimators": 250,
        "max_depth": 5,
        "learning_rate": 0.03,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.2,
        "reg_lambda": 3.0,
        "reg_alpha": 0.0,
    },
    {
        "label": "rapido_lr",
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.10,
        "min_child_weight": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    },
    {
        "label": "subsample_08",
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    },
    {
        "label": "lambda_alta",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 0.0,
        "reg_lambda": 5.0,
        "reg_alpha": 0.0,
    },
    {
        "label": "gamma_minchild",
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 0.3,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    },
    {
        "label": "balanceado_suave",
        "n_estimators": 250,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "gamma": 0.1,
        "reg_lambda": 3.0,
        "reg_alpha": 0.0,
    },
]

RANDOM_SPACE = {
    "n_estimators": [80, 100, 120, 150, 180, 220, 260, 320, 400, 500],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15],
    "min_child_weight": [1, 2, 3, 5, 7, 10],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0],
    "reg_alpha": [0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Busqueda amplia de hiperparametros XGBoost por asignatura.")
    parser.add_argument("--n-random-configs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def config_key(config: dict) -> tuple:
    return (
        config["n_estimators"],
        config["max_depth"],
        config["learning_rate"],
        config["min_child_weight"],
        config["subsample"],
        config["colsample_bytree"],
        config["gamma"],
        config["reg_lambda"],
        config["reg_alpha"],
    )


def generate_param_grid(n_random_configs: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    configs = [dict(cfg) for cfg in HANDCRAFTED_GRID]
    seen = {config_key(cfg) for cfg in configs}

    while len(configs) < len(HANDCRAFTED_GRID) + n_random_configs:
        cfg = {
            "label": f"random_{len(configs) - len(HANDCRAFTED_GRID) + 1:03d}",
            "n_estimators": int(rng.choice(RANDOM_SPACE["n_estimators"])),
            "max_depth": int(rng.choice(RANDOM_SPACE["max_depth"])),
            "learning_rate": float(rng.choice(RANDOM_SPACE["learning_rate"])),
            "min_child_weight": int(rng.choice(RANDOM_SPACE["min_child_weight"])),
            "subsample": float(rng.choice(RANDOM_SPACE["subsample"])),
            "colsample_bytree": float(rng.choice(RANDOM_SPACE["colsample_bytree"])),
            "gamma": float(rng.choice(RANDOM_SPACE["gamma"])),
            "reg_lambda": float(rng.choice(RANDOM_SPACE["reg_lambda"])),
            "reg_alpha": float(rng.choice(RANDOM_SPACE["reg_alpha"])),
        }
        key = config_key(cfg)
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)
    return configs


def build_model(config: dict, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        min_child_weight=config["min_child_weight"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        gamma=config["gamma"],
        reg_lambda=config["reg_lambda"],
        reg_alpha=config["reg_alpha"],
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
    )


def evaluate_course(df_features: pd.DataFrame, df: pd.DataFrame, curso: str, param_grid: list[dict]) -> tuple[list[dict], dict | None, dict | None]:
    df_target = build_target(df, curso)
    df_train_t = df_target[df_target["generacion"].isin(GENS_TRAIN)]
    df_test_t = df_target[df_target["generacion"] == GEN_TEST_OOT]

    dm_train = pd.merge(df_features, df_train_t, on="ID", how="inner")
    dm_test = pd.merge(df_features, df_test_t, on="ID", how="inner")
    if len(dm_train) < 30 or len(dm_test) < 10:
        return [], None, None
    if dm_train["TARGET"].nunique() < 2 or dm_test["TARGET"].nunique() < 2:
        return [], None, None

    feat_cols = [c for c in dm_train.columns if c not in ("ID", "TARGET", "generacion")]
    X_train_full = dm_train[feat_cols]
    y_train_full = dm_train["TARGET"]
    X_test = dm_test[feat_cols]
    y_test = dm_test["TARGET"]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    scale_pos_weight = (y_train == 0).sum() / max(y_train.sum(), 1)
    rows: list[dict] = []
    best_val_row: dict | None = None
    best_oot_row: dict | None = None

    for config in param_grid:
        model = build_model(config, scale_pos_weight)
        model.fit(X_train, y_train)

        val_proba = model.predict_proba(X_val)[:, 1]
        threshold, val_f1 = choose_threshold(y_val, val_proba)
        try:
            val_auc = float(roc_auc_score(y_val, val_proba))
        except ValueError:
            val_auc = 0.0

        test_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (test_proba >= threshold).astype(int)

        row = {
            "curso": curso,
            "config": config["label"],
            "threshold": round(threshold, 2),
            "val_f1": round(float(val_f1), 3),
            "val_auc": round(val_auc, 3),
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "learning_rate": config["learning_rate"],
            "min_child_weight": config["min_child_weight"],
            "subsample": config["subsample"],
            "colsample_bytree": config["colsample_bytree"],
            "gamma": config["gamma"],
            "reg_lambda": config["reg_lambda"],
            "reg_alpha": config["reg_alpha"],
        }
        row.update(metricas(y_test, y_pred, test_proba))
        rows.append(row)

        if best_val_row is None or row["val_f1"] > best_val_row["val_f1"] or (
            row["val_f1"] == best_val_row["val_f1"] and row["val_auc"] > best_val_row["val_auc"]
        ):
            best_val_row = row
        if best_oot_row is None or row["f1"] > best_oot_row["f1"] or (
            row["f1"] == best_oot_row["f1"] and row["roc_auc"] > best_oot_row["roc_auc"]
        ):
            best_oot_row = row

    return rows, best_val_row, best_oot_row


def main() -> None:
    args = parse_args()
    print("Cargando datos para tuning temporal...")
    df_raw = load_all(DATA_DIR)
    df = asignar_generacion(limpiar(df_raw))
    df_features = build_features(df)
    param_grid = generate_param_grid(args.n_random_configs, args.seed)
    print(f"Configuraciones a probar por curso: {len(param_grid)}")

    all_rows: list[dict] = []
    best_val_rows: list[dict] = []
    best_oot_rows: list[dict] = []
    for curso in CURSOS_S3:
        rows, best_val_row, best_oot_row = evaluate_course(df_features, df, curso, param_grid)
        all_rows.extend(rows)
        if best_val_row is not None:
            best_val_rows.append(best_val_row)
            print(
                f"[BEST-VAL] {curso}: cfg={best_val_row['config']} val_f1={best_val_row['val_f1']} "
                f"F1_OOT={best_val_row['f1']} AUC_OOT={best_val_row['roc_auc']} tau={best_val_row['threshold']:.2f}"
            )
        if best_oot_row is not None:
            best_oot_rows.append(best_oot_row)
            print(
                f"[BEST-OOT] {curso}: cfg={best_oot_row['config']} val_f1={best_oot_row['val_f1']} "
                f"F1_OOT={best_oot_row['f1']} AUC_OOT={best_oot_row['roc_auc']} tau={best_oot_row['threshold']:.2f}"
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    all_path = RESULTS_DIR / "tuning_temporal_todos.csv"
    best_val_path = RESULTS_DIR / "tuning_temporal_mejores_validacion.csv"
    best_oot_path = RESULTS_DIR / "tuning_temporal_mejores_oot.csv"
    pd.DataFrame(all_rows).to_csv(all_path, index=False)
    pd.DataFrame(best_val_rows).to_csv(best_val_path, index=False)
    pd.DataFrame(best_oot_rows).to_csv(best_oot_path, index=False)
    print(f"\nResultados completos: {all_path}")
    print(f"Mejores por validacion: {best_val_path}")
    print(f"Mejores por OOT: {best_oot_path}")


if __name__ == "__main__":
    main()