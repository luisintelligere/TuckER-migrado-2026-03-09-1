"""
Tuning walk-forward estricto para predicción S1 → S2.

Esquema temporal:
    Train:  generaciones 2019, 2020, 2021
    Valid:  generación 2022
    Test:   generación 2023

Etapa 1 — Semillas: toma las mejores configs del tuning amplio (tuning_temporal_s2)
          más el baseline.
Etapa 2 — Búsqueda local: genera variantes cercanas a cada semilla.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from validacion_temporal import (
    CURSOS_S2,
    DATA_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    asignar_generacion,
    build_target,
    choose_threshold,
    load_all,
    limpiar,
    metricas,
)
from validacion_temporal_s2 import build_features_s1


TRAIN_GENS = ["20191", "20201", "20211"]
VALID_GEN = "20221"
TEST_GEN = "20231"

BASELINE_CONFIG = {
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
}

SPACE = {
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
    parser = argparse.ArgumentParser(description="Tuning walk-forward S1->S2.")
    parser.add_argument("--top-seeds-per-metric", type=int, default=5)
    parser.add_argument("--local-random-per-seed", type=int, default=8)
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


def nearest_options(options: list, value, width: int = 3) -> list:
    ordered = sorted(options, key=lambda candidate: (abs(candidate - value), candidate))
    return sorted(ordered[:width])


def normalize_config_row(row: pd.Series | dict, label: str) -> dict:
    return {
        "label": label,
        "n_estimators": int(row["n_estimators"]),
        "max_depth": int(row["max_depth"]),
        "learning_rate": float(row["learning_rate"]),
        "min_child_weight": int(row["min_child_weight"]),
        "subsample": float(row["subsample"]),
        "colsample_bytree": float(row["colsample_bytree"]),
        "gamma": float(row["gamma"]),
        "reg_lambda": float(row["reg_lambda"]),
        "reg_alpha": float(row["reg_alpha"]),
    }


def load_seed_configs(top_seeds_per_metric: int) -> dict[str, list[dict]]:
    source_path = RESULTS_DIR / "tuning_temporal_s2_todos.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"No existe {source_path}; ejecuta primero tuning_temporal_s2.py")

    source = pd.read_csv(source_path)
    seeds_by_course: dict[str, list[dict]] = {}
    for curso in CURSOS_S2:
        sub = source[source["curso"] == curso].copy()
        chosen: list[dict] = [dict(BASELINE_CONFIG)]
        seen = {config_key(BASELINE_CONFIG)}

        ranked_parts = [
            sub.sort_values(["val_f1", "val_auc"], ascending=False).head(top_seeds_per_metric),
            sub.sort_values(["val_auc", "val_f1"], ascending=False).head(top_seeds_per_metric),
        ]
        for part in ranked_parts:
            for idx, row in enumerate(part.to_dict("records"), start=1):
                cfg = normalize_config_row(row, label=f"seed_{row['config']}_{idx:02d}")
                key = config_key(cfg)
                if key in seen:
                    continue
                seen.add(key)
                chosen.append(cfg)

        seeds_by_course[curso] = chosen
    return seeds_by_course


def generate_local_configs(course: str, seeds: list[dict], local_random_per_seed: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed + sum(ord(ch) for ch in course))
    configs: list[dict] = []
    seen: set[tuple] = set()

    for seed_cfg in seeds:
        base = dict(seed_cfg)
        key = config_key(base)
        if key not in seen:
            seen.add(key)
            configs.append(base)

        local_space = {
            "n_estimators": nearest_options(SPACE["n_estimators"], base["n_estimators"], width=4),
            "max_depth": nearest_options(SPACE["max_depth"], base["max_depth"], width=3),
            "learning_rate": nearest_options(SPACE["learning_rate"], base["learning_rate"], width=3),
            "min_child_weight": nearest_options(SPACE["min_child_weight"], base["min_child_weight"], width=3),
            "subsample": nearest_options(SPACE["subsample"], base["subsample"], width=3),
            "colsample_bytree": nearest_options(SPACE["colsample_bytree"], base["colsample_bytree"], width=3),
            "gamma": nearest_options(SPACE["gamma"], base["gamma"], width=3),
            "reg_lambda": nearest_options(SPACE["reg_lambda"], base["reg_lambda"], width=3),
            "reg_alpha": nearest_options(SPACE["reg_alpha"], base["reg_alpha"], width=3),
        }

        generated = 0
        attempts = 0
        while generated < local_random_per_seed and attempts < local_random_per_seed * 10:
            attempts += 1
            cfg = {
                "label": f"{base['label']}_local_{generated + 1:02d}",
                "n_estimators": int(rng.choice(local_space["n_estimators"])),
                "max_depth": int(rng.choice(local_space["max_depth"])),
                "learning_rate": float(rng.choice(local_space["learning_rate"])),
                "min_child_weight": int(rng.choice(local_space["min_child_weight"])),
                "subsample": float(rng.choice(local_space["subsample"])),
                "colsample_bytree": float(rng.choice(local_space["colsample_bytree"])),
                "gamma": float(rng.choice(local_space["gamma"])),
                "reg_lambda": float(rng.choice(local_space["reg_lambda"])),
                "reg_alpha": float(rng.choice(local_space["reg_alpha"])),
            }
            key = config_key(cfg)
            if key in seen:
                continue
            seen.add(key)
            configs.append(cfg)
            generated += 1

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


def evaluate_course(df_features: pd.DataFrame, df: pd.DataFrame, curso: str, configs: list[dict]) -> tuple[list[dict], dict | None, dict | None]:
    df_target = build_target(df, curso)
    df_train_t = df_target[df_target["generacion"].isin(TRAIN_GENS)]
    df_val_t = df_target[df_target["generacion"] == VALID_GEN]
    df_test_t = df_target[df_target["generacion"] == TEST_GEN]

    dm_train = pd.merge(df_features, df_train_t, on="ID", how="inner")
    dm_val = pd.merge(df_features, df_val_t, on="ID", how="inner")
    dm_test = pd.merge(df_features, df_test_t, on="ID", how="inner")
    if min(len(dm_train), len(dm_val), len(dm_test)) < 10:
        return [], None, None
    if dm_train["TARGET"].nunique() < 2 or dm_val["TARGET"].nunique() < 2 or dm_test["TARGET"].nunique() < 2:
        return [], None, None

    feat_cols = [c for c in dm_train.columns if c not in ("ID", "TARGET", "generacion")]
    X_train = dm_train[feat_cols]
    y_train = dm_train["TARGET"]
    X_val = dm_val[feat_cols]
    y_val = dm_val["TARGET"]
    X_test = dm_test[feat_cols]
    y_test = dm_test["TARGET"]

    scale_pos_weight = (y_train == 0).sum() / max(y_train.sum(), 1)

    rows: list[dict] = []
    best_val_row: dict | None = None
    best_oot_row: dict | None = None

    for config in configs:
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
            "split": "train_2019_2021__valid_2022__test_2023",
            "curso": curso,
            "config": config["label"],
            "threshold": round(float(threshold), 2),
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
            "n_train": int(len(dm_train)),
            "n_valid": int(len(dm_val)),
            "n_test": int(len(dm_test)),
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


def build_report_summary(all_rows: pd.DataFrame, best_val_df: pd.DataFrame, best_oot_df: pd.DataFrame) -> pd.DataFrame:
    baseline = all_rows[all_rows["config"] == "baseline"][
        ["curso", "precision", "recall", "f1", "roc_auc", "threshold"]
    ].rename(
        columns={
            "precision": "prec_baseline",
            "recall": "recall_baseline",
            "f1": "f1_baseline",
            "roc_auc": "auc_baseline",
            "threshold": "tau_baseline",
        }
    )

    best_val = best_val_df[
        ["curso", "config", "precision", "recall", "f1", "roc_auc", "threshold", "val_f1", "val_auc"]
    ].rename(
        columns={
            "config": "config_best_val",
            "precision": "prec_best_val",
            "recall": "recall_best_val",
            "f1": "f1_best_val",
            "roc_auc": "auc_best_val",
            "threshold": "tau_best_val",
            "val_f1": "val_f1_best_val",
            "val_auc": "val_auc_best_val",
        }
    )

    best_oot = best_oot_df[
        ["curso", "config", "precision", "recall", "f1", "roc_auc", "threshold"]
    ].rename(
        columns={
            "config": "config_best_oot",
            "precision": "prec_best_oot",
            "recall": "recall_best_oot",
            "f1": "f1_best_oot",
            "roc_auc": "auc_best_oot",
            "threshold": "tau_best_oot",
        }
    )

    summary = baseline.merge(best_val, on="curso").merge(best_oot, on="curso")
    summary["delta_f1_best_val_vs_baseline"] = (summary["f1_best_val"] - summary["f1_baseline"]).round(3)
    summary["delta_f1_best_oot_vs_baseline"] = (summary["f1_best_oot"] - summary["f1_baseline"]).round(3)
    return summary


def main() -> None:
    args = parse_args()
    print("=== Tuning walk-forward S1 -> S2 ===")
    print("Cargando datos...")
    df_raw = load_all(DATA_DIR)
    df = asignar_generacion(limpiar(df_raw))
    df_features = build_features_s1(df)

    seed_configs = load_seed_configs(args.top_seeds_per_metric)

    all_rows: list[dict] = []
    best_val_rows: list[dict] = []
    best_oot_rows: list[dict] = []

    for curso in CURSOS_S2:
        configs = generate_local_configs(curso, seed_configs[curso], args.local_random_per_seed, args.seed)
        print(f"{curso}: {len(configs)} configuraciones enfocadas")
        rows, best_val_row, best_oot_row = evaluate_course(df_features, df, curso, configs)
        all_rows.extend(rows)
        if best_val_row is not None:
            best_val_rows.append(best_val_row)
            print(
                f"[BEST-VAL] {curso}: cfg={best_val_row['config']} val_f1={best_val_row['val_f1']} "
                f"F1_2023={best_val_row['f1']} AUC_2023={best_val_row['roc_auc']} tau={best_val_row['threshold']:.2f}"
            )
        if best_oot_row is not None:
            best_oot_rows.append(best_oot_row)
            print(
                f"[BEST-2023] {curso}: cfg={best_oot_row['config']} val_f1={best_oot_row['val_f1']} "
                f"F1_2023={best_oot_row['f1']} AUC_2023={best_oot_row['roc_auc']} tau={best_oot_row['threshold']:.2f}"
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    all_df = pd.DataFrame(all_rows)
    best_val_df = pd.DataFrame(best_val_rows)
    best_oot_df = pd.DataFrame(best_oot_rows)
    summary_df = build_report_summary(all_df, best_val_df, best_oot_df)

    all_path = RESULTS_DIR / "tuning_walkforward_s2_todos.csv"
    best_val_path = RESULTS_DIR / "tuning_walkforward_s2_mejores_validacion.csv"
    best_oot_path = RESULTS_DIR / "tuning_walkforward_s2_mejores_2023.csv"
    summary_path = RESULTS_DIR / "tuning_walkforward_s2_resumen.csv"

    all_df.to_csv(all_path, index=False)
    best_val_df.to_csv(best_val_path, index=False)
    best_oot_df.to_csv(best_oot_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\nResultados completos: {all_path}")
    print(f"Mejores por validacion 2022: {best_val_path}")
    print(f"Mejores por test 2023: {best_oot_path}")
    print(f"Resumen para informe: {summary_path}")


if __name__ == "__main__":
    main()
