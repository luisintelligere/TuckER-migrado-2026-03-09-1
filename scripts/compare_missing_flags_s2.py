from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tuning_temporal_walkforward_s2 import BASELINE_CONFIG, evaluate_course as evaluate_walkforward_course
from validacion_temporal import (
    CURSOS_S1,
    CURSOS_S2,
    DATA_DIR,
    FAIL_GRADE,
    RESULTS_DIR,
    asignar_generacion,
    load_all,
    limpiar,
)
from validacion_temporal_s2 import (
    build_features_s1,
    experimento_expandido,
    experimento_out_of_time,
)


def build_features_s1_only_grades(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[df["CURSO"].isin(CURSOS_S1)].copy()
    notas = rows.pivot_table(index="ID", columns="CURSO", values="NOTA", aggfunc="max").reset_index()

    for curso in CURSOS_S1:
        if curso not in notas.columns:
            notas[curso] = np.nan

    notas[CURSOS_S1] = notas[CURSOS_S1].fillna(FAIL_GRADE)
    return notas[["ID"] + CURSOS_S1]


def evaluate_walkforward_baseline(df_features: pd.DataFrame, df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for curso in CURSOS_S2:
        course_rows, best_val_row, _ = evaluate_walkforward_course(
            df_features=df_features,
            df=df,
            curso=curso,
            configs=[dict(BASELINE_CONFIG)],
        )
        if not course_rows or best_val_row is None:
            continue

        row = dict(best_val_row)
        row["experimento"] = "walkforward_base"
        rows.append(row)
    return rows


def run_variant(variant: str, df_features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    expanded_rows = experimento_expandido(df_features, df)
    oot_rows = experimento_out_of_time(df_features, df)
    wf_rows = evaluate_walkforward_baseline(df_features, df)

    all_rows = pd.DataFrame(expanded_rows + oot_rows + wf_rows)
    all_rows["variant"] = variant
    return all_rows


def build_comparison(all_results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "threshold",
        "val_f1",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "balanced_acc",
        "fp",
        "fn",
        "tp",
        "tn",
        "n",
        "reprobados",
    ]

    with_flags = all_results[all_results["variant"] == "with_flags"][["experimento", "curso"] + metric_cols].rename(
        columns={col: f"{col}_with_flags" for col in metric_cols}
    )
    without_flags = all_results[all_results["variant"] == "without_flags"][["experimento", "curso"] + metric_cols].rename(
        columns={col: f"{col}_without_flags" for col in metric_cols}
    )

    comparison = with_flags.merge(without_flags, on=["experimento", "curso"], how="inner")
    for metric in ("precision", "recall", "f1", "roc_auc", "balanced_acc"):
        comparison[f"delta_{metric}"] = (
            comparison[f"{metric}_without_flags"] - comparison[f"{metric}_with_flags"]
        ).round(3)
    return comparison.sort_values(["experimento", "curso"]).reset_index(drop=True)


def main() -> None:
    print("=== S1 -> S2 ablation: with vs without missing flags ===")
    df_raw = load_all(DATA_DIR)
    df = asignar_generacion(limpiar(df_raw))

    print("Building feature matrices...")
    features_with_flags = build_features_s1(df)
    features_without_flags = build_features_s1_only_grades(df)

    print("Running baseline experiments WITH missing flags...")
    with_flags = run_variant("with_flags", features_with_flags, df)

    print("Running baseline experiments WITHOUT missing flags...")
    without_flags = run_variant("without_flags", features_without_flags, df)

    all_results = pd.concat([with_flags, without_flags], ignore_index=True)
    comparison = build_comparison(all_results)

    RESULTS_DIR.mkdir(exist_ok=True)
    raw_path = RESULTS_DIR / "ablation_missing_flags_s2_raw.csv"
    comparison_path = RESULTS_DIR / "ablation_missing_flags_s2_comparison.csv"
    all_results.to_csv(raw_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    print(f"Raw results saved to {raw_path}")
    print(f"Comparison saved to {comparison_path}")
    print("\n=== DELTA (without - with) ===")
    display_cols = [
        "experimento",
        "curso",
        "f1_with_flags",
        "f1_without_flags",
        "delta_f1",
        "roc_auc_with_flags",
        "roc_auc_without_flags",
        "delta_roc_auc",
    ]
    print(comparison[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()