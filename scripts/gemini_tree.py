from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


DEFAULT_SEM1_FILES = ["df_20191.csv", "df_20201.csv", "df_20211.csv"]
DEFAULT_SEM2_FILES = ["df_20192.csv", "df_20202.csv", "df_20212.csv"]
DEFAULT_CURSOS_S1 = ["MA1101", "MA1001", "FI1000", "BT1211"]
DEFAULT_CURSOS_S2 = ["CC1002", "MA1002", "MA1102", "FI1100"]
FAIL_GRADE = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara RandomForest, XGBoost y LightGBM para predecir reprobación en cursos del segundo semestre."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "dataframes_por_semestre",
        help="Directorio donde viven los archivos df_*.csv.",
    )
    parser.add_argument("--sem1-files", nargs="+", default=DEFAULT_SEM1_FILES)
    parser.add_argument("--sem2-files", nargs="+", default=DEFAULT_SEM2_FILES)
    parser.add_argument("--courses-s1", nargs="+", default=DEFAULT_CURSOS_S1)
    parser.add_argument("--courses-s2", nargs="+", default=DEFAULT_CURSOS_S2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument(
        "--threshold-metric",
        choices=["f1", "recall", "balanced_accuracy"],
        default="f1",
        help="Métrica usada para elegir el threshold sobre validación.",
    )
    parser.add_argument(
        "--save-report",
        type=Path,
        default=Path("results") / "gemini_tree_model_comparison.csv",
        help="Ruta donde guardar el resumen CSV.",
    )
    return parser.parse_args()


def load_semester_files(data_dir: Path, file_names: list[str]) -> pd.DataFrame:
    missing_files = [file_name for file_name in file_names if not (data_dir / file_name).exists()]
    if missing_files:
        missing_str = ", ".join(missing_files)
        raise FileNotFoundError(f"No se encontraron estos archivos en {data_dir}: {missing_str}")

    frames = [pd.read_csv(data_dir / file_name, sep=";", encoding="utf-8") for file_name in file_names]
    return pd.concat(frames, ignore_index=True)


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["ESTADO_CURSO"] = cleaned["ESTADO_CURSO"].astype(str).str.strip()
    cleaned = cleaned[cleaned["ESTADO_CURSO"].str.contains("Aprobado|Reprobado", na=False)].copy()

    cleaned["NOTA"] = pd.to_numeric(
        cleaned["NOTA"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    # Regla pedida: si la nota falta, la consideramos como reprobado.
    missing_grade_mask = cleaned["NOTA"].isna()
    cleaned.loc[missing_grade_mask, "NOTA"] = FAIL_GRADE
    cleaned["NOTA_FALTANTE"] = missing_grade_mask.astype(int)
    cleaned["ES_REPROBADO"] = cleaned["ESTADO_CURSO"].str.contains("Reprobado", na=False).astype(int)
    return cleaned


def prepare_feature_matrix(df_sem1: pd.DataFrame, cursos_s1: list[str]) -> pd.DataFrame:
    feature_rows = df_sem1[df_sem1["CURSO"].isin(cursos_s1)].copy()
    feature_rows["NOTA_AJUSTADA"] = feature_rows["NOTA"]

    df_s1_pivot = (
        feature_rows.pivot_table(index="ID", columns="CURSO", values="NOTA_AJUSTADA", aggfunc="max")
        .reset_index()
    )
    missing_flags = (
        feature_rows.pivot_table(index="ID", columns="CURSO", values="NOTA_FALTANTE", aggfunc="max")
        .reset_index()
    )

    for curso in cursos_s1:
        if curso not in df_s1_pivot.columns:
            df_s1_pivot[curso] = np.nan
        if curso not in missing_flags.columns:
            missing_flags[curso] = np.nan

    # Si el curso predictor no tiene nota registrada, se codifica como reprobado.
    df_s1_pivot[cursos_s1] = df_s1_pivot[cursos_s1].fillna(FAIL_GRADE)
    missing_columns = {curso: f"{curso}_nota_faltante" for curso in cursos_s1}
    missing_flags = missing_flags.rename(columns=missing_columns)
    missing_flag_columns = list(missing_columns.values())
    missing_flags[missing_flag_columns] = missing_flags[missing_flag_columns].fillna(1).astype(int)

    return df_s1_pivot[["ID"] + cursos_s1].merge(missing_flags[["ID"] + missing_flag_columns], on="ID", how="left")


def build_target_frame(df_sem2: pd.DataFrame, curso_objetivo: str) -> pd.DataFrame:
    df_target = df_sem2[df_sem2["CURSO"] == curso_objetivo].copy()
    df_target["TARGET"] = np.where(df_target["ESTADO_CURSO"].str.contains("Reprobado", na=False), 1, 0)
    df_target = df_target.sort_values("SEMESTRE").groupby("ID").first().reset_index()
    return df_target[["ID", "TARGET"]]


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray, metric_name: str) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.linspace(0.05, 0.95, 19):
        predictions = (probabilities >= threshold).astype(int)
        if metric_name == "recall":
            score = recall_score(y_true, predictions, zero_division=0)
        elif metric_name == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, predictions)
        else:
            score = f1_score(y_true, predictions, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score


def compute_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float | int | str]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "test_accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision_reprueba": precision_score(y_true, predictions, zero_division=0),
        "recall_reprueba": recall_score(y_true, predictions, zero_division=0),
        "f1_reprueba": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_models(random_state: int, n_estimators: int, pos_weight: float) -> dict[str, object]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
            min_samples_leaf=3,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=pos_weight,
            random_state=random_state,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=pos_weight,
            random_state=random_state,
            force_col_wise=True,
            n_jobs=1,
            verbosity=-1,
        ),
    }


def feature_importance_summary(model: object, feature_names: list[str]) -> str:
    if not hasattr(model, "feature_importances_"):
        return "n/a"
    importances = dict(zip(feature_names, model.feature_importances_))
    return max(importances, key=importances.get)


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    data_dir = args.data_dir.resolve()
    print(f"Usando data_dir: {data_dir}")
    print(f"Inputs semestre 1: {', '.join(args.sem1_files)}")
    print(f"Inputs semestre 2: {', '.join(args.sem2_files)}")
    print(f"Cursos predictoras: {', '.join(args.courses_s1)}")
    print(f"Cursos objetivo: {', '.join(args.courses_s2)}")
    print(f"Regla activa: nota faltante => {FAIL_GRADE} (reprobado)\n")

    df_sem1 = limpiar_datos(load_semester_files(data_dir, args.sem1_files))
    df_sem2 = limpiar_datos(load_semester_files(data_dir, args.sem2_files))
    df_s1_pivot = prepare_feature_matrix(df_sem1, args.courses_s1)
    feature_columns = [column for column in df_s1_pivot.columns if column != "ID"]

    results: list[dict[str, object]] = []

    for curso_objetivo in args.courses_s2:
        print("=" * 70)
        print(f"CURSO OBJETIVO: {curso_objetivo}")
        print("=" * 70)

        df_target = build_target_frame(df_sem2, curso_objetivo)
        df_model = pd.merge(df_s1_pivot, df_target, on="ID", how="inner")

        if len(df_model) < args.min_samples:
            print(f"Muestra insuficiente para {curso_objetivo}: {len(df_model)} alumnos. Saltando...\n")
            continue

        X = df_model[feature_columns]
        y = df_model["TARGET"]
        if len(y.unique()) < 2:
            print(f"No hay suficientes casos de ambas clases para {curso_objetivo}. Saltando...\n")
            continue

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=args.validation_size,
            random_state=args.random_state,
            stratify=y_train_full,
        )

        positives = int(y_train.sum())
        negatives = int((y_train == 0).sum())
        pos_weight = negatives / positives if positives else 1.0

        print(f"Muestra total: {len(df_model)} | Reprobados: {int(y.sum())} ({(y.sum() / len(y)):.1%})")
        print(f"Train: {len(X_train)} | Validación: {len(X_val)} | Test: {len(X_test)}")

        for model_name, model in build_models(args.random_state, args.n_estimators, pos_weight).items():
            model.fit(X_train, y_train)
            val_probabilities = model.predict_proba(X_val)[:, 1]
            threshold, validation_score = choose_threshold(y_val, val_probabilities, args.threshold_metric)
            test_probabilities = model.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, test_probabilities, threshold)

            print(f"\nModelo: {model_name}")
            print(f"Threshold óptimo ({args.threshold_metric} en validación): {threshold:.2f} | Score val: {validation_score:.4f}")
            print(
                classification_report(
                    y_test,
                    (test_probabilities >= threshold).astype(int),
                    target_names=["Aprueba (0)", "Reprueba (1)"],
                    zero_division=0,
                )
            )

            results.append(
                {
                    "curso_objetivo": curso_objetivo,
                    "modelo": model_name,
                    "muestra": len(df_model),
                    "reprobados": int(y.sum()),
                    "validation_metric": args.threshold_metric,
                    "validation_score": validation_score,
                    "feature_importance_top": feature_importance_summary(model, feature_columns),
                    **metrics,
                }
            )

        print()

    return pd.DataFrame(results)


def main() -> None:
    args = parse_args()
    results_df = run_experiment(args)
    if results_df.empty:
        print("No se pudo entrenar ningún modelo con los parámetros seleccionados.")
        return

    print("=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(results_df.to_string(index=False))

    save_path = args.save_report.resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(save_path, index=False)
    print(f"\nResumen guardado en: {save_path}")


if __name__ == "__main__":
    main()
