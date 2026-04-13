"""
Análisis de falsos positivos para el tuning walk-forward S1 → S2.

Reconstruye los mejores modelos seleccionados y genera:
  1. CSV detalle de cada falso positivo (notas S1, probabilidad, vulnerabilidad).
  2. CSV resumen por curso.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tuning_temporal_walkforward_s2 import (
    TEST_GEN,
    TRAIN_GENS,
    VALID_GEN,
    build_model,
    normalize_config_row,
)
from validacion_temporal import (
    CURSOS_S1,
    CURSOS_S2,
    DATA_DIR,
    RESULTS_DIR,
    asignar_generacion,
    build_target,
    load_all,
    limpiar,
    metricas,
)
from validacion_temporal_s2 import build_features_s1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruye los mejores modelos del tuning walk-forward S1->S2 y genera "
            "un reporte de falsos positivos y vulnerabilidad sobre 2023."
        )
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=RESULTS_DIR / "tuning_walkforward_s2_mejores_validacion.csv",
    )
    parser.add_argument(
        "--selection-label",
        type=str,
        default="mejor_validacion_2022",
    )
    parser.add_argument(
        "--output-fp",
        type=Path,
        default=RESULTS_DIR / "tuning_walkforward_s2_falsos_positivos_validacion.csv",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=RESULTS_DIR / "tuning_walkforward_s2_falsos_positivos_resumen_validacion.csv",
    )
    return parser.parse_args()


def first_course_record(historial: pd.DataFrame, curso: str) -> pd.Series | None:
    datos_curso = historial[historial["CURSO"] == curso].sort_values("SEMESTRE")
    if datos_curso.empty:
        return None
    return datos_curso.iloc[0]


def analyze_false_positives(
    fp_ids: list,
    df_historico: pd.DataFrame,
    curso_target: str,
    probabilities_map: dict,
    config_label: str,
    threshold: float,
    selection_label: str,
) -> pd.DataFrame:
    registros: list[dict[str, object]] = []

    for alumno_id in fp_ids:
        historial = df_historico[df_historico["ID"] == alumno_id]
        row: dict[str, object] = {
            "ID": alumno_id,
            "generacion_test": TEST_GEN,
            "curso_objetivo_s2": curso_target,
            "criterio_seleccion": selection_label,
            "config": config_label,
            "threshold": threshold,
            "prob_reprobacion": probabilities_map.get(alumno_id, np.nan),
        }

        # Solo notas de S1 como features de contexto
        for curso in CURSOS_S1:
            registro = first_course_record(historial, curso)
            if registro is None:
                row[f"{curso}_nota"] = np.nan
                row[f"{curso}_estado"] = "Sin registro"
            else:
                row[f"{curso}_nota"] = registro["NOTA"]
                row[f"{curso}_estado"] = registro["ESTADO_CURSO"]

        # Resultado real en el curso S2 objetivo
        registro_s2 = first_course_record(historial, curso_target)
        if registro_s2 is None:
            row["nota_real_s2"] = np.nan
            row["estado_real_s2"] = "Sin registro"
        else:
            row["nota_real_s2"] = registro_s2["NOTA"]
            row["estado_real_s2"] = registro_s2["ESTADO_CURSO"]

        notas_s1 = [row.get(f"{c}_nota") for c in CURSOS_S1 if pd.notna(row.get(f"{c}_nota"))]
        notas_s1_numeric = [n for n in notas_s1 if isinstance(n, (int, float))]

        row["promedio_s1"] = float(np.mean(notas_s1_numeric)) if notas_s1_numeric else np.nan

        reprobados_s1 = sum(
            1
            for c in CURSOS_S1
            if "Reprobado" in str(row.get(f"{c}_estado", ""))
        )
        sin_registro = sum(
            1 for c in CURSOS_S1 if row.get(f"{c}_estado") == "Sin registro"
        )
        row["cursos_reprobados_s1"] = reprobados_s1
        row["cursos_sin_registro"] = sin_registro
        # Vulnerabilidad basada solo en S1
        row["vulnerable"] = "Si" if (
            reprobados_s1 >= 2
            or (notas_s1_numeric and np.mean(notas_s1_numeric) < 4.5)
        ) else "No"

        registros.append(row)

    return pd.DataFrame(registros)


def build_summary_row(
    curso: str,
    config_label: str,
    threshold: float,
    selection_label: str,
    fp_df: pd.DataFrame,
) -> dict[str, object]:
    if fp_df.empty:
        return {
            "curso_objetivo_s2": curso,
            "criterio_seleccion": selection_label,
            "config": config_label,
            "threshold": threshold,
            "falsos_positivos": 0,
            "vulnerables": 0,
            "porcentaje_vulnerables": 0.0,
            "prob_reprobacion_media": np.nan,
            "nota_real_media_s2": np.nan,
            "promedio_s1": np.nan,
            "reprobados_s1_promedio": np.nan,
        }

    n_total = len(fp_df)
    n_vulnerables = int((fp_df["vulnerable"] == "Si").sum())
    return {
        "curso_objetivo_s2": curso,
        "criterio_seleccion": selection_label,
        "config": config_label,
        "threshold": threshold,
        "falsos_positivos": n_total,
        "vulnerables": n_vulnerables,
        "porcentaje_vulnerables": round(n_vulnerables / n_total * 100.0, 1),
        "prob_reprobacion_media": round(float(fp_df["prob_reprobacion"].mean()), 3),
        "nota_real_media_s2": round(float(fp_df["nota_real_s2"].mean()), 3),
        "promedio_s1": round(float(fp_df["promedio_s1"].mean()), 3),
        "reprobados_s1_promedio": round(float(fp_df["cursos_reprobados_s1"].mean()), 3),
    }


def warn_if_metrics_do_not_match(curso: str, stored_row: dict, recomputed_row: dict) -> None:
    mismatches: list[str] = []
    for campo in ("precision", "recall", "f1", "roc_auc"):
        if abs(float(stored_row[campo]) - float(recomputed_row[campo])) > 0.002:
            mismatches.append(f"{campo}: esperado={stored_row[campo]} recreado={recomputed_row[campo]}")
    for campo in ("tn", "fp", "fn", "tp"):
        if int(stored_row[campo]) != int(recomputed_row[campo]):
            mismatches.append(f"{campo}: esperado={stored_row[campo]} recreado={recomputed_row[campo]}")

    if mismatches:
        print(f"[ADVERTENCIA] {curso}: la reconstruccion no reprodujo exactamente el CSV de seleccion.")
        for mismatch in mismatches:
            print(f"  - {mismatch}")


def evaluate_selected_model(
    df_features: pd.DataFrame,
    df: pd.DataFrame,
    selected_row: dict,
    selection_label: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    curso = str(selected_row["curso"])
    config = normalize_config_row(selected_row, label=str(selected_row["config"]))
    threshold = float(selected_row["threshold"])

    df_target = build_target(df, curso)
    df_train_t = df_target[df_target["generacion"].isin(TRAIN_GENS)]
    df_test_t = df_target[df_target["generacion"] == TEST_GEN]

    dm_train = pd.merge(df_features, df_train_t, on="ID", how="inner")
    dm_test = pd.merge(df_features, df_test_t, on="ID", how="inner")
    if min(len(dm_train), len(dm_test)) < 10:
        raise ValueError(f"{curso}: muestra insuficiente para reconstruir el modelo.")

    feat_cols = [c for c in dm_train.columns if c not in ("ID", "TARGET", "generacion")]
    X_train = dm_train[feat_cols]
    y_train = dm_train["TARGET"]
    X_test = dm_test[feat_cols]
    y_test = dm_test["TARGET"]
    ids_test = dm_test["ID"].tolist()

    scale_pos_weight = (y_train == 0).sum() / max(y_train.sum(), 1)
    model = build_model(config, float(scale_pos_weight))
    model.fit(X_train, y_train)

    test_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (test_proba >= threshold).astype(int)
    recomputed_row = metricas(y_test, y_pred, test_proba)
    warn_if_metrics_do_not_match(curso, selected_row, recomputed_row)

    fp_mask = (y_pred == 1) & (y_test.to_numpy() == 0)
    fp_ids = [ids_test[idx] for idx, is_fp in enumerate(fp_mask) if is_fp]
    fp_probas = {ids_test[idx]: float(test_proba[idx]) for idx, is_fp in enumerate(fp_mask) if is_fp}
    fp_df = analyze_false_positives(
        fp_ids=fp_ids,
        df_historico=df,
        curso_target=curso,
        probabilities_map=fp_probas,
        config_label=config["label"],
        threshold=threshold,
        selection_label=selection_label,
    )
    summary_row = build_summary_row(curso, config["label"], threshold, selection_label, fp_df)
    return fp_df, summary_row


def main() -> None:
    args = parse_args()
    selection_path = args.selection_file.resolve()
    print(f"=== Analisis FP S1 -> S2 ===")
    print(f"Leyendo seleccion desde: {selection_path}")
    print(f"Criterio de seleccion: {args.selection_label}")

    selected_df = pd.read_csv(selection_path)
    if selected_df.empty:
        raise ValueError(f"El archivo {selection_path} no contiene filas.")

    print("Cargando datos historicos para reconstruir modelos...")
    df_raw = load_all(DATA_DIR)
    df = asignar_generacion(limpiar(df_raw))
    df_features = build_features_s1(df)

    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for selected_row in selected_df.to_dict("records"):
        curso = selected_row["curso"]
        print(f"\n[{curso}] reconstruyendo modelo seleccionado...")
        fp_df, summary_row = evaluate_selected_model(df_features, df, selected_row, args.selection_label)
        if not fp_df.empty:
            detail_frames.append(fp_df)
        summary_rows.append(summary_row)
        print(
            f"[{curso}] FP={summary_row['falsos_positivos']} | "
            f"vulnerables={summary_row['vulnerables']} | "
            f"threshold={summary_row['threshold']:.2f}"
        )

    output_fp = args.output_fp.resolve()
    output_summary = args.output_summary.resolve()
    output_fp.parent.mkdir(parents=True, exist_ok=True)

    if detail_frames:
        fp_all = pd.concat(detail_frames, ignore_index=True)
        fp_all.to_csv(output_fp, index=False)
        print(f"\nDetalle FP guardado en {output_fp}")
    else:
        print("\nNo se encontraron falsos positivos en ningún curso.")

    pd.DataFrame(summary_rows).to_csv(output_summary, index=False)
    print(f"Resumen FP guardado en {output_summary}")


if __name__ == "__main__":
    main()
