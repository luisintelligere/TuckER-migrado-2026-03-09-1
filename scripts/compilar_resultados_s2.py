"""Compila los mejores resultados (por val_mrr) de cada run S2 y genera un resumen."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUTPUT_CSV = RESULTS_DIR / "resumen_s2_mejores_por_val_mrr.csv"

DATASETS = [
    "dataset_2021_hacia_s2_binario",
    "dataset_2020_2021_hacia_s2_binario",
    "dataset_2019_2020_2021_hacia_s2_binario",
]
RDIMS = range(1, 13)


def main() -> None:
    rows = []
    for ds in DATASETS:
        for rdim in RDIMS:
            folder = RESULTS_DIR / f"s2_{ds}_rdim{rdim}"
            csv_path = folder / "training_metrics.csv"
            if not csv_path.exists():
                print(f"FALTA: {csv_path}")
                continue

            df = pd.read_csv(csv_path)

            # Mejor epoch por val_mrr
            best_val_idx = df["val_mrr"].idxmax()
            best_val_row = df.loc[best_val_idx]

            # Si el best epoch no tiene test (epoch impar), buscar el epoch par más cercano
            if pd.isna(best_val_row.get("test_mrr")):
                best_epoch = int(best_val_row["epoch"])
                # Buscar el epoch par más cercano con test evaluado
                df_with_test = df.dropna(subset=["test_mrr"])
                if len(df_with_test) > 0:
                    closest_idx = (df_with_test["epoch"] - best_epoch).abs().idxmin()
                    test_row = df_with_test.loc[closest_idx]
                else:
                    test_row = best_val_row
            else:
                test_row = best_val_row

            rows.append({
                "dataset": ds,
                "rdim": rdim,
                "best_epoch": int(best_val_row["epoch"]),
                "loss": best_val_row["loss"],
                "val_mrr": best_val_row["val_mrr"],
                "val_hits@1": best_val_row["val_hits@1"],
                "val_hits@3": best_val_row["val_hits@3"],
                "val_hits@10": best_val_row["val_hits@10"],
                "val_mr": best_val_row["val_mr"],
                "test_epoch": int(test_row["epoch"]),
                "test_mrr": test_row.get("test_mrr"),
                "test_hits@1": test_row.get("test_hits@1"),
                "test_hits@3": test_row.get("test_hits@3"),
                "test_hits@10": test_row.get("test_hits@10"),
                "test_mr": test_row.get("test_mr"),
            })

    resumen = pd.DataFrame(rows)
    resumen.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResumen guardado en: {OUTPUT_CSV}\n")

    # Mostrar tabla resumen
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)

    for ds in DATASETS:
        sub = resumen[resumen["dataset"] == ds]
        print(f"\n{'='*80}")
        print(f"  {ds}")
        print(f"{'='*80}")
        cols = ["rdim", "best_epoch", "val_mrr", "val_hits@1", "val_hits@10", "test_mrr", "test_hits@1", "test_hits@10"]
        print(sub[cols].to_string(index=False))
        best = sub.loc[sub["val_mrr"].idxmax()]
        print(f"\n  >>> Mejor rdim={int(best['rdim'])} (epoch {int(best['best_epoch'])}): val_mrr={best['val_mrr']:.4f}, test_mrr={best['test_mrr']:.4f}")


if __name__ == "__main__":
    main()
