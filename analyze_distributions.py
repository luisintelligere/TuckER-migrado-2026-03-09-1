#!/usr/bin/env python3
"""Analyze approval/reprobation distributions in dataset splits.

Outputs CSV summaries and bar charts for:
- test.txt
- train_original.txt
- valid.txt
- train.txt

Usage:
  python analyze_distributions.py --data-dir "C:\\Users\\56946\\TuckER\\data\\dataset_2019_2020_2021_multirelacional_4d"
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False


@dataclass
class FileStats:
    file_name: str
    row_counts_by_outcome: Counter
    row_counts_by_label: Counter
    unique_students_by_outcome: Dict[str, int]
    total_rows: int
    total_unique_students: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze dataset distribution by approval ranges.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to dataset folder containing the .txt files.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["test.txt", "train_original.txt", "valid.txt", "train.txt"],
        help="Files to analyze (default: test.txt train_original.txt valid.txt train.txt).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <data-dir>/analysis_outputs.",
    )
    return parser.parse_args()


def outcome_from_relation(relation: str) -> str:
    if relation.startswith("aprueba_"):
        return relation
    return "reprueba"


def outcome_sort_key(outcome: str) -> Tuple[int, int, int, str]:
    if outcome == "reprueba":
        return (1, 0, 0, outcome)
    # aprueba_x_y
    parts = outcome.split("_")
    if len(parts) == 3:
        try:
            low = int(parts[1])
            high = int(parts[2])
            return (0, low, high, outcome)
        except ValueError:
            return (0, 0, 0, outcome)
    return (0, 0, 0, outcome)


def read_file_stats(file_path: Path) -> FileStats:
    row_counts_by_outcome: Counter = Counter()
    row_counts_by_label: Counter = Counter()
    students_by_outcome: Dict[str, set] = defaultdict(set)

    total_rows = 0
    all_students = set()

    with file_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or len(row) < 3:
                continue
            student, relation, _ = row[0], row[1], row[2]
            outcome = outcome_from_relation(relation)
            row_counts_by_outcome[outcome] += 1
            row_counts_by_label[relation] += 1
            students_by_outcome[outcome].add(student)
            all_students.add(student)
            total_rows += 1

    unique_students_by_outcome = {k: len(v) for k, v in students_by_outcome.items()}

    return FileStats(
        file_name=file_path.name,
        row_counts_by_outcome=row_counts_by_outcome,
        row_counts_by_label=row_counts_by_label,
        unique_students_by_outcome=unique_students_by_outcome,
        total_rows=total_rows,
        total_unique_students=len(all_students),
    )


def write_summary_csv(out_dir: Path, stats: List[FileStats]) -> None:
    summary_path = out_dir / "summary_counts.csv"
    label_path = out_dir / "label_counts.csv"

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "file",
                "outcome",
                "row_count",
                "unique_students",
                "total_rows",
                "total_unique_students",
            ]
        )
        for file_stats in stats:
            outcomes = sorted(file_stats.row_counts_by_outcome.keys(), key=outcome_sort_key)
            for outcome in outcomes:
                writer.writerow(
                    [
                        file_stats.file_name,
                        outcome,
                        file_stats.row_counts_by_outcome.get(outcome, 0),
                        file_stats.unique_students_by_outcome.get(outcome, 0),
                        file_stats.total_rows,
                        file_stats.total_unique_students,
                    ]
                )

    with label_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "label", "row_count"])
        for file_stats in stats:
            for label, count in file_stats.row_counts_by_label.most_common():
                writer.writerow([file_stats.file_name, label, count])


def plot_counts(out_dir: Path, stats: FileStats) -> None:
    if not HAS_MPL:
        return

    outcomes = sorted(stats.row_counts_by_outcome.keys(), key=outcome_sort_key)
    row_counts = [stats.row_counts_by_outcome[o] for o in outcomes]
    unique_counts = [stats.unique_students_by_outcome.get(o, 0) for o in outcomes]

    plt.figure(figsize=(10, 6))
    plt.bar(outcomes, row_counts, color="#4c78a8")
    plt.title(f"Row counts by outcome - {stats.file_name}")
    plt.xlabel("Outcome")
    plt.ylabel("Row count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stats.file_name}_rows.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(outcomes, unique_counts, color="#f58518")
    plt.title(f"Unique students by outcome - {stats.file_name}")
    plt.xlabel("Outcome")
    plt.ylabel("Unique students")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stats.file_name}_students.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir
    out_dir = args.out_dir or (data_dir / "analysis_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_list: List[FileStats] = []
    for file_name in args.files:
        file_path = data_dir / file_name
        if not file_path.exists():
            print(f"[WARN] Missing file: {file_path}")
            continue
        stats = read_file_stats(file_path)
        stats_list.append(stats)
        plot_counts(out_dir, stats)

    if stats_list:
        write_summary_csv(out_dir, stats_list)

    if not HAS_MPL:
        print("[WARN] matplotlib not installed. Skipping plots.")

    print(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
