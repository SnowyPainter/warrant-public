#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.marginal_value_probe.run import aggregate, report, write_csv


def main() -> None:
    root = Path(__file__).resolve().parent / "outputs"
    with (root / "results.csv").open("r", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            converted = dict(row)
            for key in ("seed", "train_examples", "test_examples", "test_rows", "parameters"):
                converted[key] = int(converted[key])
            for key in ("rmse", "mae", "r2", "pearson", "spearman", "sign_accuracy", "harmful_auc"):
                converted[key] = float(converted[key])
            rows.append(converted)
    summary = aggregate(rows)
    write_csv(root / "summary.csv", summary)
    text = report(summary)
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    (root / "analysis" / "frozen_probe_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
