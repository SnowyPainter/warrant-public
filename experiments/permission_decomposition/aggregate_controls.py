#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/permission_decomposition"


def collect_json(root: Path) -> list[dict]:
    rows = []
    for path in root.glob("**/final_metrics.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("status", "completed") == "completed" or "primary_metric" in row:
                rows.append(row)
        except Exception:
            pass
    return rows


def main() -> None:
    rows = collect_json(EXP / "outputs_full_controls")
    rows += collect_json(EXP / "outputs_placement_controls")
    rows += collect_json(EXP / "outputs_frozen_controls")
    groups = defaultdict(list)
    for row in rows:
        if "primary_metric" in row and all(row.get(key) is not None for key in ("domain", "dataset", "model", "variant")):
            groups[(row.get("domain"), row.get("dataset"), row.get("model"), row.get("variant"))].append(float(row["primary_metric"]))

    summary = []
    for key, values in sorted(groups.items()):
        summary.append({
            "domain": key[0], "dataset": key[1], "model": key[2], "variant": key[3],
            "seeds": len(values), "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    out = EXP / "outputs" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "all_control_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["domain", "dataset", "model", "variant", "seeds", "mean", "std"])
        writer.writeheader(); writer.writerows(summary)
    lines = ["# Additional Control Experiments", "", "All controls use the 65,536-example, 30-epoch main budget and seeds 7/17/37.", "", "| Domain | Dataset / model | Variant | n | Primary metric |", "|---|---|---|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['domain']} | {row['dataset']} / {row['model']} | {row['variant']} | {row['seeds']} | {row['mean']:.4f} ± {row['std']:.4f} |")
    (out / "all_control_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"collected {len(rows)} runs into {len(summary)} rows")


if __name__ == "__main__":
    main()
