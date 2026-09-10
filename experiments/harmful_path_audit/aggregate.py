#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/harmful_path_audit/outputs/results.csv"
OUT = ROOT / "experiments/harmful_path_audit/outputs/analysis"


def main() -> None:
    frame = pd.read_csv(RESULTS)
    frame = frame[frame.status.eq("completed")]
    open_rows = frame[frame.variant.eq("open_path_no_gate")].set_index("seed")
    full_rows = frame[frame.variant.eq("full_warrant")].set_index("seed")
    seeds = sorted(set(open_rows.index) & set(full_rows.index))
    paired = []
    for seed in seeds:
        o, f = open_rows.loc[seed], full_rows.loc[seed]
        paired.append({
            "seed": seed,
            "open_mrr": o.support_mrr,
            "full_mrr": f.support_mrr,
            "permission_delta_mrr": f.support_mrr - o.support_mrr,
            "open_gold_attention": o.gold_attention_mass,
            "open_hard_attention": o.hard_attention_mass,
            "open_random_attention": o.random_attention_mass,
            "full_gold_gate": f.gold_gate_mean,
            "full_hard_gate": f.hard_gate_mean,
            "full_random_gate": f.random_gate_mean,
            "full_gold_hard_effective_ratio": f.gold_hard_effective_ratio,
            "full_gold_random_effective_ratio": f.gold_random_effective_ratio,
            "full_unsupported": f.unsupported_at_gold_count,
            "open_unsupported": o.unsupported_at_gold_count,
        })
    result = pd.DataFrame(paired)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "paired_seed_audit.csv", index=False)
    mean = result.mean(numeric_only=True)
    wins = int((result.permission_delta_mrr > 0).sum())
    conditions = {
        "open_exposes_hard_mass": bool(mean.open_hard_attention > mean.open_gold_attention),
        "open_exposes_random_mass": bool(mean.open_random_attention > mean.open_gold_attention),
        "gate_suppresses_hard": bool(mean.full_hard_gate < mean.full_gold_gate),
        "gate_suppresses_random": bool(mean.full_random_gate < mean.full_gold_gate),
        "full_improves_mean_mrr": bool(mean.permission_delta_mrr > 0),
    }
    lines = [
        "# Harmful OpenPath → Selective Suppression → Recovery Audit", "",
        f"Matched seeds: {len(result)}; Full wins: {wins}/{len(result)}.", "",
        "| Test | Result |", "|---|---:|",
        f"| OpenPath hard attention > gold | {conditions['open_exposes_hard_mass']} ({mean.open_hard_attention:.4f} vs {mean.open_gold_attention:.4f}) |",
        f"| OpenPath random attention > gold | {conditions['open_exposes_random_mass']} ({mean.open_random_attention:.4f} vs {mean.open_gold_attention:.4f}) |",
        f"| Full hard gate < gold gate | {conditions['gate_suppresses_hard']} ({mean.full_hard_gate:.4f} vs {mean.full_gold_gate:.4f}) |",
        f"| Full random gate < gold gate | {conditions['gate_suppresses_random']} ({mean.full_random_gate:.4f} vs {mean.full_gold_gate:.4f}) |",
        f"| Full − OpenPath MRR | {mean.permission_delta_mrr:+.4f} |",
        f"| Unsupported fraction change | {mean.full_unsupported-mean.open_unsupported:+.4f} |",
        "", "## Decision", "",
        "All five pre-registered conditions must be reported; a selective-suppression claim is not made if the mean recovery condition fails.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
