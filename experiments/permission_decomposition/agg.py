#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC = ROOT / "experiments/permission_decomposition/outputs/ctdg/lastfm/DyGFormer"
MAIN_OPEN = ROOT / "experiments/permission_decomposition/outputs_main_budget/ctdg/lastfm/DyGFormer/open_path_no_gate"
MAIN_EXISTING = ROOT / "experiments/main_benchmark/outputs/DyGFormer"
OUTPUT = ROOT / "experiments/permission_decomposition/outputs/analysis/permission_decomposition_report.md"


def read_variant(root: Path, variant: str) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for path in (root / variant).glob("seed_*/final_metrics.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") == "completed":
            rows[int(row["seed"])] = row
    return rows


def mean_std(values: list[float]) -> str:
    if not values:
        return ""
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.4f} ± {sd:.4f}"


def main() -> None:
    variants = [
        "base",
        "open_path_no_gate",
        "scalar_gate",
        "item_only_gate",
        "normalized_gate",
        "correct_path_warrant",
        "generic_open_path",
        "combined_full",
    ]
    lines = [
        "# Permission Decomposition Results",
        "",
        "## Controlled CTDG factorial (32,768 examples, 10 epochs)",
        "",
        "All rows below were trained afresh with the same split, optimizer, example budget, and seeds.",
        "The interrupted run left a subset of seeds for some variants; seed counts are reported explicitly.",
        "",
        "| Variant | Seeds | AUC | Δ vs OpenPath | Gate mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    open_rows = read_variant(DIAGNOSTIC, "open_path_no_gate")
    open_mean = statistics.mean(float(row["auc"]) for row in open_rows.values())
    for variant in variants:
        rows = read_variant(DIAGNOSTIC, variant)
        if not rows:
            continue
        aucs = [float(row["auc"]) for row in rows.values()]
        gates = [float(row["edge_warrant_gate_mean"]) for row in rows.values() if row.get("edge_warrant_gate_mean") is not None]
        lines.append(
            f"| {variant} | {len(rows)} | {mean_std(aucs)} | {statistics.mean(aucs)-open_mean:+.4f} | {mean_std(gates)} |"
        )

    main_open: dict[int, float] = {}
    for path in MAIN_OPEN.glob("seed_*/final_metrics.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        main_open[int(row["seed"])] = float(row["auc"])
    main_base: dict[int, float] = {}
    main_full: dict[int, float] = {}
    for seed_dir in MAIN_EXISTING.iterdir():
        if not seed_dir.is_dir() or not seed_dir.name.isdigit():
            continue
        seed = int(seed_dir.name)
        for variant, target in (("base", main_base), ("warrant", main_full)):
            path = seed_dir / "ctdg/lastfm" / variant / "metrics.json"
            if path.exists():
                target[seed] = float(json.loads(path.read_text(encoding="utf-8"))["auc"])
    seeds = sorted(set(main_open) & set(main_base) & set(main_full))
    base_values = [main_base[s] for s in seeds]
    open_values = [main_open[s] for s in seeds]
    full_values = [main_full[s] for s in seeds]
    permission_delta = [main_full[s] - main_open[s] for s in seeds]
    lines.extend(
        [
            "",
            "## Main-benchmark budget CTDG decomposition (65,536 examples, 30 epochs)",
            "",
            "| Seed | Base | OpenPath g=1 | Existing Full | Full−OpenPath |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed in seeds:
        lines.append(f"| {seed} | {main_base[seed]:.4f} | {main_open[seed]:.4f} | {main_full[seed]:.4f} | {main_full[seed]-main_open[seed]:+.4f} |")
    lines.extend(
        [
            f"| **Mean** | **{statistics.mean(base_values):.4f}** | **{statistics.mean(open_values):.4f}** | **{statistics.mean(full_values):.4f}** | **{statistics.mean(permission_delta):+.4f}** |",
            "",
            "## Direct answers to the reviewer questions",
            "",
            f"- Path exposure accounts for the LastFM-DyGFormer main-budget gain: Base→OpenPath is `{statistics.mean(open_values)-statistics.mean(base_values):+.4f}` AUC, while OpenPath→Full is `{statistics.mean(permission_delta):+.4f}` AUC.",
            "- The 10-epoch controlled seed-7 run also does not show an advantage for the unnormalized edge gate over OpenPath. The normalized query-item control also remains below OpenPath in that run.",
            "- The earlier path-localization report mixed a 10,000-example OpenPath row with a 32,768-example Full row and labeled a model with generic attention Warrant enabled as correct-path-only. Those rows cannot support a gate-versus-path conclusion.",
            "- The trained edge gate is nearly identity in the unnormalized Full variants. The matched inference-time g=1 intervention consequently changes AUC by approximately zero; this is consistent with the gate distribution and is not evidence of final-prediction suppression.",
            "",
            "These findings answer the requested CTDG ablation but do not test all meanings of permission. They specifically show that the current learned unnormalized g does not explain the reported LastFM-DyGFormer gain under the present implementation.",
            "",
        ]
    )
    aux_path = ROOT / "experiments/permission_decomposition/outputs_auxiliary/ctdg/lastfm/DyGFormer/correct_path_warrant/seed_7/final_metrics.json"
    if aux_path.exists():
        aux = json.loads(aux_path.read_text(encoding="utf-8"))
        aux_001_path = ROOT / "experiments/permission_decomposition/outputs_auxiliary_001/ctdg/lastfm/DyGFormer/correct_path_warrant/seed_7/final_metrics.json"
        aux_001 = json.loads(aux_001_path.read_text(encoding="utf-8")) if aux_001_path.exists() else None
        plain = read_variant(DIAGNOSTIC, "correct_path_warrant").get(7, {})
        lines.extend(
            [
                "",
                "## Auxiliary permission supervision diagnostic (seed 7)",
                "",
                "The target is whether each endpoint-history item is the current counterpart, used as a CTDG evidence proxy. This is a diagnostic target rather than a general permission label.",
                "",
                "| Variant | AUC | Gate mean | Gate std | High saturation | Gate grad norm | Inference native−g=1 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| Task loss only | {float(plain.get('auc', float('nan'))):.4f} | {float(plain.get('edge_warrant_gate_mean', float('nan'))):.4f} | {float(plain.get('edge_warrant_gate_std', float('nan'))):.4f} | {float(plain.get('edge_warrant_gate_high_fraction', float('nan'))):.4f} | {float(plain.get('edge_gate_grad_norm', float('nan'))):.6f} | {float(plain.get('inference_g1_delta', float('nan'))):+.4f} |",
                f"| + auxiliary evidence loss | {float(aux['auc']):.4f} | {float(aux['edge_warrant_gate_mean']):.4f} | {float(aux['edge_warrant_gate_std']):.4f} | {float(aux['edge_warrant_gate_high_fraction']):.4f} | {float(aux['edge_gate_grad_norm']):.6f} | {float(aux['inference_g1_delta']):+.4f} |",
                "",
                "The auxiliary objective prevents the identity-gate collapse and creates measurable inference-time dependence on g, but this first target/weight does not improve AUC over OpenPath.",
            ]
        )
        if aux_001 is not None:
            lines.insert(
                -2,
                f"| + auxiliary evidence loss (0.01) | {float(aux_001['auc']):.4f} | {float(aux_001['edge_warrant_gate_mean']):.4f} | {float(aux_001['edge_warrant_gate_std']):.4f} | {float(aux_001['edge_warrant_gate_high_fraction']):.4f} | {float(aux_001['edge_gate_grad_norm']):.6f} | {float(aux_001['inference_g1_delta']):+.4f} |",
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
