#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pandas.errors import EmptyDataError

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.negative_row_diagnostics.run import EXTRA_COLUMNS  # noqa: E402
from experiments.warrant_need_score.agg import run_aggregation as run_wns_aggregation  # noqa: E402


DEFAULT_RESULTS = Path(__file__).resolve().parent / "outputs" / "results.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "analysis"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate negative-row micro-WNS and failure diagnostics.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--main-base-vs-warrant", type=Path, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    if abs(value) < 0.5 * (10 ** -digits):
        value = 0.0
    return f"{value:.{digits}f}"


def mean_std_text(mean: float, std: float, digits: int = 4) -> str:
    if not math.isfinite(float(mean)):
        return "n/a"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def markdown_table(frame: pd.DataFrame, *, digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._\n"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}")
    lines = [
        "| " + " | ".join(str(column) for column in display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(lines) + "\n"


def read_completed(results_path: Path) -> pd.DataFrame:
    if not results_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(results_path)
    if "status" in frame:
        frame = frame[frame["status"].eq("completed")].copy()
    for column in ["seed", "primary_metric", *EXTRA_COLUMNS]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def apply_row_main_summary(need_scores: pd.DataFrame, main_path: Path | None) -> pd.DataFrame:
    if need_scores.empty or main_path is None or not main_path.exists():
        return need_scores
    main = pd.read_csv(main_path)
    required = {"domain", "dataset", "model"}
    if not required.issubset(main.columns):
        return need_scores
    value_cols = [
        "improvement_delta_mean",
        "relative_delta_pct_mean",
        "improved",
        "practical_tier",
        "paired_ci95",
        "paired_t_p",
    ]
    keep = ["domain", "dataset", "model", *[col for col in value_cols if col in main.columns]]
    exact = main[keep].copy()
    exact = exact.rename(
        columns={
            "improvement_delta_mean": "main_row_improvement",
            "relative_delta_pct_mean": "main_row_relative_pct",
            "improved": "main_row_improved",
            "practical_tier": "main_row_tier",
            "paired_ci95": "main_row_paired_ci95",
            "paired_t_p": "main_row_paired_t_p",
        }
    )
    drop_cols = [
        "main_mean_improvement",
        "main_mean_relative_pct",
        "main_win_rate",
        "substantial_rate",
        "drop_rate",
        "main_row_improvement",
        "main_row_relative_pct",
        "main_row_improved",
        "main_row_tier",
        "main_row_paired_ci95",
        "main_row_paired_t_p",
    ]
    out = need_scores.drop(columns=[col for col in drop_cols if col in need_scores.columns], errors="ignore")
    out = out.merge(exact, on=["domain", "dataset", "model"], how="left")
    return out


def aggregate_extra_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    available = [column for column in EXTRA_COLUMNS if column in frame.columns]
    if frame.empty or not available:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["domain", "dataset", "model", "variant"], dropna=False)
    for (domain, dataset, model, variant), subset in grouped:
        row: dict[str, Any] = {
            "domain": domain,
            "dataset": dataset,
            "model": model,
            "variant": variant,
            "seeds": int(subset["seed"].nunique()) if "seed" in subset else len(subset),
        }
        for column in available:
            values = pd.to_numeric(subset[column], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_compact_failure_tables(extra: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if extra.empty:
        return pd.DataFrame(), pd.DataFrame()
    stpp_cols = [
        "domain",
        "dataset",
        "model",
        "variant",
        "seeds",
        "stpp_rmse_revisit_mean",
        "stpp_rmse_new_place_mean",
        "stpp_rmse_short_move_mean",
        "stpp_rmse_long_move_mean",
        "stpp_useful_mass_retention_mean",
        "stpp_nonuseful_mass_retention_mean",
        "stpp_false_suppression_index_mean",
        "stpp_prior_rmse_delta_mean",
    ]
    tkg_cols = [
        "domain",
        "dataset",
        "model",
        "variant",
        "seeds",
        "tkg_true_tail_seen_rate_mean",
        "tkg_mrr_seen_mean",
        "tkg_mrr_unseen_mean",
        "tkg_true_copy_retention_mean",
        "tkg_false_copy_retention_mean",
        "tkg_copy_saturation_attention_mean",
        "tkg_copy_saturation_warrant_mean",
        "tkg_copy_gate_mean_seen_mean",
        "tkg_copy_gate_mean_unseen_mean",
    ]
    stpp = extra[extra["domain"].eq("stpp")][[column for column in stpp_cols if column in extra.columns]].copy()
    tkg = extra[extra["domain"].eq("tkg")][[column for column in tkg_cols if column in extra.columns]].copy()
    return stpp, tkg


def variant_lookup(extra: pd.DataFrame, domain: str, variant: str, column: str) -> float:
    if extra.empty or column not in extra.columns:
        return float("nan")
    subset = extra[extra["domain"].eq(domain) & extra["variant"].eq(variant)]
    if subset.empty:
        return float("nan")
    return float(pd.to_numeric(subset.iloc[0].get(column), errors="coerce"))


def write_failure_report(output_dir: Path, results: pd.DataFrame, extra: pd.DataFrame, need_scores: pd.DataFrame) -> None:
    stpp_table, tkg_table = build_compact_failure_tables(extra)
    report_path = output_dir / "negative_row_diagnostic_report.md"
    lines = [
        "# Negative Row Diagnostics",
        "",
        "This diagnostic audits the two main-benchmark negative rows that are not fully explained by domain-level WNS alone: `Gowalla / DeepSTPP` and `ICEWS18 / CyGNet`.",
        "",
        "The experiment has two layers.",
        "",
        "1. **micro-WNS** recomputes `PathReach`, `Correct > Generic`, and `Correct > Shuffled` on the exact negative row.",
        "2. **failure diagnostics** inspect whether the learned permission suppresses useful STPP history or disrupts CyGNet's copy/generation balance.",
        "",
        "## micro-WNS",
        "",
    ]
    if need_scores.empty:
        lines.append("_No micro-WNS table was produced._")
    else:
        keep = [
            "domain",
            "dataset",
            "model",
            "warrant_need_score",
            "warrant_need_score_minmax",
            "warrant_need_tier",
            "main_row_relative_pct",
            "main_row_tier",
            "path_reach_rel_mean",
            "correct_over_generic_rel_mean",
            "correct_over_shuffled_rel_mean",
        ]
        table = need_scores[[column for column in keep if column in need_scores.columns]].copy()
        lines.append(markdown_table(table))

    lines.extend(
        [
            "",
            "## Gowalla / DeepSTPP: false suppression diagnostic",
            "",
            "The STPP diagnostic treats the closest half of the valid history events to the target location as useful local history for that prediction.  It reports whether the permission gate preserves that useful local mass or suppresses it more than the remaining history.",
            "",
            markdown_table(stpp_table),
            "",
            "Reading rule: positive `stpp_false_suppression_index_mean` means non-useful history is retained more than useful history.  Positive `stpp_prior_rmse_delta_mean` means the Warrant-weighted location prior is farther from the target than the raw-attention prior.",
            "",
            "## ICEWS18 / CyGNet: copy-saturation diagnostic",
            "",
            "The TKG diagnostic separates true-tail copy mass from the largest false-tail copy mass.  It checks whether Warrant preserves the useful copy path or disturbs CyGNet's existing copy/generation mixture.",
            "",
            markdown_table(tkg_table),
            "",
            "Reading rule: `tkg_true_copy_retention_mean < tkg_false_copy_retention_mean` indicates that true-tail copy mass is suppressed more than false-tail copy mass.  A larger `tkg_copy_saturation_warrant_mean` than `tkg_copy_saturation_attention_mean` indicates a worse false-over-true copy imbalance after permission scaling.",
            "",
            "## Suggested paper use",
            "",
            "Use this report as a row-level failure audit rather than a new main benchmark.  WNS explains cross-domain effect heterogeneity, while this diagnostic explains why particular negative rows can occur when Warrant collides with an existing useful prior path: Gowalla revisit/place-prior dynamics or CyGNet copy saturation.",
        ]
    )

    # Add compact verdicts when the relevant numbers exist.
    stpp_false = variant_lookup(extra, "stpp", "correct_path_warrant", "stpp_false_suppression_index_mean")
    stpp_prior = variant_lookup(extra, "stpp", "correct_path_warrant", "stpp_prior_rmse_delta_mean")
    tkg_true = variant_lookup(extra, "tkg", "correct_path_warrant", "tkg_true_copy_retention_mean")
    tkg_false = variant_lookup(extra, "tkg", "correct_path_warrant", "tkg_false_copy_retention_mean")
    tkg_sat_attn = variant_lookup(extra, "tkg", "correct_path_warrant", "tkg_copy_saturation_attention_mean")
    tkg_sat_warrant = variant_lookup(extra, "tkg", "correct_path_warrant", "tkg_copy_saturation_warrant_mean")
    verdicts = [
        "",
        "## Automatic verdict hints",
        "",
        f"- STPP false suppression index on correct path: `{fmt(stpp_false)}`.",
        f"- STPP Warrant-prior minus attention-prior RMSE: `{fmt(stpp_prior)}`.",
        f"- TKG true-copy retention vs false-copy retention: `{fmt(tkg_true)}` vs `{fmt(tkg_false)}`.",
        f"- TKG copy saturation attention vs Warrant: `{fmt(tkg_sat_attn)}` vs `{fmt(tkg_sat_warrant)}`.",
    ]
    lines.extend(verdicts)
    if math.isfinite(stpp_prior) and abs(stpp_prior) < 5.0e-5:
        lines.extend(
            [
                "",
                "The STPP prior-RMSE delta is numerically near zero.  This indicates that Warrant changes mass retention between useful and non-useful history, but does not materially move the simple attention-weighted location prior in this diagnostic.",
            ]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_micro_wns_report(output_dir: Path, need_scores: pd.DataFrame) -> None:
    lines = [
        "# Negative-Row micro-WNS",
        "",
        "This report recomputes WNS on the exact negative rows rather than using a domain-level representative setting.",
        "",
        "The score uses three components: `PathReach`, `Correct > Generic`, and `Correct > Shuffled`.  The `Main row relative` column is copied from the exact row in the main benchmark table.",
        "",
    ]
    if need_scores.empty:
        lines.append("_No micro-WNS rows were produced._")
    else:
        keep = [
            "domain",
            "dataset",
            "model",
            "warrant_need_score",
            "warrant_need_score_minmax",
            "warrant_need_tier",
            "main_row_relative_pct",
            "main_row_tier",
            "path_reach_rel_mean",
            "correct_over_generic_rel_mean",
            "correct_over_shuffled_rel_mean",
        ]
        table = need_scores[[column for column in keep if column in need_scores.columns]].copy()
        lines.append(markdown_table(table))
    lines.extend(
        [
            "",
            "Reading rule: a high micro-WNS on a negative row means the row is highly sensitive to path/pairing controls even if the final Warrant variant drops against Base.  That pattern points to path collision or false suppression, not absence of a Warrant-relevant bottleneck.",
        ]
    )
    (output_dir / "warrant_need_score_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_aggregation(*, results_path: Path, output_dir: Path, config: dict[str, Any], main_path: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_wns_aggregation(results_path=results_path, output_dir=output_dir, config=config, main_path=main_path)
    results = read_completed(results_path)
    extra = aggregate_extra_metrics(results)
    extra.to_csv(output_dir / "failure_diagnostics_by_variant.csv", index=False)
    stpp_table, tkg_table = build_compact_failure_tables(extra)
    stpp_table.to_csv(output_dir / "stpp_false_suppression.csv", index=False)
    tkg_table.to_csv(output_dir / "tkg_copy_saturation.csv", index=False)
    need_scores_path = output_dir / "need_scores.csv"
    if need_scores_path.exists():
        try:
            need_scores = pd.read_csv(need_scores_path)
        except EmptyDataError:
            need_scores = pd.DataFrame()
    else:
        need_scores = pd.DataFrame()
    need_scores = apply_row_main_summary(need_scores, main_path)
    if not need_scores.empty:
        need_scores.to_csv(need_scores_path, index=False)
    write_micro_wns_report(output_dir, need_scores)
    write_failure_report(output_dir, results, extra, need_scores)
    print(f"wrote {output_dir / 'failure_diagnostics_by_variant.csv'}")
    print(f"wrote {output_dir / 'negative_row_diagnostic_report.md'}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    main_path = args.main_base_vs_warrant
    if main_path is None:
        configured = config.get("inputs", {}).get("main_base_vs_warrant")
        main_path = REPO_ROOT / str(configured) if configured else None
    run_aggregation(
        results_path=args.results,
        output_dir=args.output_dir,
        config=config,
        main_path=main_path if main_path and main_path.exists() else None,
    )


if __name__ == "__main__":
    main()
