#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plots.common import save_figure, set_paper_style  # noqa: E402


DEFAULT_RESULTS = Path(__file__).resolve().parent / "outputs" / "results.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "analysis"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
VARIANT_ORDER = ["base", "generic_qk_warrant", "open_path_no_gate", "correct_path_warrant", "shuffled_pairing"]
DOMAIN_ORDER = ["ctdg", "mtpp", "rag", "stpp", "tkg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate dedicated Warrant Need Score diagnostic runs.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--main-base-vs-warrant", type=Path, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "completed":
                continue
            parsed = dict(row)
            parsed["seed"] = int(row["seed"])
            parsed["primary_metric"] = parse_float(row.get("primary_metric"))
            parsed["eval_loss"] = parse_float(row.get("eval_loss"))
            rows.append(parsed)
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], 0.0
    return statistics.mean(clean), statistics.stdev(clean)


def direction_sign(direction: str) -> float:
    return -1.0 if str(direction).lower() == "lower" else 1.0


def safe_rel(delta: float, base: float) -> float:
    return float(delta) / max(abs(float(base)), 1.0e-8)


def positive_rel(delta: float, base: float) -> float:
    return safe_rel(max(float(delta), 0.0), base)


def rank_minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() <= 1:
        return values * 0.0
    lo = values.min()
    hi = values.max()
    if abs(hi - lo) < 1.0e-12:
        return values * 0.0
    return (values - lo) / (hi - lo)


def need_tier(value: float) -> str:
    if not math.isfinite(float(value)):
        return "n/a"
    value = float(value)
    if value < 0.005:
        return "negligible"
    if value < 0.02:
        return "weak"
    if value < 0.10:
        return "moderate"
    if value < 0.25:
        return "strong"
    return "very_high"


def variant_sort(variant: str) -> int:
    try:
        return VARIANT_ORDER.index(variant)
    except ValueError:
        return len(VARIANT_ORDER)


def aggregate_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["domain"], row["dataset"], row["model"], row["variant"])].append(row)

    summary: list[dict[str, Any]] = []
    for (domain, dataset, model, variant), subset in sorted(
        grouped.items(),
        key=lambda item: (DOMAIN_ORDER.index(item[0][0]) if item[0][0] in DOMAIN_ORDER else 99, item[0][1], item[0][2], variant_sort(item[0][3])),
    ):
        primary_mean, primary_std = mean_std([row["primary_metric"] for row in subset])
        loss_mean, loss_std = mean_std([row["eval_loss"] for row in subset])
        summary.append(
            {
                "domain": domain,
                "dataset": dataset,
                "model": model,
                "variant": variant,
                "path_role": subset[0].get("path_role", variant),
                "seeds": len(subset),
                "metric_name": subset[0].get("metric_name", ""),
                "direction": subset[0].get("direction", "higher"),
                "primary_mean": primary_mean,
                "primary_std": primary_std,
                "eval_loss_mean": loss_mean,
                "eval_loss_std": loss_std,
            }
        )
    return summary


def aggregate_components(rows: list[dict[str, Any]], weights: dict[str, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_seed: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_seed[(row["domain"], row["dataset"], row["model"], int(row["seed"]))][row["variant"]] = row

    paired: list[dict[str, Any]] = []
    for (domain, dataset, model, seed), variants in sorted(by_seed.items()):
        base = variants.get("base")
        correct = variants.get("correct_path_warrant")
        if base is None or correct is None:
            continue
        sign = direction_sign(base.get("direction", "higher"))
        base_value = float(base["primary_metric"])
        correct_value = float(correct["primary_metric"])
        generic_value = float(variants["generic_qk_warrant"]["primary_metric"]) if "generic_qk_warrant" in variants else float("nan")
        shuffled_value = float(variants["shuffled_pairing"]["primary_metric"]) if "shuffled_pairing" in variants else float("nan")
        open_value = float(variants["open_path_no_gate"]["primary_metric"]) if "open_path_no_gate" in variants else float("nan")

        def signed_delta(left: float, right: float) -> float:
            return sign * (left - right)

        path_reach = signed_delta(correct_value, base_value)
        correct_over_generic = signed_delta(correct_value, generic_value) if math.isfinite(generic_value) else float("nan")
        correct_over_shuffled = signed_delta(correct_value, shuffled_value) if math.isfinite(shuffled_value) else float("nan")
        correct_over_open = signed_delta(correct_value, open_value) if math.isfinite(open_value) else float("nan")
        open_over_base = signed_delta(open_value, base_value) if math.isfinite(open_value) else float("nan")

        component_values = {
            "path_reach_rel": safe_rel(path_reach, base_value),
            "correct_over_generic_rel": safe_rel(correct_over_generic, base_value) if math.isfinite(correct_over_generic) else float("nan"),
            "correct_over_shuffled_rel": safe_rel(correct_over_shuffled, base_value) if math.isfinite(correct_over_shuffled) else float("nan"),
            "correct_over_open_path_rel": safe_rel(correct_over_open, base_value) if math.isfinite(correct_over_open) else float("nan"),
        }
        clipped_component_values = {
            name: max(value, 0.0) if math.isfinite(value) else float("nan")
            for name, value in component_values.items()
        }
        weight_total = sum(float(weights.get(name, 0.0)) for name, value in clipped_component_values.items() if math.isfinite(value))
        seed_raw_need = (
            sum(float(weights.get(name, 0.0)) * value for name, value in clipped_component_values.items() if math.isfinite(value))
            / max(weight_total, 1.0e-12)
        )

        paired.append(
            {
                "domain": domain,
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "metric_name": base.get("metric_name", ""),
                "direction": base.get("direction", "higher"),
                "base": base_value,
                "generic_qk_warrant": generic_value,
                "open_path_no_gate": open_value,
                "correct_path_warrant": correct_value,
                "shuffled_pairing": shuffled_value,
                "path_reach": path_reach,
                "correct_over_generic": correct_over_generic,
                "correct_over_shuffled": correct_over_shuffled,
                "correct_over_open_path": correct_over_open,
                "open_path_over_base": open_over_base,
                "path_reach_rel_signed": component_values["path_reach_rel"],
                "correct_over_generic_rel_signed": component_values["correct_over_generic_rel"],
                "correct_over_shuffled_rel_signed": component_values["correct_over_shuffled_rel"],
                "correct_over_open_path_rel_signed": component_values["correct_over_open_path_rel"],
                "path_reach_rel": clipped_component_values["path_reach_rel"],
                "correct_over_generic_rel": clipped_component_values["correct_over_generic_rel"],
                "correct_over_shuffled_rel": clipped_component_values["correct_over_shuffled_rel"],
                "correct_over_open_path_rel": clipped_component_values["correct_over_open_path_rel"],
                "seed_need_score_raw": seed_raw_need,
            }
        )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        grouped[(row["domain"], row["dataset"], row["model"])].append(row)

    scores: list[dict[str, Any]] = []
    numeric_cols = [
        "base",
        "generic_qk_warrant",
        "open_path_no_gate",
        "correct_path_warrant",
        "shuffled_pairing",
        "path_reach",
        "correct_over_generic",
        "correct_over_shuffled",
        "correct_over_open_path",
        "open_path_over_base",
        "path_reach_rel_signed",
        "correct_over_generic_rel_signed",
        "correct_over_shuffled_rel_signed",
        "correct_over_open_path_rel_signed",
        "path_reach_rel",
        "correct_over_generic_rel",
        "correct_over_shuffled_rel",
        "correct_over_open_path_rel",
        "seed_need_score_raw",
    ]
    for (domain, dataset, model), subset in sorted(grouped.items(), key=lambda item: DOMAIN_ORDER.index(item[0][0]) if item[0][0] in DOMAIN_ORDER else 99):
        out: dict[str, Any] = {
            "domain": domain,
            "dataset": dataset,
            "model": model,
            "metric_name": subset[0]["metric_name"],
            "direction": subset[0]["direction"],
            "seeds": len(subset),
        }
        for col in numeric_cols:
            mean, std = mean_std([parse_float(row.get(col)) for row in subset])
            out[f"{col}_mean"] = mean
            out[f"{col}_std"] = std
        signed_components = {
            "path_reach_rel": parse_float(out.get("path_reach_rel_signed_mean")),
            "correct_over_generic_rel": parse_float(out.get("correct_over_generic_rel_signed_mean")),
            "correct_over_shuffled_rel": parse_float(out.get("correct_over_shuffled_rel_signed_mean")),
            "correct_over_open_path_rel": parse_float(out.get("correct_over_open_path_rel_signed_mean")),
        }
        clipped_domain_components = {
            name: max(value, 0.0) if math.isfinite(value) else float("nan")
            for name, value in signed_components.items()
        }
        weight_total = sum(float(weights.get(name, 0.0)) for name, value in clipped_domain_components.items() if math.isfinite(value))
        domain_raw_need = (
            sum(float(weights.get(name, 0.0)) * value for name, value in clipped_domain_components.items() if math.isfinite(value))
            / max(weight_total, 1.0e-12)
        )
        out["path_reach_rel_mean"] = clipped_domain_components["path_reach_rel"]
        out["correct_over_generic_rel_mean"] = clipped_domain_components["correct_over_generic_rel"]
        out["correct_over_shuffled_rel_mean"] = clipped_domain_components["correct_over_shuffled_rel"]
        out["correct_over_open_path_rel_mean"] = clipped_domain_components["correct_over_open_path_rel"]
        out["need_score_raw_mean"] = domain_raw_need
        scores.append(out)
    return paired, scores


def build_main_summary(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty or "domain" not in frame:
        return pd.DataFrame()
    frame["improved"] = frame["improved"].astype(str).str.lower().isin(["true", "1", "yes"])
    for col in ["improvement_delta_mean", "relative_delta_pct_mean"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    rows = []
    for domain, subset in frame.groupby("domain"):
        tiers = subset["practical_tier"].fillna("").astype(str) if "practical_tier" in subset else pd.Series([""] * len(subset))
        rows.append(
            {
                "domain": domain,
                "main_rows": int(len(subset)),
                "main_mean_improvement": float(subset["improvement_delta_mean"].mean()),
                "main_mean_relative_pct": float(subset["relative_delta_pct_mean"].mean()),
                "main_win_rate": float(subset["improved"].mean()),
                "substantial_rate": float(tiers.str.contains("substantial", case=False).mean()),
                "drop_rate": float(tiers.str.contains("drop", case=False).mean()),
            }
        )
    return pd.DataFrame(rows)


def exact_spearman_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    observed = float(pd.Series(xr).corr(pd.Series(yr), method="pearson"))
    if n > 8:
        return observed, float("nan")
    total = 0
    extreme = 0
    for perm in itertools.permutations(yr.tolist()):
        total += 1
        corr = float(pd.Series(xr).corr(pd.Series(perm), method="pearson"))
        if abs(corr) >= abs(observed) - 1.0e-12:
            extreme += 1
    return observed, extreme / max(total, 1)


def compute_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    predictors = [
        "need_score_raw_mean",
        "warrant_need_score",
        "path_reach_rel_mean",
        "correct_over_generic_rel_mean",
        "correct_over_shuffled_rel_mean",
        "correct_over_open_path_rel_mean",
    ]
    targets = ["main_mean_relative_pct", "main_win_rate", "substantial_rate", "drop_rate"]
    rows = []
    for predictor in predictors:
        if predictor not in frame:
            continue
        for target in targets:
            if target not in frame:
                continue
            x = pd.to_numeric(frame[predictor], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 3:
                continue
            pearson = float(pd.Series(x[mask]).corr(pd.Series(y[mask]), method="pearson"))
            spearman, exact_p = exact_spearman_p(x[mask], y[mask])
            rows.append(
                {
                    "predictor": predictor,
                    "target": target,
                    "n": int(mask.sum()),
                    "pearson": pearson,
                    "spearman": spearman,
                    "spearman_exact_p_two_sided": exact_p,
                }
            )
    return pd.DataFrame(rows)


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def plot_need_vs_gain(frame: pd.DataFrame, output_dir: Path, stem: str, annotate: bool) -> tuple[Path, Path] | None:
    if frame.empty or "main_mean_relative_pct" not in frame:
        return None
    plot_frame = frame.dropna(subset=["warrant_need_score", "main_mean_relative_pct"]).copy()
    if plot_frame.empty:
        return None
    set_paper_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    sns.scatterplot(
        data=plot_frame,
        x="warrant_need_score",
        y="main_mean_relative_pct",
        hue="domain",
        s=90,
        edgecolor="white",
        linewidth=0.9,
        ax=ax,
        legend=False,
    )
    if annotate:
        for _, row in plot_frame.iterrows():
            ax.text(
                float(row["warrant_need_score"]) + 0.012,
                float(row["main_mean_relative_pct"]),
                str(row["domain"]).upper(),
                fontsize=8.3,
                va="center",
            )
    ax.set_xlabel("Dedicated Warrant Need Score")
    ax.set_ylabel("main benchmark relative gain (%)")
    ax.axhline(0.0, color="#9CA3AF", linewidth=0.9, linestyle="--")
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    fig.tight_layout()
    paths = save_figure(fig, stem, output_dir=output_dir)
    plt.close(fig)
    return paths


def write_report(
    path: Path,
    variant_summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    need_frame: pd.DataFrame,
    corr: pd.DataFrame,
    fig_paths: tuple[Path, Path] | None,
) -> None:
    lines = [
        "# Dedicated Warrant Need Score",
        "",
        "This report is produced from dedicated WNS diagnostic runs.  The model is trained and evaluated for each control variant under `experiments/warrant_need_score/outputs`; the score is not computed by reusing previous path-localization or mass-diagnostic outputs.",
        "",
        "## Design",
        "",
        "| Variant | Diagnostic role |",
        "| --- | --- |",
        "| `base` | No Warrant and no localized permission path. |",
        "| `generic_qk_warrant` | Warrant on generic query-key attention while the localized metric path is disabled. |",
        "| `open_path_no_gate` | Metric-facing path opened with `g=1` and no learned permission gate. |",
        "| `correct_path_warrant` | Warrant on the metric-facing weighted value path. |",
        "| `shuffled_pairing` | Correct path retained, but query-item pairing mismatched. |",
        "",
        "A domain receives a high WNS when the correct path improves over base, beats generic placement, degrades when query-item pairing is broken, and improves over the open-path control.  The open-path control separates learned permission from merely exposing the metric-facing contribution path.",
        "",
        "## Need Scores",
        "",
        "| Domain | Dataset | Model | Metric | Seeds | WNS | Relative WNS | Tier | Main rel. gain % | Path reach | Correct-Generic | Correct-Shuffled | Correct-OpenPath |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered = need_frame.copy()
    if not ordered.empty:
        ordered["_order"] = ordered["domain"].map({domain: idx for idx, domain in enumerate(DOMAIN_ORDER)}).fillna(99)
        ordered = ordered.sort_values(["_order", "domain"])
    for _, row in ordered.iterrows():
        lines.append(
            "| {domain} | {dataset} | {model} | {metric} | {seeds} | {wns} | {relative} | {tier} | {main} | {reach} | {generic} | {shuffled} | {open_path} |".format(
                domain=str(row["domain"]).upper(),
                dataset=row["dataset"],
                model=row["model"],
                metric=row["metric_name"],
                seeds=int(row["seeds"]),
                wns=fmt(row.get("warrant_need_score")),
                relative=fmt(row.get("warrant_need_score_minmax")),
                tier=row.get("warrant_need_tier", "n/a"),
                main=fmt(row.get("main_mean_relative_pct"), 3),
                reach=fmt(row.get("path_reach_rel_mean")),
                generic=fmt(row.get("correct_over_generic_rel_mean")),
                shuffled=fmt(row.get("correct_over_shuffled_rel_mean")),
                open_path=fmt(row.get("correct_over_open_path_rel_mean")),
            )
        )

    lines.extend(["", "## Correlations", ""])
    if corr.empty:
        lines.append("_No correlation table was produced because the main benchmark target file was unavailable or too few domains were complete._")
    else:
        lines.extend(
            [
                "| Predictor | Target | n | Pearson | Spearman | Exact p |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in corr.iterrows():
            lines.append(
                "| {predictor} | {target} | {n} | {pearson} | {spearman} | {p} |".format(
                    predictor=row["predictor"],
                    target=row["target"],
                    n=int(row["n"]),
                    pearson=fmt(row["pearson"], 3),
                    spearman=fmt(row["spearman"], 3),
                    p=fmt(row["spearman_exact_p_two_sided"], 3),
                )
            )

    if fig_paths is not None:
        png, pdf = fig_paths
        lines.extend(
            [
                "",
                "## Figure",
                "",
                f"![Dedicated Warrant Need Score vs Gain]({png})",
                "",
                f"PDF: [{pdf.name}]({pdf})",
            ]
        )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "WNS is the raw direction-aware diagnostic effect size normalized by each domain's base metric.  `Relative WNS` is a min-max rescaling used only for ranking and plotting; it always assigns 0 to the smallest observed domain and 1 to the largest observed domain.  The tier column is a practical reading of the raw score: negligible (<0.005), weak (0.005-0.02), moderate (0.02-0.10), strong (0.10-0.25), and very_high (>=0.25).",
            "",
            "WNS is high when the trained controls show that the metric-facing path is reachable, more useful than generic attention placement, and sensitive to the correct query-item pairing.  The score should be compared against the main benchmark only after being computed from these dedicated diagnostic runs.",
            "",
            "This report does not treat WNS as a ground-truth label.  It is an operational measurement of the bottleneck that Warrant claims to solve.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_aggregation(
    *,
    results_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    main_path: Path | None,
) -> None:
    rows = read_rows(results_path)
    weights = dict(config.get("need_score", {}).get("weights", {}))
    variant_summary = aggregate_variants(rows)
    paired, need_scores = aggregate_components(rows, weights)
    need_frame = pd.DataFrame(need_scores)
    if not need_frame.empty:
        need_frame["warrant_need_score"] = pd.to_numeric(need_frame["need_score_raw_mean"], errors="coerce")
        need_frame["warrant_need_score_minmax"] = rank_minmax(need_frame["need_score_raw_mean"])
        need_frame["warrant_need_tier"] = need_frame["warrant_need_score"].map(need_tier)

    main_summary = build_main_summary(main_path)
    if not main_summary.empty and not need_frame.empty:
        need_frame = need_frame.merge(main_summary, on="domain", how="left")
    corr = compute_correlations(need_frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "variant_summary.csv", variant_summary)
    write_csv(output_dir / "paired_components.csv", paired)
    write_csv(output_dir / "need_scores.csv", need_frame)
    write_csv(output_dir / "correlations.csv", corr)
    fig_paths = plot_need_vs_gain(
        need_frame,
        output_dir,
        str(config.get("plot", {}).get("stem", "warrant_need_score_vs_gain")),
        bool(config.get("plot", {}).get("annotate_domains", True)),
    )
    write_report(output_dir / "warrant_need_score_report.md", variant_summary, paired, need_frame, corr, fig_paths)
    print(f"wrote {output_dir / 'variant_summary.csv'}")
    print(f"wrote {output_dir / 'paired_components.csv'}")
    print(f"wrote {output_dir / 'need_scores.csv'}")
    print(f"wrote {output_dir / 'correlations.csv'}")
    print(f"wrote {output_dir / 'warrant_need_score_report.md'}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    main_path = args.main_base_vs_warrant
    if main_path is None:
        configured = config.get("inputs", {}).get("main_base_vs_warrant")
        main_path = REPO_ROOT / str(configured) if configured else None
    run_aggregation(results_path=args.results, output_dir=args.output_dir, config=config, main_path=main_path)


if __name__ == "__main__":
    main()
