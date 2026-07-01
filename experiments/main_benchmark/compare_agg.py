#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO_ROOT / "experiments" / "main_benchmark" / "outputs" / "results.csv"
DEFAULT_OUT = REPO_ROOT / "experiments" / "main_benchmark" / "outputs" / "analysis"


KEYS = ["domain", "dataset", "model", "implementation"]

DOMAIN_PRIMARY_METRICS = {
    "ctdg": ("auc", "higher"),
    "mtpp": ("mark_mrr", "higher"),
    "stpp": ("rmse_location", "lower"),
    "tkg": ("mrr", "higher"),
    "rag": ("support_mrr", "higher"),
}

LOWER_IS_BETTER_METRICS = {
    "eval_loss",
    "joint_nll",
    "mae_time",
    "rmse_location",
    "train_loss",
}


def metric_label(domain: str, metric: str) -> str:
    if metric == "primary_metric":
        return DOMAIN_PRIMARY_METRICS.get(str(domain), ("primary_metric", "higher"))[0]
    return metric


def metric_direction(domain: str, metric: str) -> str:
    if metric == "primary_metric":
        return DOMAIN_PRIMARY_METRICS.get(str(domain), ("primary_metric", "higher"))[1]
    return "lower" if metric in LOWER_IS_BETTER_METRICS else "higher"


def direction_sign(direction: str) -> float:
    return -1.0 if direction == "lower" else 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate main benchmark results and plot base-vs-warrant deltas.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Path to results.csv.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Directory for tables and plots.")
    parser.add_argument("--metric", default="primary_metric", help="Metric column to aggregate.")
    parser.add_argument("--base", default="base", help="Base variant name.")
    parser.add_argument("--warrant", default="warrant", help="Warrant variant name.")
    parser.add_argument("--include-incomplete", action="store_true", help="Keep non-completed rows in raw copy only.")
    parser.add_argument("--top-k", type=int, default=80, help="Maximum bars in the global delta plot.")
    return parser.parse_args()


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return value or "unknown"


def fmt_mean_std(mean: float, std: float, n: int, digits: int = 4) -> str:
    if pd.isna(mean):
        return ""
    if n <= 1 or pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def fmt_signed(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{value:+.{digits}f}"


def fmt_signed_pct(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:+.{digits}f}%"


def ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in results.csv: {missing}")


def load_results(path: Path, metric: str, include_incomplete: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"results file does not exist: {path}")
    frame = pd.read_csv(path)
    ensure_columns(frame, [*KEYS, "variant", "seed", "status", metric])
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").astype("Int64")
    numeric_cols = [
        metric,
        "primary_metric",
        "accuracy",
        "mark_mrr",
        "mrr",
        "auc",
        "mae_time",
        "rmse_location",
        "train_loss",
        "eval_loss",
        "elapsed_sec",
        "warrant_active_blocks",
        "warrant_replacements",
    ]
    for column in numeric_cols:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["metric_name"] = frame["domain"].map(lambda domain: metric_label(domain, metric))
    frame["metric_direction"] = frame["domain"].map(lambda domain: metric_direction(domain, metric))
    if include_incomplete:
        return frame
    completed = frame[frame["status"].eq("completed") & frame[metric].notna()].copy()
    if metric == "primary_metric":
        if "mark_mrr" not in completed.columns:
            completed = completed[~completed["domain"].eq("mtpp")]
        else:
            completed = completed[~(completed["domain"].eq("mtpp") & completed["mark_mrr"].isna())]
        if "mrr" not in completed.columns:
            completed = completed[~completed["domain"].eq("tkg")]
        else:
            completed = completed[~(completed["domain"].eq("tkg") & completed["mrr"].isna())]
    return completed


def aggregate_variants(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = frame.groupby([*KEYS, "variant"], dropna=False)
    agg = grouped.agg(
        metric_name=("metric_name", "first"),
        metric_direction=("metric_direction", "first"),
        metric_mean=(metric, "mean"),
        metric_std=(metric, "std"),
        metric_min=(metric, "min"),
        metric_max=(metric, "max"),
        n_seeds=(metric, "count"),
        train_loss_mean=("train_loss", "mean"),
        eval_loss_mean=("eval_loss", "mean"),
        elapsed_sec_mean=("elapsed_sec", "mean"),
        active_blocks_mean=("warrant_active_blocks", "mean"),
    ).reset_index()
    agg["metric_mean_std"] = [
        fmt_mean_std(mean, std, int(n))
        for mean, std, n in zip(agg["metric_mean"], agg["metric_std"], agg["n_seeds"])
    ]
    return agg.sort_values([*KEYS, "variant"]).reset_index(drop=True)


def paired_seed_deltas(frame: pd.DataFrame, metric: str, base: str, warrant: str) -> pd.DataFrame:
    cols = [*KEYS, "seed", "variant", "metric_name", "metric_direction", metric]
    pivot = frame[cols].pivot_table(index=[*KEYS, "seed"], columns="variant", values=metric, aggfunc="mean").reset_index()
    if base not in pivot.columns or warrant not in pivot.columns:
        return pd.DataFrame(columns=[*KEYS, "seed", "metric_name", "metric_direction", "base", "warrant", "delta", "improvement_delta", "relative_delta_pct"])
    meta = frame[[*KEYS, "seed", "metric_name", "metric_direction"]].drop_duplicates([*KEYS, "seed"])
    paired = pivot[[*KEYS, "seed", base, warrant]].dropna().copy()
    paired = paired.rename(columns={base: "base", warrant: "warrant"})
    paired = paired.merge(meta, on=[*KEYS, "seed"], how="left")
    paired["delta"] = paired["warrant"] - paired["base"]
    paired["direction_sign"] = paired["metric_direction"].map(direction_sign).astype(float)
    paired["improvement_delta"] = paired["direction_sign"] * paired["delta"]
    paired["relative_delta_pct"] = np.where(
        paired["base"].abs() > 1.0e-12,
        100.0 * paired["improvement_delta"] / paired["base"].abs(),
        np.nan,
    )
    return paired.sort_values([*KEYS, "seed"]).reset_index(drop=True)


def aggregate_deltas(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return paired
    grouped = paired.groupby(KEYS, dropna=False)
    agg = grouped.agg(
        metric_name=("metric_name", "first"),
        metric_direction=("metric_direction", "first"),
        base_mean=("base", "mean"),
        base_std=("base", "std"),
        warrant_mean=("warrant", "mean"),
        warrant_std=("warrant", "std"),
        delta_mean=("delta", "mean"),
        delta_std=("delta", "std"),
        delta_min=("delta", "min"),
        delta_max=("delta", "max"),
        improvement_delta_mean=("improvement_delta", "mean"),
        improvement_delta_std=("improvement_delta", "std"),
        relative_delta_pct_mean=("relative_delta_pct", "mean"),
        relative_delta_pct_std=("relative_delta_pct", "std"),
        n_seeds=("delta", "count"),
    ).reset_index()
    agg["base"] = [fmt_mean_std(m, s, int(n)) for m, s, n in zip(agg["base_mean"], agg["base_std"], agg["n_seeds"])]
    agg["warrant"] = [
        fmt_mean_std(m, s, int(n)) for m, s, n in zip(agg["warrant_mean"], agg["warrant_std"], agg["n_seeds"])
    ]
    agg["delta"] = [fmt_signed(v) for v in agg["delta_mean"]]
    agg["improvement_delta"] = [fmt_signed(v) for v in agg["improvement_delta_mean"]]
    agg["relative_delta_pct"] = [fmt_signed_pct(v, digits=2) for v in agg["relative_delta_pct_mean"]]
    agg["improved"] = agg["improvement_delta_mean"] > 0
    inference_rows = []
    for key, subset in paired.groupby(KEYS, dropna=False):
        diffs = subset["improvement_delta"].dropna().to_numpy(dtype=float)
        n = len(diffs)
        mean = float(np.mean(diffs)) if n else float("nan")
        if n >= 2:
            sem = float(stats.sem(diffs))
            if sem == 0.0:
                ci_low = ci_high = mean
                p_value = 0.0 if mean != 0.0 else 1.0
            else:
                ci_low, ci_high = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
                sign = subset["direction_sign"].to_numpy(dtype=float)
                p_value = float(
                    stats.ttest_rel(
                        sign * subset["warrant"].to_numpy(dtype=float),
                        sign * subset["base"].to_numpy(dtype=float),
                    ).pvalue
                )
        else:
            ci_low = ci_high = p_value = float("nan")
        inference_rows.append(
            {
                **dict(zip(KEYS, key)),
                "paired_ci95_low": ci_low,
                "paired_ci95_high": ci_high,
                "paired_t_pvalue": p_value,
                "significant_0_05": bool(math.isfinite(p_value) and p_value < 0.05),
            }
        )
    agg = agg.merge(pd.DataFrame(inference_rows), on=KEYS, how="left")
    agg["paired_ci95"] = [
        "" if pd.isna(low) or pd.isna(high) else f"[{low:+.4f}, {high:+.4f}]"
        for low, high in zip(agg["paired_ci95_low"], agg["paired_ci95_high"])
    ]
    agg["paired_t_p"] = [
        "" if pd.isna(value) else f"{value:.4f}"
        for value in agg["paired_t_pvalue"]
    ]
    agg["practical_tier"] = [
        practical_tier(delta, relative, low, high)
        for delta, relative, low, high in zip(
            agg["improvement_delta_mean"],
            agg["relative_delta_pct_mean"],
            agg["paired_ci95_low"],
            agg["paired_ci95_high"],
        )
    ]
    return agg.sort_values(["domain", "dataset", "improvement_delta_mean"], ascending=[True, True, False]).reset_index(drop=True)


def practical_tier(delta: float, relative_pct: float, ci_low: float, ci_high: float) -> str:
    if not math.isfinite(delta):
        return "unavailable"
    if delta < 0:
        return "drop"
    if abs(delta) < 5.0e-5:
        return "tie/negligible"
    if math.isfinite(ci_low) and math.isfinite(ci_high) and ci_low <= 0.0 <= ci_high:
        return "tie/negligible" if abs(relative_pct) < 1.0 else "positive, uncertain"
    return "substantial gain" if relative_pct >= 1.0 else "marginal gain"


def write_markdown_table(delta_agg: pd.DataFrame, path: Path) -> None:
    columns = [
        "domain",
        "dataset",
        "model",
        "implementation",
        "metric_name",
        "metric_direction",
        "base",
        "warrant",
        "delta",
        "improvement_delta",
        "paired_ci95",
        "paired_t_p",
        "significant_0_05",
        "practical_tier",
        "relative_delta_pct",
        "n_seeds",
    ]
    if delta_agg.empty:
        path.write_text("No paired base/warrant results found.\n", encoding="utf-8")
        return
    table = delta_agg[columns].copy()
    path.write_text(markdown_table(table) + "\n", encoding="utf-8")


def primary_metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    counts = {}
    if not frame.empty and "domain" in frame.columns:
        counts = frame.groupby("domain").size().to_dict()
    rows = [
        {
            "domain": domain,
            "metric_name": metric,
            "metric_direction": direction,
            "completed_rows": int(counts.get(domain, 0)),
        }
        for domain, (metric, direction) in sorted(DOMAIN_PRIMARY_METRICS.items())
    ]
    return pd.DataFrame(rows)


def write_primary_metric_table(frame: pd.DataFrame, tables_dir: Path) -> None:
    table = primary_metric_table(frame)
    table.to_csv(tables_dir / "primary_metrics.csv", index=False)
    (tables_dir / "primary_metrics.md").write_text(markdown_table(table) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.astype(object).to_numpy()]
    widths = [
        max(len(columns[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(columns))
    ]
    header = "| " + " | ".join(column.ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        print(f"warning: matplotlib unavailable, skipping plots: {exc}")
        return None


def plot_global_delta(delta_agg: pd.DataFrame, out_dir: Path, metric: str, top_k: int) -> None:
    plt = try_import_matplotlib()
    if plt is None or delta_agg.empty:
        return
    plot_df = delta_agg.copy()
    plot_df["label"] = plot_df["domain"] + "/" + plot_df["dataset"] + " · " + plot_df["model"]
    plot_df = plot_df.reindex(plot_df["improvement_delta_mean"].abs().sort_values(ascending=False).index).head(top_k)
    plot_df = plot_df.sort_values("improvement_delta_mean")
    colors = np.where(plot_df["improvement_delta_mean"] >= 0, "#2f7d4f", "#b84a4a")

    height = max(5.0, 0.28 * len(plot_df))
    fig, ax = plt.subplots(figsize=(12, height))
    ax.barh(
        plot_df["label"],
        plot_df["improvement_delta_mean"],
        xerr=plot_df["improvement_delta_std"].fillna(0.0),
        color=colors,
        alpha=0.88,
    )
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.set_xlabel(f"Improvement delta ({metric}; direction-aware)")
    ax.set_ylabel("")
    ax.set_title("Base vs Warrant Paired Seed Improvement")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "delta_bar_all.png", dpi=220)
    plt.close(fig)


def plot_scatter(delta_agg: pd.DataFrame, out_dir: Path, metric: str) -> None:
    plt = try_import_matplotlib()
    if plt is None or delta_agg.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    domains = list(delta_agg["domain"].drop_duplicates())
    cmap = plt.get_cmap("tab10")
    for idx, domain in enumerate(domains):
        subset = delta_agg[delta_agg["domain"].eq(domain)]
        ax.scatter(subset["base_mean"], subset["warrant_mean"], label=domain, s=52, alpha=0.85, color=cmap(idx % 10))
    low = float(np.nanmin([delta_agg["base_mean"].min(), delta_agg["warrant_mean"].min()]))
    high = float(np.nanmax([delta_agg["base_mean"].max(), delta_agg["warrant_mean"].max()]))
    pad = max(1.0e-3, (high - low) * 0.05)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], linestyle="--", color="#555555", linewidth=1)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlabel(f"Base mean {metric}")
    ax.set_ylabel(f"Warrant mean {metric}")
    ax.set_title("Base vs Warrant Mean Performance")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "base_vs_warrant_scatter.png", dpi=220)
    plt.close(fig)


def plot_domain_heatmaps(delta_agg: pd.DataFrame, out_dir: Path, metric: str) -> None:
    plt = try_import_matplotlib()
    if plt is None or delta_agg.empty:
        return
    for domain, subset in delta_agg.groupby("domain", sort=True):
        pivot = subset.pivot_table(index="dataset", columns="model", values="improvement_delta_mean", aggfunc="mean")
        if pivot.empty:
            continue
        values = pivot.to_numpy(dtype=float)
        vmax = np.nanmax(np.abs(values))
        vmax = 1.0e-6 if not np.isfinite(vmax) or vmax == 0 else vmax
        fig_w = max(6.0, 0.8 * len(pivot.columns))
        fig_h = max(3.5, 0.55 * len(pivot.index))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    ax.text(j, i, f"{values[i, j]:+.3f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"{domain}: Direction-aware Warrant Improvement ({metric})")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"delta_heatmap_{slug(domain)}.png", dpi=220)
        plt.close(fig)


def plot_domain_bars(delta_agg: pd.DataFrame, out_dir: Path, metric: str) -> None:
    plt = try_import_matplotlib()
    if plt is None or delta_agg.empty:
        return
    for domain, subset in delta_agg.groupby("domain", sort=True):
        subset = subset.sort_values(["dataset", "model"]).copy()
        labels = subset["dataset"] + " · " + subset["model"]
        x = np.arange(len(subset))
        width = 0.38
        fig_w = max(10.0, 0.55 * len(subset))
        fig, ax = plt.subplots(figsize=(fig_w, 5.5))
        ax.bar(
            x - width / 2,
            subset["base_mean"],
            width,
            yerr=subset["base_std"].fillna(0.0),
            label="base",
            color="#6b7280",
            alpha=0.85,
        )
        ax.bar(
            x + width / 2,
            subset["warrant_mean"],
            width,
            yerr=subset["warrant_std"].fillna(0.0),
            label="warrant",
            color="#2563eb",
            alpha=0.85,
        )
        ax.set_xticks(x, labels=labels, rotation=45, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{domain}: Base vs Warrant")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"variant_bars_{slug(domain)}.png", dpi=220)
        plt.close(fig)


def write_summary(delta_agg: pd.DataFrame, out_dir: Path, metric: str) -> None:
    metric_note = "domain-aware primary metric" if metric == "primary_metric" else "explicit metric override"
    lines = [f"# Main Benchmark Aggregation", "", f"Metric: `{metric}` ({metric_note})", ""]
    if delta_agg.empty:
        lines.append("No paired base/warrant results found.")
    else:
        total = len(delta_agg)
        improved = int(delta_agg["improved"].sum())
        sign_test = stats.binomtest(improved, total, p=0.5, alternative="two-sided")
        tiers = delta_agg["practical_tier"].value_counts().to_dict()
        lines.extend(
            [
                f"- paired model/dataset groups: {total}",
                f"- direction-aware positive mean changes: {improved}/{total} ({100.0 * improved / max(1, total):.1f}%)",
                f"- exact two-sided sign test: p={sign_test.pvalue:.6f} under H0: P(positive change)=0.5",
                f"- practical tiers: {', '.join(f'{key}={value}' for key, value in sorted(tiers.items()))}",
                f"- mean improvement delta: {delta_agg['improvement_delta_mean'].mean():+.4f}",
                f"- median improvement delta: {delta_agg['improvement_delta_mean'].median():+.4f}",
                "",
                "## Top Improvements",
                "",
                markdown_table(
                    delta_agg.sort_values("improvement_delta_mean", ascending=False)
                    .head(12)[["domain", "dataset", "model", "metric_name", "metric_direction", "base", "warrant", "improvement_delta", "paired_ci95", "paired_t_p", "practical_tier", "relative_delta_pct", "n_seeds"]]
                ),
                "",
                "## Largest Drops",
                "",
                markdown_table(
                    delta_agg.sort_values("improvement_delta_mean", ascending=True)
                    .head(12)[["domain", "dataset", "model", "metric_name", "metric_direction", "base", "warrant", "improvement_delta", "paired_ci95", "paired_t_p", "practical_tier", "relative_delta_pct", "n_seeds"]]
                ),
            ]
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    tables_dir = out_dir / "tables"
    plots_dir = out_dir / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.results)
    raw.to_csv(tables_dir / "results_raw_copy.csv", index=False)
    completed = load_results(args.results, args.metric, include_incomplete=args.include_incomplete)
    if args.include_incomplete:
        completed = completed[completed["status"].eq("completed") & completed[args.metric].notna()].copy()

    variant_agg = aggregate_variants(completed, args.metric)
    paired = paired_seed_deltas(completed, args.metric, args.base, args.warrant)
    delta_agg = aggregate_deltas(paired)

    variant_agg.to_csv(tables_dir / "variant_aggregate.csv", index=False)
    paired.to_csv(tables_dir / "paired_seed_deltas.csv", index=False)
    delta_agg.to_csv(tables_dir / "base_vs_warrant.csv", index=False)
    write_primary_metric_table(completed, tables_dir)
    write_markdown_table(delta_agg, tables_dir / "base_vs_warrant.md")
    write_summary(delta_agg, out_dir, args.metric)

    plot_global_delta(delta_agg, plots_dir, args.metric, args.top_k)
    plot_scatter(delta_agg, plots_dir, args.metric)
    plot_domain_heatmaps(delta_agg, plots_dir, args.metric)
    plot_domain_bars(delta_agg, plots_dir, args.metric)

    print(f"wrote tables: {tables_dir}")
    print(f"wrote plots:  {plots_dir}")
    print(f"wrote summary: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
