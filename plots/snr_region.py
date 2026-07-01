#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plots.common import PLOT_OUTPUT_DIR, save_figure, set_paper_style


DEFAULT_SIM = REPO_ROOT / "math" / "outputs" / "snr_simulation.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the SNR improvement region R_S > R_N.")
    parser.add_argument("--simulation", type=Path, default=DEFAULT_SIM)
    parser.add_argument("--output-dir", type=Path, default=PLOT_OUTPUT_DIR)
    parser.add_argument("--stem", default="snr_improvement_region")
    parser.add_argument("--show-simulation", action="store_true", help="Overlay Monte Carlo scenario points.")
    parser.add_argument("--trials", type=int, default=2500, help="Monte Carlo samples for the right panel.")
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


def load_points(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    for col in ["mean_R_S", "mean_R_N", "mean_snr_ratio", "p_snr_improves"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["mean_R_S", "mean_R_N"]).copy()


def short_label(name: str) -> str:
    mapping = {
        "weak_alignment_8_32": "weak",
        "moderate_alignment_8_32": "moderate",
        "strong_alignment_8_32": "strong",
        "moderate_alignment_4_8": "few items",
        "moderate_alignment_16_64": "many items",
        "false_suppression": "false suppress",
    }
    return mapping.get(str(name), str(name).replace("_", " "))

def random_simplex(rng: np.random.Generator, size: int, concentration: float = 1.0) -> np.ndarray:
    values = rng.gamma(concentration, 1.0, size=size)
    return values / values.sum()


def beta_params(mean: float, concentration: float) -> tuple[float, float]:
    mean = min(max(mean, 1.0e-4), 1.0 - 1.0e-4)
    return mean * concentration, (1.0 - mean) * concentration


def monte_carlo_points(*, trials: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    support_items = 8
    noise_items = 32
    # Weak but positive evidence alignment. This intentionally avoids a
    # perfect-looking simulation while still showing concentration below the
    # SNR boundary.
    support_a, support_b = beta_params(0.72, 30.0)
    noise_a, noise_b = beta_params(0.60, 30.0)
    rows = []
    for _ in range(int(trials)):
        alpha_support = random_simplex(rng, support_items)
        alpha_noise = random_simplex(rng, noise_items)
        mu = rng.lognormal(mean=0.0, sigma=0.25, size=support_items)
        sigma = rng.lognormal(mean=0.0, sigma=0.25, size=noise_items)
        support_gate = rng.beta(support_a, support_b, size=support_items)
        noise_gate = rng.beta(noise_a, noise_b, size=noise_items)

        signal_base = np.sum(alpha_support * mu)
        signal_warrant = np.sum(alpha_support * support_gate * mu)
        noise_var_base = np.sum((alpha_noise**2) * (sigma**2))
        noise_var_warrant = np.sum((alpha_noise**2) * (noise_gate**2) * (sigma**2))
        rs = signal_warrant / max(signal_base, 1.0e-12)
        rn = np.sqrt(noise_var_warrant / max(noise_var_base, 1.0e-12))
        rows.append({"R_S": rs, "R_N": rn, "improves": float(rs > rn), "snr_ratio": rs / max(rn, 1.0e-12)})
    return pd.DataFrame(rows)


def draw_regions(ax: plt.Axes) -> None:
    x = np.linspace(0.0, 1.0, 400)
    ax.fill_between(x, 0.0, x, color="#D9F0E7", alpha=0.95, zorder=0)
    ax.fill_between(x, x, 1.0, color="#F7D9D7", alpha=0.88, zorder=0)
    ax.plot(x, x, color="#333333", linewidth=1.35, linestyle="--", zorder=2)
    ax.text(
        0.72,
        0.26,
        r"$R_S > R_N$" + "\nSNR improves",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="semibold",
        color="#0F766E",
    )
    ax.text(
        0.29,
        0.71,
        r"$R_S \leq R_N$" + "\nSNR degrades",
        ha="center",
        va="center",
        fontsize=9.6,
        fontweight="semibold",
        color="#A33A35",
    )
    ax.text(
        0.55,
        0.58,
        r"$R_S=R_N$",
        ha="left",
        va="bottom",
        fontsize=8.4,
        color="#333333",
        rotation=37,
    )
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"retained support signal $R_S$")
    ax.set_ylabel(r"retained noise std. $R_N$")
    ax.grid(True, color="#FFFFFF", linewidth=1.05)
    ax.set_axisbelow(True)


def plot_region(points: pd.DataFrame, *, show_simulation: bool, trials: int, seed: int) -> plt.Figure:
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.05), sharex=True, sharey=True)
    ax = axes[0]
    draw_regions(ax)
    ax.set_title("SNR improvement condition", fontsize=10.5, fontweight="semibold", pad=8)

    # Main-paper version: a single illustrative point keeps the message legible.
    example_rs = 0.90
    example_rn = 0.50
    ax.scatter(
        [example_rs],
        [example_rn],
        s=130,
        c=["#0F766E"],
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )
    ax.annotate(
        r"example: $R_S=0.9,\ R_N=0.5$" + "\n" + r"$\mathrm{SNR}_W/\mathrm{SNR}_B=1.8$",
        xy=(0.60, 0.45),
        xytext=(0.60, 0.45),
        textcoords="data",
        fontsize=8.6,
        color="#25313B",
        ha="left",
        va="center",
        zorder=4,
    )

    if show_simulation and not points.empty:
        colors = []
        for _, row in points.iterrows():
            colors.append("#0F766E" if float(row["mean_R_S"]) > float(row["mean_R_N"]) else "#B33B35")
        sizes = 34 + 60 * points["p_snr_improves"].clip(0, 1).to_numpy()
        ax.scatter(
            points["mean_R_S"],
            points["mean_R_N"],
            s=sizes,
            c=colors,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )

    ax2 = axes[1]
    draw_regions(ax2)
    ax2.set_title("Illustrative Monte Carlo under evidence-aligned gates", fontsize=10.5, fontweight="semibold", pad=8)
    cloud = monte_carlo_points(trials=trials, seed=seed)
    improve = cloud["improves"].to_numpy(dtype=bool)
    ax2.scatter(
        cloud.loc[~improve, "R_S"],
        cloud.loc[~improve, "R_N"],
        s=8,
        c="#B33B35",
        alpha=0.22,
        linewidth=0,
        zorder=3,
    )
    ax2.scatter(
        cloud.loc[improve, "R_S"],
        cloud.loc[improve, "R_N"],
        s=8,
        c="#0F766E",
        alpha=0.22,
        linewidth=0,
        zorder=3,
    )
    p_improve = float(cloud["improves"].mean())
    median_ratio = float(cloud["snr_ratio"].median())
    ax2.text(
        0.06,
        0.94,
        f"P(SNR improves) = {p_improve:.3f}\nmedian ratio = {median_ratio:.2f}x",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#25313B",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.92},
        zorder=5,
    )
    ax2.set_ylabel("")
    fig.tight_layout(w_pad=1.7)
    return fig


def main() -> None:
    args = parse_args()
    points = load_points(args.simulation)
    fig = plot_region(points, show_simulation=args.show_simulation, trials=args.trials, seed=args.seed)
    png_path, pdf_path = save_figure(fig, args.stem, output_dir=args.output_dir)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
