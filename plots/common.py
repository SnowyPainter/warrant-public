from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_OUTPUT_DIR = REPO_ROOT / "plots" / "outputs"
# Default colormap for diverging heatmaps used in plots
HEATMAP_CMAP = "RdBu_r"


def set_paper_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def save_figure(fig: plt.Figure, stem: str, *, output_dir: Path | None = None) -> tuple[Path, Path]:
    target = output_dir or PLOT_OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    png_path = target / f"{stem}.png"
    pdf_path = target / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path
