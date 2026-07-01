#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plots.common import PLOT_OUTPUT_DIR, save_figure, set_paper_style


DEFAULT_STEM = "attention_permission_comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot raw attention vs Warrant permission maps.")
    parser.add_argument("--attention", type=Path, default=None, help="Optional .npy/.csv file containing the raw attention matrix")
    parser.add_argument("--permission", type=Path, default=None, help="Optional .npy/.csv file containing the Warrant permission matrix")
    parser.add_argument("--matrix-pair", type=Path, default=None, help="Optional .npz file containing attention and permission arrays")
    parser.add_argument("--demo", action="store_true", help="Use a built-in example matrix pair for a smoke test")
    parser.add_argument("--row-labels", default=None, help="Comma-separated row labels")
    parser.add_argument("--column-labels", default=None, help="Comma-separated column labels")
    parser.add_argument("--output-dir", type=Path, default=PLOT_OUTPUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--title", default="")
    return parser.parse_args()


def load_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".csv":
        return np.loadtxt(path, delimiter=",", dtype=float)
    raise ValueError(f"unsupported matrix format: {path}")


def load_matrix_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        attention = None
        permission = None
        for candidate in ("attention", "alpha", "attention_map", "attn"):
            if candidate in data:
                attention = data[candidate]
                break
        for candidate in ("permission", "gate", "warrant_gate", "permission_map", "warrant"):
            if candidate in data:
                permission = data[candidate]
                break
        if attention is None or permission is None:
            raise ValueError(f"{path} must contain attention and permission arrays")
        return np.asarray(attention, dtype=float), np.asarray(permission, dtype=float)
    if path.suffix.lower() == ".npy":
        arrays = np.load(path, allow_pickle=True)
        if isinstance(arrays, np.ndarray) and arrays.shape == ():
            raise ValueError(f"{path} must contain a matrix or a pair of arrays")
        if isinstance(arrays, np.ndarray) and arrays.ndim == 2:
            return arrays, arrays.copy()
        raise ValueError(f"{path} must be a 2D array or an .npz pair")
    if path.suffix.lower() == ".csv":
        return np.loadtxt(path, delimiter=",", dtype=float), np.loadtxt(path, delimiter=",", dtype=float)
    raise ValueError(f"unsupported matrix format: {path}")


def load_matrix_bundle(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        attention, permission = load_matrix_pair(path)
        labels = None
        for candidate in ("labels", "support_labels", "support"):
            if candidate in data:
                labels = np.asarray(data[candidate], dtype=float)
                break
        return attention, permission, labels
    attention, permission = load_matrix_pair(path)
    return attention, permission, None


def synthetic_example() -> tuple[np.ndarray, np.ndarray]:
    attention = np.array(
        [
            [0.75, 0.14, 0.06, 0.05],
            [0.11, 0.61, 0.18, 0.10],
            [0.09, 0.23, 0.54, 0.14],
        ],
        dtype=float,
    )
    permission = np.array(
        [
            [0.10, 0.91, 0.94, 0.90],
            [0.22, 0.80, 0.87, 0.84],
            [0.31, 0.24, 0.92, 0.82],
        ],
        dtype=float,
    )
    return attention, permission


def parse_labels(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    labels = [item.strip() for item in raw.split(",") if item.strip()]
    return labels or None


def highlight_blocked_cells(ax: plt.Axes, attention: np.ndarray, permission: np.ndarray) -> None:
    attn_cut = np.percentile(attention, 70)
    perm_cut = np.percentile(permission, 30)
    blocked_mask = (attention >= attn_cut) & (permission <= perm_cut)
    if not np.any(blocked_mask):
        idx = np.unravel_index(np.argmax(attention - permission), attention.shape)
        blocked_mask[idx] = True
    for row_idx, col_idx in np.argwhere(blocked_mask):
        ax.add_patch(Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor="#DC2626", linewidth=1.6))


def align_labels(labels: list[str] | None, size: int) -> list[str]:
    if labels is None:
        return [f"Item {i + 1}" for i in range(size)]
    if len(labels) == size:
        return labels
    if len(labels) < size:
        return labels + [f"Item {i + 1}" for i in range(len(labels), size)]
    return labels[:size]


def support_mask_from_labels(labels: list[str]) -> np.ndarray:
    return np.asarray(["*" in label or "support" in label.lower() for label in labels], dtype=bool)


def clean_passage_labels(labels: list[str]) -> list[str]:
    cleaned = []
    for label in labels:
        is_support = "*" in label or "support" in label.lower()
        base = label.replace("*", "").strip()
        cleaned.append(f"{base}\n{'S' if is_support else 'D'}")
    return cleaned


def plot_single_query_case(
    attention: np.ndarray,
    permission: np.ndarray,
    *,
    title: str,
    column_labels: list[str],
) -> plt.Figure:
    set_paper_style()
    alpha = np.asarray(attention[0], dtype=float)
    gate = np.asarray(permission[0], dtype=float)
    effective = alpha * gate
    x = np.arange(alpha.shape[0])
    support_mask = support_mask_from_labels(column_labels)
    labels = clean_passage_labels(column_labels)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.4, 3.85),
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.32},
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.05, hspace=0.02, wspace=0.05)

    support_color = "#DCFCE7"
    distractor_color = "#F3F4F6"
    for ax in axes:
        for idx, is_support in enumerate(support_mask):
            ax.axvspan(idx - 0.5, idx + 0.5, color=support_color if is_support else distractor_color, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.grid(axis="y", alpha=0.22)
        ax.grid(axis="x", visible=False)

    width = 0.34
    axes[0].bar(x - width / 2, alpha, width=width, color="#6BAED6", label=r"attention $\alpha$")
    axes[0].bar(x + width / 2, effective, width=width, color="#7E57C2", label=r"effective mass $\alpha g$")
    axes[0].set_title("Contribution mass")
    axes[0].set_ylabel("Mass")
    mass_max = max(float(alpha.max()), float(effective.max()), 1.0e-6)
    axes[0].set_ylim(0.0, mass_max * 1.32)
    axes[0].legend(loc="upper left", ncol=2, frameon=False)

    for idx, value in enumerate(alpha):
        axes[0].text(idx - width / 2, value + mass_max * 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=8, color="#1F4E79")
    for idx, value in enumerate(effective):
        axes[0].text(idx + width / 2, value + mass_max * 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=8, color="#4C1D95")

    bars = axes[1].bar(x, gate, width=0.52, color=np.where(support_mask, "#15803D", "#F97316"))
    axes[1].set_title("Warrant permission")
    axes[1].set_ylabel(r"Gate $g$")
    lower = max(0.0, min(float(gate.min()) - 0.08, 0.65))
    axes[1].set_ylim(lower, 1.02)
    axes[1].axhline(1.0, color="#6B7280", linewidth=0.8, linestyle="--", alpha=0.55)
    for bar, value in zip(bars, gate):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.015, 1.01),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111827",
        )

    support_alpha = float(alpha[support_mask].sum()) if bool(support_mask.any()) else float("nan")
    support_effective = float(effective[support_mask].sum()) if bool(support_mask.any()) else float("nan")
    total_alpha = float(alpha.sum())
    total_effective = float(effective.sum())
    attention_ratio = support_alpha / total_alpha if total_alpha > 0 else float("nan")
    mass_ratio = support_effective / total_effective if total_effective > 0 else float("nan")
    summary = (
        f"Support ratio: attention {attention_ratio:.3f} -> warranted mass {mass_ratio:.3f}\n"
        "S = supporting passage, D = distractor."
    )
    fig.text(0.5, -0.035, summary, ha="center", va="top", fontsize=10, color="#374151")
    if title:
        fig.suptitle(title, y=1.04, fontsize=12)
    fig.legend(
        handles=[Patch(facecolor=support_color, label="support passage"), Patch(facecolor=distractor_color, label="distractor")],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
    )
    return fig


def plot_matrix_comparison(
    attention: np.ndarray,
    permission: np.ndarray,
    *,
    title: str,
    row_labels: list[str] | None = None,
    column_labels: list[str] | None = None,
    support_labels: np.ndarray | None = None,
) -> plt.Figure:
    set_paper_style()
    attention = np.asarray(attention, dtype=float)
    permission = np.asarray(permission, dtype=float)
    if attention.shape != permission.shape:
        raise ValueError(f"attention and permission shapes must match: {attention.shape} vs {permission.shape}")

    row_labels = align_labels(row_labels, attention.shape[0])
    column_labels = align_labels(column_labels, attention.shape[1])

    if attention.shape[0] == 1:
        return plot_single_query_case(attention, permission, title=title, column_labels=column_labels)

    effective = attention * permission
    support_labels = None if support_labels is None else np.asarray(support_labels, dtype=float)
    if support_labels is not None and support_labels.shape != attention.shape:
        support_labels = None
    fig_height = max(3.4, 0.42 * attention.shape[0] + 1.25)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, fig_height), constrained_layout=True)

    def mark_support(ax: plt.Axes) -> None:
        if support_labels is None:
            return
        for row_idx, col_idx in np.argwhere(support_labels > 0.5):
            ax.add_patch(Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor="#111827", linewidth=1.15))
            ax.text(col_idx + 0.5, row_idx + 0.5, "★", ha="center", va="center", color="#111827", fontsize=8)

    for ax in axes:
        ax.set_xlabel("Retrieved passage")
        ax.set_ylabel("Question")

    mass_vmax = max(
        float(np.nanpercentile(np.concatenate([attention.ravel(), effective.ravel()]), 95)),
        float(attention.max()),
        float(effective.max()),
        1.0e-8,
    )
    permission_vmin = max(0.0, min(float(np.nanpercentile(permission, 5)) - 0.03, float(permission.min())))

    mass_cmap = "Blues"
    sns.heatmap(attention, ax=axes[0], cmap=mass_cmap, vmin=0.0, vmax=mass_vmax, cbar=True, annot=False)
    axes[0].set_title("Raw attention $\\alpha_{ij}$")
    axes[0].set_xticklabels(column_labels, rotation=0, ha="center")
    axes[0].set_yticklabels(row_labels, rotation=0)
    mark_support(axes[0])

    sns.heatmap(permission, ax=axes[1], cmap="YlGn", vmin=permission_vmin, vmax=1.0, cbar=True, annot=False)
    axes[1].set_title("Warrant permission $g_{ij}$")
    axes[1].set_xticklabels(column_labels, rotation=0, ha="center")
    axes[1].set_yticklabels(row_labels, rotation=0)
    mark_support(axes[1])

    sns.heatmap(effective, ax=axes[2], cmap=mass_cmap, vmin=0.0, vmax=mass_vmax, cbar=True, annot=False)
    axes[2].set_title("Effective mass $\\alpha_{ij} g_{ij}$")
    axes[2].set_xticklabels(column_labels, rotation=0, ha="center")
    axes[2].set_yticklabels(row_labels, rotation=0)
    mark_support(axes[2])

    if title:
        fig.suptitle(title, y=1.02, fontsize=12)
    fig.text(
        0.5,
        -0.01,
        "Outlined ★ cells are annotated supporting passages. Raw attention and effective mass use the same color scale.",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    return fig


def main() -> None:
    args = parse_args()
    row_labels = parse_labels(args.row_labels)
    column_labels = parse_labels(args.column_labels)

    support_labels = None
    if args.demo:
        attention, permission = synthetic_example()
    elif args.matrix_pair is not None:
        attention, permission, support_labels = load_matrix_bundle(args.matrix_pair)
    elif args.attention is not None and args.permission is not None:
        attention = load_matrix(args.attention)
        permission = load_matrix(args.permission)
    else:
        raise SystemExit("provide --attention and --permission, or --matrix-pair, or pass --demo")

    fig = plot_matrix_comparison(
        attention,
        permission,
        title=args.title,
        row_labels=row_labels,
        column_labels=column_labels,
        support_labels=support_labels,
    )
    png_path, pdf_path = save_figure(fig, args.stem, output_dir=args.output_dir)
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
