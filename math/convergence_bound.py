#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "math" / "outputs"


def stationary_bound(gap: float, lipschitz: float, noise_var: float, t_steps: int) -> tuple[float, float]:
    eta = min(1.0 / lipschitz, 1.0 / (t_steps**0.5))
    bound = 2.0 * gap / (eta * t_steps) + eta * lipschitz * noise_var
    return eta, bound


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    leak_rows = []
    for gate_leak in [0.0, 0.01, 0.05, 0.10, 0.25, 0.50]:
        max_gate_derivative = (1.0 - gate_leak) / 4.0
        max_single_item_shrinkage = 1.0 - gate_leak
        leak_rows.append(
            {
                "gate_leak": gate_leak,
                "gate_range": f"[{gate_leak:.2f}, 1.00]",
                "max_abs_dg_dpsi": max_gate_derivative,
                "max_single_item_shrinkage": max_single_item_shrinkage,
            }
        )

    convergence_rows = []
    for lipschitz in [0.5, 1.0, 2.0]:
        for t_steps in [100, 300, 1000, 3000, 10000, 30000]:
            eta, bound = stationary_bound(gap=1.0, lipschitz=lipschitz, noise_var=0.1, t_steps=t_steps)
            convergence_rows.append(
                {
                    "objective_gap": 1.0,
                    "lipschitz_L_W": lipschitz,
                    "noise_variance": 0.1,
                    "steps_T": t_steps,
                    "eta": eta,
                    "avg_grad_norm_sq_bound": bound,
                }
            )

    leak_csv = OUTPUT_DIR / "leak_sigmoid_bounds.csv"
    with leak_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(leak_rows[0].keys()))
        writer.writeheader()
        writer.writerows(leak_rows)

    conv_csv = OUTPUT_DIR / "convergence_bound.csv"
    with conv_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(convergence_rows[0].keys()))
        writer.writeheader()
        writer.writerows(convergence_rows)

    md_lines = [
        "# Warrant Smoothness and Stationary Convergence Bounds",
        "",
        "The leak-sigmoid gate is",
        "",
        "\\[",
        "g_j=\\lambda+(1-\\lambda)\\sigma(\\psi_j),",
        "\\]",
        "",
        "so `g_j` is bounded and its derivative is uniformly bounded by `(1-lambda)/4`.",
        "",
        "## Leak-Sigmoid Bounds",
        "",
        "| gate_leak | Gate range | max abs dg/dpsi | max shrinkage |",
        "| ---: | --- | ---: | ---: |",
    ]
    for row in leak_rows:
        md_lines.append(
            "| {gate_leak:.2f} | {gate_range} | {max_abs_dg_dpsi:.4f} | {max_single_item_shrinkage:.4f} |".format(
                **row
            )
        )

    md_lines.extend(
        [
            "",
            "## Smooth Nonconvex SGD Bound",
            "",
            "For an `L_W`-smooth objective with bounded stochastic-gradient variance `sigma^2`, SGD satisfies",
            "",
            "\\[",
            "\\frac1T\\sum_{t=0}^{T-1}\\mathbb E\\|\\nabla J(\\Theta_t)\\|^2",
            "\\le",
            "\\frac{2(J(\\Theta_0)-J_*)}{\\eta T}+\\eta L_W\\sigma^2.",
            "\\]",
            "",
            "The numbers below instantiate the bound with objective gap `1.0`, noise variance `0.1`, and `eta=min(1/L_W,1/sqrt(T))`.",
            "",
            "| L_W | T | eta | Average gradient norm squared bound |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in convergence_rows:
        if row["lipschitz_L_W"] == 1.0:
            md_lines.append(
                "| {lipschitz_L_W:.1f} | {steps_T} | {eta:.5f} | {avg_grad_norm_sq_bound:.5f} |".format(
                    **row
                )
            )
    md_lines.extend(
        [
            "",
            "Reading: this is not a global-optimum guarantee. It states that Warrant remains a smooth bounded neural-network parameterization, so the usual first-order stationary convergence argument still applies.",
        ]
    )

    md_path = OUTPUT_DIR / "convergence_bound.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"wrote {leak_csv}")
    print(f"wrote {conv_csv}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
