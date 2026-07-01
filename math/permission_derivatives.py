#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "math" / "outputs"


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = np.array([2.0, 1.2, 0.4, -0.3])
    gate = np.array([0.9, 0.6, 0.4, 0.2])
    alpha = softmax(scores)
    warrant_mass = alpha * gate
    logit_gated_alpha = softmax(scores + np.log(gate))

    # d(alpha_j g_j) / d(log g_l) = alpha_j g_j 1[j=l]
    warrant_jac = np.diag(warrant_mass)

    # d softmax(scores + log g)_j / d log g_l
    n = len(scores)
    logit_jac = np.zeros((n, n))
    for j in range(n):
        for l in range(n):
            logit_jac[j, l] = logit_gated_alpha[j] * ((1.0 if j == l else 0.0) - logit_gated_alpha[l])

    rows = []
    for j in range(n):
        for l in range(n):
            rows.append(
                {
                    "j": j,
                    "l": l,
                    "warrant_derivative": warrant_jac[j, l],
                    "attention_logit_derivative": logit_jac[j, l],
                }
            )

    csv_path = OUTPUT_DIR / "permission_derivatives.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["j", "l", "warrant_derivative", "attention_logit_derivative"])
        writer.writeheader()
        writer.writerows(rows)

    off_diag_warrant = float(np.sum(np.abs(warrant_jac - np.diag(np.diag(warrant_jac)))))
    off_diag_logit = float(np.sum(np.abs(logit_jac - np.diag(np.diag(logit_jac)))))

    md = [
        "# Permission Derivative Demo",
        "",
        "This script compares Warrant value gating with attention-logit gating.",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        f"| Sum off-diagonal derivative, Warrant | {off_diag_warrant:.6f} |",
        f"| Sum off-diagonal derivative, attention-logit gate | {off_diag_logit:.6f} |",
        "",
        "Warrant has diagonal item-wise permission derivatives. Attention-logit gating has non-zero off-diagonal derivatives because softmax re-normalizes mass across items.",
        "",
        "## Alpha and Mass",
        "",
        "| Item | alpha | gate | alpha*g | softmax(score+log gate) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx in range(n):
        md.append(
            f"| {idx} | {alpha[idx]:.6f} | {gate[idx]:.6f} | {warrant_mass[idx]:.6f} | {logit_gated_alpha[idx]:.6f} |"
        )
    (OUTPUT_DIR / "permission_derivatives.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {OUTPUT_DIR / 'permission_derivatives.md'}")


if __name__ == "__main__":
    main()

