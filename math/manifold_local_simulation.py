#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "math" / "outputs"


@dataclass(frozen=True)
class Regime:
    name: str
    tangent_dim: int
    items: int
    harmful_fraction: float
    common_strength: float
    perp_noise: float
    beta_m: float
    gate_leak: float


REGIMES = [
    Regime(
        name="low_entanglement",
        tangent_dim=32,
        items=48,
        harmful_fraction=0.45,
        common_strength=0.05,
        perp_noise=0.70,
        beta_m=1.0,
        gate_leak=0.05,
    ),
    Regime(
        name="moderate_entanglement",
        tangent_dim=32,
        items=48,
        harmful_fraction=0.45,
        common_strength=0.35,
        perp_noise=0.55,
        beta_m=1.0,
        gate_leak=0.05,
    ),
    Regime(
        name="high_entanglement",
        tangent_dim=32,
        items=48,
        harmful_fraction=0.45,
        common_strength=0.85,
        perp_noise=0.30,
        beta_m=1.0,
        gate_leak=0.05,
    ),
    Regime(
        name="high_curvature_entangled_counterexample",
        tangent_dim=32,
        items=48,
        harmful_fraction=0.45,
        common_strength=0.95,
        perp_noise=0.15,
        beta_m=4.0,
        gate_leak=0.05,
    ),
]


def unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm < 1.0e-12:
        return x
    return x / norm


def make_tangent_contributions(regime: Regime, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return tangent contribution matrix Z and harmful mask.

    The first tangent coordinate is the local Riemannian gradient direction.
    Positive projection on this coordinate is loss-increasing and should be
    attenuated. Negative projection is helpful and should be retained.
    """

    r = regime.tangent_dim
    n = regime.items
    harmful_count = int(round(n * regime.harmful_fraction))
    harmful = np.zeros(n, dtype=bool)
    harmful[:harmful_count] = True
    rng.shuffle(harmful)

    common = unit(rng.normal(size=r))
    common[0] = 0.0
    common = unit(common)

    z = np.zeros((n, r), dtype=float)
    for idx in range(n):
        direction_sign = 1.0 if harmful[idx] else -1.0
        along_grad = direction_sign * rng.uniform(0.45, 1.25)
        independent = rng.normal(size=r)
        independent[0] = 0.0
        independent = unit(independent)
        tangent_vec = np.zeros(r, dtype=float)
        tangent_vec[0] = along_grad
        tangent_vec += regime.common_strength * common
        tangent_vec += regime.perp_noise * independent
        tangent_vec *= rng.lognormal(mean=0.0, sigma=0.15)
        z[idx] = tangent_vec
    return z, harmful


def bound_change(delta: np.ndarray, a: np.ndarray, gram: np.ndarray, beta_m: float) -> float:
    # Local bound difference B(delta)-B(0).
    return float(-delta @ a + 0.5 * beta_m * delta @ gram @ delta)


def run_regime(regime: Regime, *, trials: int, seed: int) -> dict[str, float | str | int]:
    rng = np.random.default_rng(seed)
    max_delta = 1.0 - regime.gate_leak

    improved = []
    bound_reductions = []
    diagonal_bound_reductions = []
    offdiag_ratios = []
    harmful_deltas = []
    helpful_deltas = []
    false_suppression_rates = []
    cross_penalties = []
    cross_to_diag_ratios = []

    for _ in range(trials):
        z, harmful = make_tangent_contributions(regime, rng)
        # In the theory, the controlled contribution is c_j = alpha_j v_j.
        # Multiplying by attention weights is essential; otherwise the
        # tangent Gram cross terms are unrealistically large for many items.
        alpha = rng.gamma(shape=4.0, scale=1.0, size=regime.items)
        alpha = alpha / alpha.sum()
        z = z * alpha[:, None]
        a = z[:, 0].copy()
        gram = z @ z.T
        diag = np.diag(gram)

        # Item-wise Warrant is a diagonal approximation to the tangent quadratic.
        delta = np.clip(a / (regime.beta_m * diag + 1.0e-8), 0.0, max_delta)

        full_change = bound_change(delta, a, gram, regime.beta_m)
        diag_change = float(np.sum(-delta * a + 0.5 * regime.beta_m * delta**2 * diag))
        offdiag_change = full_change - diag_change

        gram_norm = np.linalg.norm(gram, ord="fro")
        offdiag = gram - np.diag(diag)
        offdiag_ratio = float(np.linalg.norm(offdiag, ord="fro") / max(gram_norm, 1.0e-12))

        improved.append(float(full_change < 0.0))
        bound_reductions.append(float(-full_change))
        diagonal_bound_reductions.append(float(-diag_change))
        offdiag_ratios.append(offdiag_ratio)
        harmful_deltas.append(float(np.mean(delta[harmful])) if harmful.any() else 0.0)
        helpful_deltas.append(float(np.mean(delta[~harmful])) if (~harmful).any() else 0.0)
        false_suppression_rates.append(float(np.mean(delta[~harmful] > 1.0e-4)) if (~harmful).any() else 0.0)
        cross_penalties.append(float(offdiag_change))
        cross_to_diag_ratios.append(float(offdiag_change / max(-diag_change, 1.0e-12)))

    return {
        "regime": regime.name,
        "trials": trials,
        "tangent_dim": regime.tangent_dim,
        "items": regime.items,
        "harmful_fraction": regime.harmful_fraction,
        "common_strength": regime.common_strength,
        "perp_noise": regime.perp_noise,
        "p_local_bound_improves": float(np.mean(improved)),
        "mean_local_bound_reduction": float(np.mean(bound_reductions)),
        "mean_diagonal_bound_reduction": float(np.mean(diagonal_bound_reductions)),
        "mean_cross_penalty": float(np.mean(cross_penalties)),
        "mean_cross_to_diagonal_ratio": float(np.mean(cross_to_diag_ratios)),
        "mean_offdiag_gram_ratio": float(np.mean(offdiag_ratios)),
        "mean_harmful_shrinkage_delta": float(np.mean(harmful_deltas)),
        "mean_helpful_shrinkage_delta": float(np.mean(helpful_deltas)),
        "mean_false_suppression_rate": float(np.mean(false_suppression_rates)),
    }


def write_markdown(rows: list[dict[str, float | str | int]], path: Path) -> None:
    lines = [
        "# Manifold-Local Simulation",
        "",
        "This diagnostic simulates Warrant as a diagonal shrinkage rule on tangent-space weighted value terms.",
        "The local Riemannian smoothness bound is written as",
        "",
        "\\[",
        "B(\\delta)-B(0)=-\\delta^\\top a+\\frac{\\beta_{\\mathcal M}}{2}\\delta^\\top G\\delta,",
        "\\]",
        "",
        "where `a` is the tangent loss-gradient alignment and `G` is the tangent Gram matrix of weighted value terms.",
        "The item-wise Warrant rule uses only the diagonal of `G`; large off-diagonal mass means the path is entangled.",
        "",
        "| Regime | P(bound improves) | Bound reduction | Diagonal reduction | Cross penalty | Cross/Diagonal | Harmful shrinkage | Helpful shrinkage | False suppression |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {regime} | {p_local_bound_improves:.3f} | {mean_local_bound_reduction:.4f} | "
            "{mean_diagonal_bound_reduction:.4f} | {mean_cross_penalty:.4f} | "
            "{mean_cross_to_diagonal_ratio:.3f} | {mean_harmful_shrinkage_delta:.3f} | "
            "{mean_helpful_shrinkage_delta:.3f} | {mean_false_suppression_rate:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Reading: the manifold-local argument supports local stability, not global superiority.",
            "As tangent Gram off-diagonal mass increases, the diagonal item-wise gate becomes a rougher approximation and the cross penalty grows.",
            "The counterexample row shows that high curvature plus strong tangent entanglement can overturn the diagonal shrinkage benefit.",
            "This is the mathematical failure mode behind entangled negative rows: useful and harmful directions can share the same tangent subspace.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [run_regime(regime, trials=5000, seed=101 + idx) for idx, regime in enumerate(REGIMES)]

    csv_path = OUTPUT_DIR / "manifold_local_simulation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = OUTPUT_DIR / "manifold_local_simulation.md"
    write_markdown(rows, md_path)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
