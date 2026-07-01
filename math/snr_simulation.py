#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "math" / "outputs"


@dataclass(frozen=True)
class Scenario:
    name: str
    support_gate_mean: float
    noise_gate_mean: float
    gate_concentration: float
    support_items: int
    noise_items: int


SCENARIOS = [
    Scenario("weak_alignment_8_32", 0.72, 0.60, 30.0, 8, 32),
    Scenario("moderate_alignment_8_32", 0.85, 0.50, 30.0, 8, 32),
    Scenario("strong_alignment_8_32", 0.92, 0.35, 30.0, 8, 32),
    Scenario("moderate_alignment_4_8", 0.85, 0.50, 30.0, 4, 8),
    Scenario("moderate_alignment_16_64", 0.85, 0.50, 30.0, 16, 64),
    Scenario("false_suppression", 0.45, 0.80, 30.0, 8, 32),
]


def beta_params(mean: float, concentration: float) -> tuple[float, float]:
    mean = min(max(mean, 1.0e-4), 1.0 - 1.0e-4)
    return mean * concentration, (1.0 - mean) * concentration


def random_simplex(rng: np.random.Generator, size: int, concentration: float = 1.0) -> np.ndarray:
    values = rng.gamma(concentration, 1.0, size=size)
    return values / values.sum()


def run_scenario(scenario: Scenario, *, trials: int, seed: int) -> dict[str, float | str | int]:
    rng = np.random.default_rng(seed)
    support_a, support_b = beta_params(scenario.support_gate_mean, scenario.gate_concentration)
    noise_a, noise_b = beta_params(scenario.noise_gate_mean, scenario.gate_concentration)

    wins = []
    rs_values = []
    rn_values = []
    snr_ratios = []
    support_gate_means = []
    noise_gate_rms = []

    for _ in range(trials):
        alpha_support = random_simplex(rng, scenario.support_items)
        alpha_noise = random_simplex(rng, scenario.noise_items)
        mu = rng.lognormal(mean=0.0, sigma=0.25, size=scenario.support_items)
        sigma = rng.lognormal(mean=0.0, sigma=0.25, size=scenario.noise_items)

        support_gate = rng.beta(support_a, support_b, size=scenario.support_items)
        noise_gate = rng.beta(noise_a, noise_b, size=scenario.noise_items)

        signal_base = np.sum(alpha_support * mu)
        signal_warrant = np.sum(alpha_support * support_gate * mu)
        noise_var_base = np.sum((alpha_noise**2) * (sigma**2))
        noise_var_warrant = np.sum((alpha_noise**2) * (noise_gate**2) * (sigma**2))

        rs = signal_warrant / max(signal_base, 1.0e-12)
        rn = np.sqrt(noise_var_warrant / max(noise_var_base, 1.0e-12))
        snr_ratio = rs / max(rn, 1.0e-12)

        rs_values.append(rs)
        rn_values.append(rn)
        snr_ratios.append(snr_ratio)
        wins.append(float(snr_ratio > 1.0))
        support_gate_means.append(float(np.mean(support_gate)))
        noise_gate_rms.append(float(np.sqrt(np.mean(noise_gate**2))))

    return {
        "scenario": scenario.name,
        "support_items": scenario.support_items,
        "noise_items": scenario.noise_items,
        "target_support_gate_mean": scenario.support_gate_mean,
        "target_noise_gate_mean": scenario.noise_gate_mean,
        "observed_support_gate_mean": float(np.mean(support_gate_means)),
        "observed_noise_gate_rms": float(np.mean(noise_gate_rms)),
        "mean_R_S": float(np.mean(rs_values)),
        "mean_R_N": float(np.mean(rn_values)),
        "mean_snr_ratio": float(np.mean(snr_ratios)),
        "median_snr_ratio": float(np.median(snr_ratios)),
        "p_snr_improves": float(np.mean(wins)),
    }


def write_markdown(rows: list[dict[str, float | str | int]], path: Path) -> None:
    lines = [
        "# SNR Simulation",
        "",
        "Monte Carlo check for the theorem `SNR_W > SNR_B iff R_S > R_N` under evidence-aligned gates.",
        "",
        "| Scenario | Support items | Noise items | E[g_S] | RMS(g_N) | R_S | R_N | Mean SNR Ratio | P(SNR improves) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {support_items} | {noise_items} | {observed_support_gate_mean:.3f} | "
            "{observed_noise_gate_rms:.3f} | {mean_R_S:.3f} | {mean_R_N:.3f} | "
            "{mean_snr_ratio:.3f} | {p_snr_improves:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Reading: SNR improves when retained support signal `R_S` exceeds retained noise standard deviation `R_N`.",
            "The `false_suppression` row is an explicit counterexample where support gates are lower than noise gates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [run_scenario(s, trials=20000, seed=17 + idx) for idx, s in enumerate(SCENARIOS)]
    csv_path = OUTPUT_DIR / "snr_simulation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, OUTPUT_DIR / "snr_simulation.md")
    print(f"wrote {csv_path}")
    print(f"wrote {OUTPUT_DIR / 'snr_simulation.md'}")


if __name__ == "__main__":
    main()
