#!/usr/bin/env python3
"""Exact small-coalition audit of item-wise attention contributions.

The audit changes no sequence tokens, attention logits, or model parameters.
It masks selected post-softmax weighted value terms at the final attention
aggregation feeding the reported prediction, then enumerates every subset of
the selected items. The resulting utility table supports exact LOO, Shapley,
pair-interaction, and sign-reversal diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import RunSpec, binary_auc, build_benchmark_model  # noqa: E402
from experiments.neural_dissection.run import (  # noqa: E402
    CTDGRunner,
    MTPPRunner,
    RAGRunner,
    STPPRunner,
    TKGRunner,
    set_seed,
)


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
RUNNERS = {
    "ctdg": CTDGRunner,
    "mtpp": MTPPRunner,
    "rag": RAGRunner,
    "stpp": STPPRunner,
    "tkg": TKGRunner,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--domain", default="all", help="Domain, comma list, or all")
    parser.add_argument("--examples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def make_spec(payload: dict[str, Any]) -> RunSpec:
    values = dict(payload)
    values["dataset_path"] = Path(values["dataset_path"])
    return RunSpec(**values)


def target_attention(model: torch.nn.Module, domain: str) -> torch.nn.Module:
    if domain in {"ctdg", "rag"}:
        return model.attn
    if domain in {"mtpp", "stpp"}:
        return model.layers[-1].attn
    if domain == "tkg":
        return model.hop_attn[-1]
    raise KeyError(domain)


class ContributionMaskController:
    """Temporarily masks weighted value terms without softmax renormalization."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.module = module
        self.original_forward = module.forward
        self.item_mask: torch.Tensor | None = None
        self.last_attention: torch.Tensor | None = None
        self.last_query: torch.Tensor | None = None
        self.last_keys: torch.Tensor | None = None
        self.last_values: torch.Tensor | None = None

        def patched(query: torch.Tensor, memory: torch.Tensor, *, mask: torch.Tensor | None = None):
            squeeze = query.ndim == 2
            query_3d = query.unsqueeze(1) if squeeze else query
            q = module.query_proj(query_3d)
            k = module.key_proj(memory)
            v = module.value_proj(memory)
            logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(float(k.shape[-1]))
            pair_mask = None
            if mask is not None:
                pair_mask = mask.unsqueeze(1) if mask.ndim == 2 else mask
                logits = logits.masked_fill(~pair_mask, torch.finfo(logits.dtype).min)
                empty = ~pair_mask.any(dim=-1, keepdim=True)
                logits = torch.where(empty, torch.zeros_like(logits), logits)
            attention = torch.softmax(logits, dim=-1)
            if pair_mask is not None:
                attention = attention * pair_mask.to(attention.dtype)
            self.last_attention = (attention[:, 0] if squeeze else attention[:, -1]).detach()
            self.last_query = (q[:, 0] if squeeze else q[:, -1]).detach()
            self.last_keys = k.detach()
            self.last_values = v.detach()

            effective = attention
            if self.item_mask is not None:
                if self.item_mask.shape != (attention.shape[0], attention.shape[-1]):
                    raise ValueError(
                        f"item mask {tuple(self.item_mask.shape)} does not match "
                        f"attention {(attention.shape[0], attention.shape[-1])}"
                    )
                effective = attention.clone()
                if squeeze:
                    effective[:, 0, :] = effective[:, 0, :] * self.item_mask.to(effective.dtype)
                else:
                    effective[:, -1, :] = effective[:, -1, :] * self.item_mask.to(effective.dtype)
            out = torch.matmul(module.dropout(effective), v)
            gate = torch.ones_like(effective)
            zeros = torch.zeros_like(effective)
            if squeeze:
                out = out.squeeze(1)
                effective = effective.squeeze(1)
                gate = gate.squeeze(1)
                zeros = zeros.squeeze(1)
            return out, {"attention": effective, "warrant_gate": gate, "warrant_logits": zeros}

        module.forward = patched

    def close(self) -> None:
        self.module.forward = self.original_forward


def repeat_batch(items: tuple[torch.Tensor, ...], repeats: int) -> tuple[torch.Tensor, ...]:
    return tuple(item.repeat((repeats,) + (1,) * (item.ndim - 1)) for item in items)


def domain_batch(runner: Any, domain: str, row: int, *, negative: bool = False):
    rows = np.asarray([row], dtype=np.int64)
    if domain == "ctdg":
        inputs = runner.make_batch(rows, negative=negative)
        target = torch.tensor([0.0 if negative else 1.0], device=runner.device)
        return inputs, target
    batch = runner.make_batch(rows)
    if domain == "mtpp":
        marks, times, target, _dtime, mask = batch
        return (marks, times, mask), target
    if domain == "stpp":
        times, coords, marks, _target, target_coords, _dtime, mask = batch
        return (times, coords, marks, mask), target_coords
    if domain == "tkg":
        return tuple(batch[:-1]), batch[-1]
    if domain == "rag":
        question, passages, mask, target = batch
        return (question, passages, mask), target
    raise KeyError(domain)


def outputs_and_utility(domain: str, result: Any, target: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    if domain == "ctdg":
        labels = target.expand_as(result.logits)
        utility = -F.binary_cross_entropy_with_logits(result.logits, labels, reduction="none")
        primary = torch.sigmoid(result.logits)
    elif domain in {"mtpp", "tkg"}:
        repeated = target.expand(result.logits.shape[0])
        utility = -F.cross_entropy(result.logits, repeated, reduction="none")
        target_logits = result.logits.gather(1, repeated.unsqueeze(1))
        rank = (result.logits > target_logits).sum(dim=1).float() + 1.0
        primary = 1.0 / rank
    elif domain == "stpp":
        repeated = target.expand(result.aux["next_location"].shape[0], -1)
        error = torch.sqrt(((result.aux["next_location"] - repeated) ** 2).sum(dim=-1))
        utility = -error
        primary = error
    elif domain == "rag":
        repeated = target.expand(result.aux["passage_logits"].shape[0], -1)
        valid = torch.isfinite(result.aux["passage_logits"])
        logits = result.aux["passage_logits"].masked_fill(~valid, torch.finfo(result.aux["passage_logits"].dtype).min)
        log_probs = torch.log_softmax(logits, dim=-1)
        positive = repeated.gt(0.5) & valid
        utility = (log_probs * positive.to(log_probs.dtype)).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
        best_gold = logits.masked_fill(~positive, torch.finfo(logits.dtype).min).max(dim=1).values
        rank = ((logits > best_gold.unsqueeze(1)) & valid).sum(dim=1).float() + 1.0
        primary = 1.0 / rank
    else:
        raise KeyError(domain)
    return utility.detach().cpu().numpy(), primary.detach().cpu().numpy()


def exact_shapley(utility: np.ndarray, k: int) -> np.ndarray:
    values = np.zeros(k, dtype=np.float64)
    denom = math.factorial(k)
    for j in range(k):
        bit = 1 << j
        for subset in range(1 << k):
            if subset & bit:
                continue
            size = subset.bit_count()
            weight = math.factorial(size) * math.factorial(k - size - 1) / denom
            values[j] += weight * (utility[subset | bit] - utility[subset])
    return values


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def interaction_stats(utility: np.ndarray, k: int) -> tuple[float, float]:
    interactions: list[float] = []
    for j in range(k):
        for l in range(j + 1, k):
            bits = (1 << j) | (1 << l)
            for subset in range(1 << k):
                if subset & bits:
                    continue
                value = utility[subset | bits] - utility[subset | (1 << j)] - utility[subset | (1 << l)] + utility[subset]
                interactions.append(float(value))
    if not interactions:
        return 0.0, 0.0
    values = np.asarray(interactions)
    return float(np.mean(np.abs(values))), float(np.max(np.abs(values)))


def audit_example(
    model: torch.nn.Module,
    controller: ContributionMaskController,
    domain: str,
    inputs: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    top_k: int,
    tolerance: float,
) -> dict[str, Any] | None:
    controller.item_mask = None
    baseline = model(*inputs)
    raw_attention = controller.last_attention[0]
    valid = raw_attention > 0
    valid_indices = torch.where(valid)[0]
    if valid_indices.numel() == 0:
        return None
    k = min(top_k, int(valid_indices.numel()))
    selected = valid_indices[torch.topk(raw_attention[valid_indices], k=k).indices]
    subset_count = 1 << k
    masks = valid.to(torch.float32).unsqueeze(0).repeat(subset_count, 1)
    for subset in range(subset_count):
        bits = torch.tensor([(subset >> j) & 1 for j in range(k)], device=masks.device, dtype=masks.dtype)
        masks[subset, selected] = bits

    controller.item_mask = masks
    repeated_inputs = repeat_batch(inputs, subset_count)
    result = model(*repeated_inputs)
    utility, primary = outputs_and_utility(domain, result, target)
    controller.item_mask = None
    full = subset_count - 1
    shapley = exact_shapley(utility, k)
    loo = np.asarray([utility[full] - utility[full ^ (1 << j)] for j in range(k)])
    reversal = []
    marginal_spreads = []
    for j in range(k):
        bit = 1 << j
        marginal = np.asarray([utility[s | bit] - utility[s] for s in range(subset_count) if not s & bit])
        reversal.append(bool(marginal.min() < -tolerance and marginal.max() > tolerance))
        marginal_spreads.append(float(marginal.max() - marginal.min()))
    mean_interaction, max_interaction = interaction_stats(utility, k)
    attention = raw_attention[selected].detach().cpu().numpy()
    oracle = int(np.argmax(utility))
    top_attention = int(np.argmax(attention))
    return {
        "k": k,
        "full_utility": float(utility[full]),
        "empty_selected_utility": float(utility[0]),
        "oracle_utility": float(utility[oracle]),
        "oracle_headroom": float(utility[oracle] - utility[full]),
        "full_primary": float(primary[full]),
        "oracle_primary": float(primary[oracle]),
        "oracle_subset": oracle,
        "attention_shapley_spearman": spearman(attention, shapley),
        "top_attention_harmful": float(shapley[top_attention] < -tolerance),
        "harmful_item_fraction": float(np.mean(shapley < -tolerance)),
        "sign_reversal_fraction": float(np.mean(reversal)),
        "mean_marginal_spread": float(np.mean(marginal_spreads)),
        "mean_abs_pair_interaction": mean_interaction,
        "max_abs_pair_interaction": max_interaction,
        "mean_abs_loo_shapley_gap": float(np.mean(np.abs(loo - shapley))),
        "attention": json.dumps(attention.tolist()),
        "loo": json.dumps(loo.tolist()),
        "shapley": json.dumps(shapley.tolist()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def summarize(domain_rows: list[dict[str, Any]], domain: str) -> dict[str, Any]:
    keys = [
        "oracle_headroom",
        "attention_shapley_spearman",
        "top_attention_harmful",
        "harmful_item_fraction",
        "sign_reversal_fraction",
        "mean_marginal_spread",
        "mean_abs_pair_interaction",
        "max_abs_pair_interaction",
        "mean_abs_loo_shapley_gap",
    ]
    summary = {"domain": domain, "examples": len(domain_rows)}
    summary.update({key: finite_mean(domain_rows, key) for key in keys})
    if domain == "ctdg":
        labels = [float(row["label"]) for row in domain_rows]
        base = [float(row["full_primary"]) for row in domain_rows]
        oracle = [float(row["oracle_primary"]) for row in domain_rows]
        summary["base_primary"] = binary_auc(labels, base)
        summary["oracle_primary"] = binary_auc(labels, oracle)
        summary["directional_oracle_gain"] = summary["oracle_primary"] - summary["base_primary"]
    else:
        summary["base_primary"] = finite_mean(domain_rows, "full_primary")
        summary["oracle_primary"] = finite_mean(domain_rows, "oracle_primary")
        if domain == "stpp":
            summary["directional_oracle_gain"] = summary["base_primary"] - summary["oracle_primary"]
        else:
            summary["directional_oracle_gain"] = summary["oracle_primary"] - summary["base_primary"]
    return summary


def markdown_report(summaries: list[dict[str, Any]], config: dict[str, Any]) -> str:
    lines = [
        "# Marginal Value Audit",
        "",
        "This is a frozen-model diagnostic. It enumerates every subset of the top-attended items while keeping all other valid items as fixed background. The oracle column is label-informed diagnostic headroom, not a deployable result.",
        "",
        f"Top-k: `{config['audit']['top_k']}`; requested examples/domain: `{config['audit']['examples_per_domain']}`.",
        "",
        "| Domain | N | Base primary | Oracle primary | Directional gain | Sign reversal | |Interaction| | Attn-Shapley rho | Harmful top-attn |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['domain'].upper()} | {row['examples']} | {row['base_primary']:.4f} | "
            f"{row['oracle_primary']:.4f} | {row['directional_oracle_gain']:+.4f} | "
            f"{row['sign_reversal_fraction']:.3f} | {row['mean_abs_pair_interaction']:.4f} | "
            f"{row['attention_shapley_spearman']:.3f} | {row['top_attention_harmful']:.3f} |"
        )
    lines += [
        "",
        "## Go / No-Go",
        "",
        "The hypothesis receives initial support when sign reversals or pair interactions are non-zero, attention and Shapley utility are imperfectly aligned, and subset selection exposes positive utility headroom. A zero-interaction, rank-aligned result would reject the need for a set-aware router in this setting.",
        "",
        "RAG caveat: the audited final attention path is only the attention-mediated context/mass path; the passage scorer also receives passage representations directly.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.examples is not None:
        config["audit"]["examples_per_domain"] = args.examples
    if args.top_k is not None:
        config["audit"]["top_k"] = args.top_k
    device = torch.device(args.device or config["experiment"].get("device", "cpu"))
    requested = list(config["checkpoints"]) if args.domain == "all" else [item.strip() for item in args.domain.split(",")]
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for domain in requested:
        domain_rows: list[dict[str, Any]] = []
        for seed in config["experiment"].get("seeds", [7]):
            set_seed(int(seed))
            checkpoint_path = REPO_ROOT / config["checkpoints"][domain].format(seed=seed)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            spec = make_spec(checkpoint["spec"])
            runner_config = checkpoint["config"]
            runner_config.setdefault("training", {})["progress"] = False
            runner = RUNNERS[domain](spec, runner_config, device)
            frame = runner.prepare()
            model = build_benchmark_model(spec, frame, runner_config, device)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            controller = ContributionMaskController(target_attention(model, domain))
            eval_rows = np.asarray(runner.eval_idx, dtype=np.int64)
            count = int(config["audit"]["examples_per_domain"])
            if domain == "ctdg":
                source = eval_rows[: max(1, (count + 1) // 2)]
                jobs = ([(int(row), False) for row in source] + [(int(row), True) for row in source])[:count]
            else:
                jobs = [(int(row), False) for row in eval_rows[:count]]
            with torch.no_grad():
                for example_id, (row, negative) in enumerate(jobs):
                    inputs, target = domain_batch(runner, domain, row, negative=negative)
                    result = audit_example(model, controller, domain, inputs, target, int(config["audit"]["top_k"]), float(config["audit"].get("zero_tolerance", 1.0e-6)))
                    if result is None:
                        continue
                    result = {"domain": domain, "dataset": spec.dataset, "model": spec.model, "seed": seed, "example_id": example_id, "source_row": row, "label": float(target.flatten()[0].item()) if domain == "ctdg" else "", **result}
                    domain_rows.append(result)
                    if (example_id + 1) % 16 == 0 or example_id + 1 == len(jobs):
                        print(f"{domain} seed={seed} {example_id + 1}/{len(jobs)}", flush=True)
            controller.close()
        all_rows.extend(domain_rows)
        summaries.append(summarize(domain_rows, domain))

    output_root = REPO_ROOT / config["experiment"]["output_root"]
    write_csv(output_root / "per_example.csv", all_rows)
    write_csv(output_root / "summary.csv", summaries)
    report = markdown_report(summaries, config)
    (output_root / "analysis").mkdir(parents=True, exist_ok=True)
    (output_root / "analysis" / "marginal_value_audit.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
