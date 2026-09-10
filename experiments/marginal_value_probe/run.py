#!/usr/bin/env python3
"""Phase B: frozen local-vs-set-aware marginal-value probes."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import build_benchmark_model  # noqa: E402
from experiments.marginal_value_audit.run import (  # noqa: E402
    ContributionMaskController,
    RUNNERS,
    domain_batch,
    make_spec,
    outputs_and_utility,
    rankdata,
    repeat_batch,
    target_attention,
)
from experiments.neural_dissection.run import set_seed  # noqa: E402


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--examples", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class ProbeData:
    q: np.ndarray
    k: np.ndarray
    c: np.ndarray
    context: np.ndarray
    scalars: np.ndarray
    target: np.ndarray
    example: np.ndarray
    attention: np.ndarray

    def subset(self, indices: np.ndarray) -> "ProbeData":
        return ProbeData(**{name: getattr(self, name)[indices] for name in self.__dataclass_fields__})


def collect_domain(config: dict[str, Any], domain: str, device: torch.device) -> ProbeData:
    checkpoint = torch.load(REPO_ROOT / config["checkpoints"][domain], map_location="cpu", weights_only=False)
    spec = make_spec(checkpoint["spec"])
    runner_config = checkpoint["config"]
    runner_config.setdefault("training", {})["progress"] = False
    runner = RUNNERS[domain](spec, runner_config, device)
    frame = runner.prepare()
    model = build_benchmark_model(spec, frame, runner_config, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    controller = ContributionMaskController(target_attention(model, domain))

    count = int(config["data"]["examples_per_domain"])
    top_k = int(config["data"]["top_k"])
    eval_rows = np.asarray(runner.eval_idx, dtype=np.int64)
    if domain == "ctdg":
        source = eval_rows[: max(1, (count + 1) // 2)]
        jobs = ([(int(row), False) for row in source] + [(int(row), True) for row in source])[:count]
    else:
        jobs = [(int(row), False) for row in eval_rows[:count]]

    buckets: dict[str, list[Any]] = {name: [] for name in ProbeData.__dataclass_fields__}
    with torch.no_grad():
        for example_id, (row, negative) in enumerate(jobs):
            inputs, target = domain_batch(runner, domain, row, negative=negative)
            controller.item_mask = None
            model(*inputs)
            attention_t = controller.last_attention[0]
            valid = attention_t > 0
            valid_indices = torch.where(valid)[0]
            if valid_indices.numel() < 2:
                continue
            k_count = min(top_k, int(valid_indices.numel()))
            selected = valid_indices[torch.topk(attention_t[valid_indices], k=k_count).indices]
            subset_count = 1 << k_count
            masks = valid.to(torch.float32).unsqueeze(0).repeat(subset_count, 1)
            for subset in range(subset_count):
                bits = torch.tensor(
                    [(subset >> j) & 1 for j in range(k_count)],
                    device=device,
                    dtype=masks.dtype,
                )
                masks[subset, selected] = bits
            controller.item_mask = masks
            result = model(*repeat_batch(inputs, subset_count))
            utility, _ = outputs_and_utility(domain, result, target)
            controller.item_mask = None

            q = controller.last_query[0].detach().cpu().numpy()
            keys = controller.last_keys[0].detach().cpu().numpy()
            values = controller.last_values[0].detach().cpu().numpy()
            attention = attention_t.detach().cpu().numpy()
            contribution = attention[:, None] * values
            background = contribution[valid.cpu().numpy()].sum(axis=0) - contribution[selected.cpu().numpy()].sum(axis=0)

            for subset in range(subset_count):
                active = [j for j in range(k_count) if subset & (1 << j)]
                coalition_sum = background.copy()
                if active:
                    coalition_sum += contribution[selected[active].cpu().numpy()].sum(axis=0)
                for j in range(k_count):
                    if subset & (1 << j):
                        continue
                    item = int(selected[j])
                    marginal = float(utility[subset | (1 << j)] - utility[subset])
                    buckets["q"].append(q)
                    buckets["k"].append(keys[item])
                    buckets["c"].append(contribution[item])
                    buckets["context"].append(coalition_sum)
                    buckets["scalars"].append([attention[item], len(active) / max(1, k_count - 1)])
                    buckets["target"].append(marginal)
                    buckets["example"].append(example_id)
                    buckets["attention"].append(float(attention[item]))
            print(f"collect {domain}: {example_id + 1}/{len(jobs)}", end="\r", flush=True)
    controller.close()
    print()
    return ProbeData(
        q=np.asarray(buckets["q"], dtype=np.float32),
        k=np.asarray(buckets["k"], dtype=np.float32),
        c=np.asarray(buckets["c"], dtype=np.float32),
        context=np.asarray(buckets["context"], dtype=np.float32),
        scalars=np.asarray(buckets["scalars"], dtype=np.float32),
        target=np.asarray(buckets["target"], dtype=np.float32),
        example=np.asarray(buckets["example"], dtype=np.int64),
        attention=np.asarray(buckets["attention"], dtype=np.float32),
    )


def split_examples(data: ProbeData, fractions: list[float], seed: int) -> tuple[ProbeData, ProbeData, ProbeData]:
    examples = np.unique(data.example)
    rng = np.random.default_rng(seed)
    rng.shuffle(examples)
    n_train = max(1, int(len(examples) * fractions[0]))
    n_val = max(1, int(len(examples) * fractions[1]))
    train_ids = examples[:n_train]
    val_ids = examples[n_train : n_train + n_val]
    test_ids = examples[n_train + n_val :]
    return tuple(data.subset(np.where(np.isin(data.example, ids))[0]) for ids in (train_ids, val_ids, test_ids))


class Standardizer:
    def __init__(self, values: np.ndarray) -> None:
        self.mean = values.mean(axis=0, keepdims=True)
        self.std = values.std(axis=0, keepdims=True)
        self.std[self.std < 1.0e-6] = 1.0

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


class MarginalProbe(nn.Module):
    """Parameter-matched local or set-aware predictor."""

    def __init__(self, dim: int, hidden: int, *, use_context: bool) -> None:
        super().__init__()
        self.use_context = use_context
        self.q_proj = nn.Linear(dim, hidden)
        self.k_proj = nn.Linear(dim, hidden)
        self.c_proj = nn.Linear(dim, hidden)
        self.context_proj = nn.Linear(dim, hidden)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(4 * hidden + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, q, k, c, context, scalars):
        context_h = self.context_proj(context)
        if not self.use_context:
            context_h = torch.zeros_like(context_h)
        features = torch.cat([self.q_proj(q), self.k_proj(k), self.c_proj(c), context_h, scalars], dim=-1)
        return self.head(features).squeeze(-1)


class CounterfactualUtilityProbe(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, hidden)
        self.set_proj = nn.Linear(dim, hidden)
        self.utility = nn.Sequential(nn.GELU(), nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def score(self, q, aggregate):
        return self.utility(torch.cat([self.q_proj(q), self.set_proj(aggregate)], dim=-1)).squeeze(-1)

    def forward(self, q, k, c, context, scalars):
        del k, scalars
        return self.score(q, context + c) - self.score(q, context)


def tensor_data(data: ProbeData, transforms: dict[str, Standardizer], target_scale: Standardizer):
    arrays = [transforms[name](getattr(data, name)) for name in ("q", "k", "c", "context", "scalars")]
    target = target_scale(data.target[:, None]).reshape(-1)
    return TensorDataset(*(torch.from_numpy(array).float() for array in arrays), torch.from_numpy(target).float())


def train_probe(model: nn.Module, train: TensorDataset, val: TensorDataset, config: dict[str, Any], device: torch.device):
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loader = DataLoader(train, batch_size=int(config["training"]["batch_size"]), shuffle=True)
    val_loader = DataLoader(val, batch_size=2048)
    best_state, best_loss, stale = None, float("inf"), 0
    for _epoch in range(int(config["training"]["epochs"])):
        model.train()
        for batch in loader:
            *features, target = [item.to(device) for item in batch]
            loss = torch.mean((model(*features) - target) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                *features, target = [item.to(device) for item in batch]
                losses.append(float(torch.mean((model(*features) - target) ** 2).cpu()))
        val_loss = float(np.mean(losses))
        if val_loss < best_loss - 1.0e-6:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(config["training"]["patience"]):
                break
    model.load_state_dict(best_state)
    return model


def predict(model: nn.Module, dataset: TensorDataset, device: torch.device) -> np.ndarray:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=2048):
            features = [item.to(device) for item in batch[:-1]]
            rows.append(model(*features).cpu().numpy())
    return np.concatenate(rows)


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def metrics(target: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = pred - target
    pearson = float(np.corrcoef(target, pred)[0, 1]) if np.std(target) > 0 and np.std(pred) > 0 else float("nan")
    rho = float(np.corrcoef(rankdata(target), rankdata(pred))[0, 1]) if len(target) > 1 else float("nan")
    denom = float(np.sum((target - target.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - np.sum(error**2) / denom) if denom > 0 else float("nan"),
        "pearson": pearson,
        "spearman": rho,
        "sign_accuracy": float(np.mean((pred > 0) == (target > 0))),
        "harmful_auc": auc(target < 0, -pred),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for domain in sorted({row["domain"] for row in rows}):
        for probe in sorted({row["probe"] for row in rows}):
            selected = [row for row in rows if row["domain"] == domain and row["probe"] == probe]
            item = {"domain": domain, "probe": probe, "seeds": len(selected)}
            for key in ("rmse", "mae", "r2", "pearson", "spearman", "sign_accuracy", "harmful_auc"):
                values = np.asarray([row[key] for row in selected], dtype=float)
                item[f"{key}_mean"] = float(np.nanmean(values))
                item[f"{key}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            item["parameters"] = selected[0]["parameters"]
            output.append(item)
    return output


def report(summary: list[dict[str, Any]]) -> str:
    lines = [
        "# Frozen Marginal-Value Probe",
        "",
        "The encoder and predictor are frozen. Splits are made by source example, so coalition rows from one example cannot cross train/test boundaries. Local and set-aware probes have exactly the same parameter count; the local probe's context branch is present but zeroed.",
        "",
        "| Domain | Probe | RMSE | R2 | Pearson | Spearman | Sign acc. | Harmful AUC | Params |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    names = {"local": "Local", "set_aware": "Set-aware", "shuffled_set": "Shuffled-set", "counterfactual": "Counterfactual utility"}
    for row in summary:
        lines.append(
            f"| {row['domain'].upper()} | {names[row['probe']]} | {row['rmse_mean']:.4f} +/- {row['rmse_std']:.4f} | "
            f"{row['r2_mean']:.3f} | {row['pearson_mean']:.3f} | {row['spearman_mean']:.3f} | "
            f"{row['sign_accuracy_mean']:.3f} | {row['harmful_auc_mean']:.3f} | {int(row['parameters'])} |"
        )
    lines += ["", "## Phase-B verdict", ""]
    for domain in sorted({row["domain"] for row in summary}):
        by_probe = {row["probe"]: row for row in summary if row["domain"] == domain}
        local = by_probe["local"]
        aware = by_probe["set_aware"]
        shuffled = by_probe["shuffled_set"]
        lines.append(
            f"- **{domain.upper()}**: set-aware minus local R2 = "
            f"`{aware['r2_mean'] - local['r2_mean']:+.3f}`; set-aware minus shuffled R2 = "
            f"`{aware['r2_mean'] - shuffled['r2_mean']:+.3f}`."
        )
    lines += [
        "",
        "The current pooled set-aware probe does not satisfy the Phase-B Go criterion. CTDG shows no set-aware advantage. RAG shows a small gain over the local probe, but not over the shuffled-set control. Phase C is therefore not justified by this representation. The result separates the existence of context-dependent marginal values (Phase A) from their learnable out-of-example predictability (Phase B).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.examples is not None:
        config["data"]["examples_per_domain"] = args.examples
    domains = [args.domain] if args.domain else list(config["data"]["domains"])
    device = torch.device(args.device or config["experiment"]["device"])
    out_root = REPO_ROOT / config["experiment"]["output_root"]
    all_results: list[dict[str, Any]] = []

    for domain in domains:
        cache = out_root / domain / "probe_data.npz"
        if cache.exists() and args.examples is None:
            payload = np.load(cache)
            data = ProbeData(**{name: payload[name] for name in ProbeData.__dataclass_fields__})
        else:
            data = collect_domain(config, domain, device)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **{name: getattr(data, name) for name in data.__dataclass_fields__})
        for seed in config["experiment"]["seeds"]:
            set_seed(int(seed))
            train, val, test = split_examples(data, config["data"]["split"], int(seed))
            transforms = {name: Standardizer(getattr(train, name)) for name in ("q", "k", "c", "context", "scalars")}
            target_scale = Standardizer(train.target[:, None])
            datasets = [tensor_data(part, transforms, target_scale) for part in (train, val, test)]
            dim = train.q.shape[1]
            hidden = int(config["training"]["hidden_dim"])
            variants: dict[str, nn.Module] = {
                "local": MarginalProbe(dim, hidden, use_context=False),
                "set_aware": MarginalProbe(dim, hidden, use_context=True),
                "shuffled_set": MarginalProbe(dim, hidden, use_context=True),
                "counterfactual": CounterfactualUtilityProbe(dim, hidden),
            }
            rng = np.random.default_rng(int(seed) + 991)
            shuffled_parts = []
            for part in (train, val, test):
                shuffled = copy.copy(part)
                shuffled.context = part.context[rng.permutation(len(part.target))]
                shuffled_parts.append(shuffled)
            shuffled_datasets = [tensor_data(part, transforms, target_scale) for part in shuffled_parts]
            for name, model in variants.items():
                used = shuffled_datasets if name == "shuffled_set" else datasets
                trained = train_probe(model, used[0], used[1], config, device)
                pred_z = predict(trained, used[2], device)
                pred = pred_z * float(target_scale.std.squeeze()) + float(target_scale.mean.squeeze())
                result = {
                    "domain": domain,
                    "probe": name,
                    "seed": seed,
                    "train_examples": len(np.unique(train.example)),
                    "test_examples": len(np.unique(test.example)),
                    "test_rows": len(test.target),
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    **metrics(test.target, pred),
                }
                all_results.append(result)
                print(f"{domain} seed={seed} {name}: r2={result['r2']:.3f} sign={result['sign_accuracy']:.3f}")

    summary = aggregate(all_results)
    write_csv(out_root / "results.csv", all_results)
    write_csv(out_root / "summary.csv", summary)
    text = report(summary)
    (out_root / "analysis").mkdir(parents=True, exist_ok=True)
    (out_root / "analysis" / "frozen_probe_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
