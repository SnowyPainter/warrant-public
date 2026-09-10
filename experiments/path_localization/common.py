from __future__ import annotations

import csv
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import RunSpec  # noqa: E402
from models.attention import WarrantedAttention  # noqa: E402
from models.warrant_block import WarrantBlock  # noqa: E402


RESULT_COLUMNS = [
    "domain",
    "dataset",
    "model",
    "variant",
    "path_role",
    "seed",
    "status",
    "train_loss",
    "eval_loss",
    "primary_metric",
    "metric_name",
    "direction",
    "accuracy",
    "auc",
    "mark_mrr",
    "mrr",
    "support_mrr",
    "support_ap",
    "support_auc",
    "support_attention_mass",
    "distractor_attention_mass",
    "support_attention_ratio",
    "support_warrant_mass",
    "distractor_warrant_mass",
    "support_mass_ratio",
    "joint_nll",
    "mae_time",
    "rmse_location",
    "gate_mean",
    "warrant_logit_mean",
    "edge_warrant_gate_mean",
    "edge_warrant_attention_entropy",
    "edge_warrant_gate_std",
    "edge_warrant_gate_high_fraction",
    "edge_warrant_gate_low_fraction",
    "edge_gate_grad_norm",
    "edge_attention_grad_norm",
    "permission_grad_norm",
    "attention_parameter_grad_norm",
    "inference_native_primary",
    "inference_g1_primary",
    "inference_g1_delta",
    "warrant_tail_mass_mean",
    "warrant_active_blocks",
    "warrant_replacements",
    "metric_path",
    "key_value_item",
    "failure_bottleneck",
    "warrant_path",
    "elapsed_sec",
    "output_dir",
]

PATH_CLAIMS = {
    "ctdg": {
        "metric_name": "auc",
        "direction": "higher",
        "metric_path": "source/destination temporal history -> edge score",
        "key_value_item": "temporal neighbor/history item",
        "failure_bottleneck": "history terms can remain stale or hard-negative aligned while still entering the edge score",
        "warrant_path": "edge-level source/destination history weighted term",
    },
    "mtpp": {
        "metric_name": "mark_mrr",
        "direction": "higher",
        "metric_path": "past marked events -> candidate mark logits",
        "key_value_item": "past marked event",
        "failure_bottleneck": "the same event history can support one candidate mark while acting as a shortcut for another",
        "warrant_path": "candidate-wise event-history weighted term",
    },
    "rag": {
        "metric_name": "support_mrr",
        "direction": "higher",
        "metric_path": "retrieved passages -> support passage ranking score",
        "key_value_item": "retrieved passage",
        "failure_bottleneck": "distractor passages can be relevant to the question while not being supporting evidence",
        "warrant_path": "question-conditioned passage weighted term",
    },
    "stpp": {
        "metric_name": "rmse_location",
        "direction": "lower",
        "metric_path": "past spatio-temporal events -> next-location dynamics",
        "key_value_item": "past spatio-temporal event",
        "failure_bottleneck": "nearby or recent events can shift continuous location dynamics in the wrong direction",
        "warrant_path": "prediction-state event-history weighted term before the location head",
    },
    "tkg": {
        "metric_name": "mrr",
        "direction": "higher",
        "metric_path": "historical facts -> tail entity ranking score",
        "key_value_item": "historical temporal fact",
        "failure_bottleneck": "repeated or stale facts can dominate copy/tail scores for the wrong query",
        "warrant_path": "query-conditioned historical tail contribution path",
    },
}


VARIANT_ROLES = {
    "base": "no_warrant",
    "generic_qk_warrant": "weak_path_generic_attention",
    "open_path_no_gate": "metric_path_open_g_equals_one",
    "correct_path_warrant": "metric_defining_path",
    "shuffled_pairing": "query_item_pairing_control",
    "scalar_gate": "metric_path_query_scalar_gate",
    "item_only_gate": "metric_path_item_only_gate",
    "normalized_gate": "metric_path_normalized_query_item_gate",
    "attention_adapter": "metric_path_attention_reweighting_control",
    "generic_open_path": "generic_gate_plus_metric_path_g_equals_one",
    "combined_full": "generic_gate_plus_metric_path_query_item_gate",
    "frozen_gate": "frozen_backbone_metric_path_query_item_gate",
    "frozen_attention_adapter": "frozen_backbone_metric_path_attention_reweighting",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_device(config: dict[str, Any]) -> torch.device:
    value = str(config.get("experiment", {}).get("device", "auto"))
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def as_selection(value: Any, default: Iterable[Any]) -> list[str]:
    if value is None or value == "all":
        return [str(item) for item in default]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def selected_seeds(config: dict[str, Any], value: str | None) -> list[int]:
    default = config.get("experiment", {}).get("seeds", [7])
    return [int(seed) for seed in as_selection(value or "all", default)]


def selected_variants(config: dict[str, Any], value: str | None) -> list[str]:
    variants = list(config.get("variants", {}).keys())
    wanted = as_selection(value or "all", variants)
    unknown = sorted(set(wanted) - set(variants))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    return wanted


def selected_domains(config: dict[str, Any], value: str | None) -> list[str]:
    domains = list(config.get("tasks", {}).keys())
    wanted = as_selection(value or "all", domains)
    unknown = sorted(set(wanted) - set(domains))
    if unknown:
        raise ValueError(f"Unknown domains: {unknown}")
    return wanted


def output_dir(output_root: Path, spec: RunSpec) -> Path:
    return output_root / spec.domain / spec.dataset / spec.model / spec.variant / f"seed_{spec.seed}"


def should_skip(out_dir: Path, force: bool) -> bool:
    if force:
        return False
    metrics_path = out_dir / "final_metrics.json"
    if not metrics_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return metrics.get("status") == "completed"
    except Exception:
        return False


def rounded(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                continue
            out[key] = round(value, 6)
        elif isinstance(value, (np.floating, np.integer)):
            value = float(value)
            if math.isnan(value) or math.isinf(value):
                continue
            out[key] = round(value, 6)
        else:
            out[key] = value
    return out


def append_result(results_path: Path, row: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    key = (str(row["domain"]), str(row["dataset"]), str(row["model"]), str(row["variant"]), int(row["seed"]))
    if results_path.exists():
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            for existing in csv.DictReader(handle):
                existing_key = (
                    existing.get("domain", ""),
                    existing.get("dataset", ""),
                    existing.get("model", ""),
                    existing.get("variant", ""),
                    int(existing.get("seed", -1)),
                )
                if existing_key != key:
                    rows.append(existing)
    rows.append({column: row.get(column, "") for column in RESULT_COLUMNS})
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_spec(config: dict[str, Any], domain: str, variant: str, seed: int) -> RunSpec:
    task = config["tasks"][domain]
    return RunSpec(
        domain=domain,
        dataset=str(task["dataset"]),
        dataset_path=REPO_ROOT / str(task["path"]),
        model=str(task["model"]),
        variant=variant,
        seed=seed,
        implementation=str(config.get("training", {}).get("implementation", "reference")),
        model_config=dict(task.get("model_config", {})),
        use_warrant=variant != "base",
    )


def iter_specs(config: dict[str, Any], *, domains: list[str], variants: list[str], seeds: list[int]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for domain in domains:
        for seed in seeds:
            for variant in variants:
                specs.append(build_spec(config, domain, variant, seed))
    return specs


def set_attention_warrant(model: torch.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, WarrantedAttention):
            module.use_warrant = bool(enabled)


def force_open_warrant_block(block: WarrantBlock) -> None:
    """Keep a WarrantBlock's value path open while fixing permission to g=1.

    This implements the ``OpenPath-NoGate`` control: the same item-wise
    attention-weighted value path is exposed to the metric-facing head, but the
    learned permission scorer is bypassed.  Downstream projections/scales remain
    trainable, so this isolates learned permission from merely opening a path.
    """

    def open_forward(
        self: WarrantBlock,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_logits: torch.Tensor | None = None,
        attention_weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        squeeze_query = False
        if query.ndim == 2:
            query = query.unsqueeze(1)
            squeeze_query = True
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("query, key, and value must be 3D tensors, except query may be 2D")
        if key.shape[:2] != value.shape[:2]:
            raise ValueError("key and value must have the same [batch, length] shape")
        if query.shape[0] != key.shape[0]:
            raise ValueError("query, key, and value must have the same batch size")

        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]
        pair_mask = self._pair_mask(mask, batch_size, query_len, key_len, query.device)
        if attention_weights is None:
            if attention_logits is None:
                attention_logits = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(float(key.shape[-1]))
            attention_logits = attention_logits.masked_fill(~pair_mask, torch.finfo(attention_logits.dtype).min)
            empty_rows = ~pair_mask.any(dim=-1, keepdim=True)
            attention_logits = torch.where(empty_rows, torch.zeros_like(attention_logits), attention_logits)
            attention_weights = torch.softmax(attention_logits, dim=-1)
        else:
            if attention_weights.shape != (batch_size, query_len, key_len):
                raise ValueError(
                    "attention_weights must have shape "
                    f"{(batch_size, query_len, key_len)}, got {tuple(attention_weights.shape)}"
                )
            attention_weights = attention_weights.to(device=query.device, dtype=value.dtype)
            attention_weights = attention_weights * pair_mask.to(attention_weights.dtype)

        attention_weights = self.dropout(attention_weights)
        gate = pair_mask.to(dtype=value.dtype)
        gate_logits = torch.zeros_like(gate)
        output = torch.sum(attention_weights.unsqueeze(-1) * value.unsqueeze(1), dim=2)
        output = self.output_proj(output)

        if squeeze_query:
            output = output.squeeze(1)
            attention_weights = attention_weights.squeeze(1)
            gate = gate.squeeze(1)
            gate_logits = gate_logits.squeeze(1)
        return output, attention_weights, gate, gate_logits

    block.forward = types.MethodType(open_forward, block)


def decompose_warrant_block(block: WarrantBlock, mode: str) -> None:
    """Give a metric-facing WarrantBlock exactly one weighting freedom.

    The path, output projection, mask, and dropout are shared across controls.
    ``normalized_gate`` preserves total mass, while ``attention_adapter`` uses
    the capacity-matched q-item scorer solely as a normalized routing adapter.
    """
    if mode not in {"scalar_gate", "item_only_gate", "normalized_gate", "attention_adapter"}:
        raise ValueError(f"Unsupported WarrantBlock decomposition mode: {mode}")
    if mode == "scalar_gate":
        block.scalar_scorer = nn.Linear(block.query_dim, 1)
        nn.init.zeros_(block.scalar_scorer.weight)
        nn.init.constant_(block.scalar_scorer.bias, 2.944439)
    elif mode == "item_only_gate":
        block.item_scorer = nn.Linear(block.key_dim, 1)
        nn.init.zeros_(block.item_scorer.weight)
        nn.init.constant_(block.item_scorer.bias, 2.944439)

    def decomposed_forward(
        self: WarrantBlock,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_logits: torch.Tensor | None = None,
        attention_weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        squeeze_query = query.ndim == 2
        if squeeze_query:
            query = query.unsqueeze(1)
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]
        pair_mask = self._pair_mask(mask, batch_size, query_len, key_len, query.device)
        if attention_weights is None:
            if attention_logits is None:
                attention_logits = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(float(key.shape[-1]))
            attention_logits = attention_logits.masked_fill(~pair_mask, torch.finfo(attention_logits.dtype).min)
            attention_logits = torch.where(~pair_mask.any(dim=-1, keepdim=True), torch.zeros_like(attention_logits), attention_logits)
            alpha = torch.softmax(attention_logits, dim=-1)
        else:
            alpha = attention_weights.to(device=query.device, dtype=value.dtype) * pair_mask.to(value.dtype)

        if mode == "scalar_gate":
            logits = self.scalar_scorer(query).expand(-1, -1, key_len)
        elif mode == "item_only_gate":
            logits = self.item_scorer(key).squeeze(-1).unsqueeze(1).expand(-1, query_len, -1)
        else:
            logits = self._gate_logits(query, key)
        gate = self.gate_leak + (1.0 - self.gate_leak) * torch.sigmoid(logits)
        gate = torch.where(pair_mask, gate, torch.zeros_like(gate))
        logits = torch.where(pair_mask, logits, torch.zeros_like(logits))

        if mode == "attention_adapter":
            # Capacity-matched attention-only control: use the same q-item
            # scorer as an additive attention-logit adapter, without sigmoid
            # permission or independent total-mass control.
            adapted_logits = alpha.clamp_min(1.0e-9).log() + logits
            adapted_logits = adapted_logits.masked_fill(~pair_mask, torch.finfo(adapted_logits.dtype).min)
            effective = torch.softmax(adapted_logits, dim=-1) * pair_mask.to(value.dtype)
        else:
            effective = alpha * gate
        if mode == "normalized_gate":
            effective = effective / effective.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        effective = self.dropout(effective)
        output = self.output_proj(torch.sum(effective.unsqueeze(-1) * value.unsqueeze(1), dim=2))

        # Downstream metric paths reconstruct alpha*g from these tensors.  For
        # normalized controls, expose the effective distribution and unit gate
        # so that the downstream path receives exactly the context used here.
        reported_alpha = effective if mode in {"normalized_gate", "attention_adapter"} else alpha
        reported_gate = pair_mask.to(value.dtype) if mode in {"normalized_gate", "attention_adapter"} else gate
        if squeeze_query:
            output = output.squeeze(1)
            reported_alpha = reported_alpha.squeeze(1)
            reported_gate = reported_gate.squeeze(1)
            logits = logits.squeeze(1)
        return output, reported_alpha, reported_gate, logits

    block.forward = types.MethodType(decomposed_forward, block)


def disable_correct_path(model: torch.nn.Module, domain: str) -> None:
    if domain == "ctdg" and hasattr(model, "edge_warrant"):
        model.edge_warrant = None
    elif domain == "mtpp":
        if hasattr(model, "candidate_warrant"):
            model.candidate_warrant = None
        if hasattr(model, "candidate_value_scale"):
            model.candidate_value_scale = None
    elif domain == "stpp":
        if hasattr(model, "prediction_warrant"):
            model.prediction_warrant = None
        if hasattr(model, "prediction_warrant_scale"):
            model.prediction_warrant_scale = None
    elif domain == "tkg":
        if hasattr(model, "warrant_copy_scale"):
            model.warrant_copy_scale = None
    elif domain == "rag":
        if hasattr(model, "passage_warrant_scale"):
            model.passage_warrant_scale = None


def configure_variant(model: torch.nn.Module, spec: RunSpec) -> None:
    if spec.variant == "base":
        set_attention_warrant(model, False)
        disable_correct_path(model, spec.domain)
    elif spec.variant == "generic_qk_warrant":
        set_attention_warrant(model, True)
        disable_correct_path(model, spec.domain)
    elif spec.variant in {
        "open_path_no_gate",
        "correct_path_warrant",
        "shuffled_pairing",
        "scalar_gate",
        "item_only_gate",
        "normalized_gate",
        "attention_adapter",
        "generic_open_path",
        "combined_full",
    }:
        # Start from a clean localization: generic attention Warrant is off,
        # then each domain explicitly enables it only when that attention module
        # is itself part of the metric-defining path.
        set_attention_warrant(model, spec.variant in {"generic_open_path", "combined_full"})
    else:
        raise ValueError(f"Unsupported path-localization variant: {spec.variant}")


def path_metadata(domain: str, variant: str) -> dict[str, Any]:
    claim = PATH_CLAIMS[domain]
    return {
        "metric_name": claim["metric_name"],
        "direction": claim["direction"],
        "path_role": VARIANT_ROLES.get(variant, variant),
        "metric_path": claim["metric_path"],
        "key_value_item": claim["key_value_item"],
        "failure_bottleneck": claim["failure_bottleneck"],
        "warrant_path": claim["warrant_path"],
    }
