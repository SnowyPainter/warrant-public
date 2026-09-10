from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from evaluation.evaluate import binary_auc
from experiments.edge_query_adapter_ablation import run as edge_ablation
from experiments.neural_dissection.run import CTDGRunner, batches, finite_mean, gate_stats
from experiments.path_localization.common import REPO_ROOT, output_dir, path_metadata, rounded


EDGE_VARIANT_BY_PATH_VARIANT = {
    "base": "base",
    "generic_qk_warrant": "generic_qk_warrant",
    "correct_path_warrant": "edge_conditioned_warrant",
    "shuffled_pairing": "shuffled_edge_query",
}


class OpenPathEdgeAdapter(nn.Module):
    """Edge-query adapter with the metric-facing path open and g=1.

    The edge-level query construction, semantic attention, context projection,
    and residual injection are inherited from the EdgeConditionedWarrantBlock.
    Only the learned permission scorer is bypassed, so this is an
    OpenPath-NoGate control rather than a base model.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def _side_context(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        delta_t_enc: torch.Tensor,
        stats: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.inner.token_encoder(torch.cat([memory, stats], dim=-1))
        tokens = tokens * valid_mask.unsqueeze(-1).to(tokens.dtype)
        weights = self.inner._semantic_attention(query, tokens, delta_t_enc, valid_mask)
        gate = valid_mask.to(tokens.dtype)
        context = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        return context, gate, weights

    def forward(
        self,
        *,
        src_h: torch.Tensor,
        dst_h: torch.Tensor,
        time_h: torch.Tensor,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        src_delta_t_enc: torch.Tensor,
        dst_delta_t_enc: torch.Tensor,
        src_stats: torch.Tensor,
        dst_stats: torch.Tensor,
        src_mask: torch.Tensor,
        dst_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        query = self.inner.query_encoder(torch.cat([src_h, dst_h, time_h], dim=-1))
        src_context, src_gate, src_weights = self._side_context(
            query,
            src_memory,
            src_delta_t_enc,
            src_stats,
            src_mask,
        )
        dst_context, dst_gate, dst_weights = self._side_context(
            query,
            dst_memory,
            dst_delta_t_enc,
            dst_stats,
            dst_mask,
        )
        delta = self.inner.context_proj(torch.cat([src_context, dst_context], dim=-1))
        src_delta, dst_delta = delta.chunk(2, dim=-1)

        valid_count = src_mask.float().sum() + dst_mask.float().sum()
        validity_mean = (src_gate.sum() + dst_gate.sum()) / valid_count.clamp_min(1.0)
        entropy = 0.5 * (
            -(src_weights.clamp_min(1.0e-8) * src_weights.clamp_min(1.0e-8).log()).sum(dim=-1).mean()
            - (dst_weights.clamp_min(1.0e-8) * dst_weights.clamp_min(1.0e-8).log()).sum(dim=-1).mean()
        )
        self.inner.last_validity_mean = validity_mean.detach()
        self.inner.last_attention_entropy = entropy.detach()
        info = {
            "edge_warrant_gate_mean": validity_mean.detach(),
            "edge_warrant_attention_entropy": entropy.detach(),
        }
        return self.inner.context_scale * src_delta, self.inner.context_scale * dst_delta, info


class DecomposedEdgeAdapter(OpenPathEdgeAdapter):
    """Architecture-matched edge path with one isolated weighting freedom."""

    def __init__(self, inner: nn.Module, mode: str) -> None:
        super().__init__(inner)
        self.mode = mode
        hidden_dim = int(inner.hidden_dim)
        if mode == "scalar_gate":
            self.scalar_scorer = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.scalar_scorer.weight)
            nn.init.constant_(self.scalar_scorer.bias, 2.944439)
        elif mode == "item_only_gate":
            self.item_scorer = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.item_scorer.weight)
            nn.init.constant_(self.item_scorer.bias, 2.944439)

    def _side_context(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        delta_t_enc: torch.Tensor,
        stats: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.inner.token_encoder(torch.cat([memory, stats], dim=-1))
        tokens = tokens * valid_mask.unsqueeze(-1).to(tokens.dtype)
        weights = self.inner._semantic_attention(query, tokens, delta_t_enc, valid_mask)
        valid = valid_mask.to(tokens.dtype)

        if self.mode == "scalar_gate":
            scalar = torch.sigmoid(self.scalar_scorer(query))
            gate = scalar * valid
            effective = weights * gate
        elif self.mode == "item_only_gate":
            gate = torch.sigmoid(self.item_scorer(tokens)).squeeze(-1) * valid
            effective = weights * gate
        else:
            gate, _ = self.inner.validity(query, tokens, delta_t_enc, stats, valid_mask)
            if self.mode == "normalized_gate":
                effective = weights * gate
                effective = effective / effective.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
            elif self.mode == "attention_adapter":
                # Same q-item scorer, but it changes the normalized routing
                # distribution via additive logits rather than sigmoid
                # permission or independent total-mass scaling.
                _, logits = self.inner.validity(query, tokens, delta_t_enc, stats, valid_mask)
                logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
                effective = torch.softmax(weights.clamp_min(1.0e-8).log() + logits, dim=-1) * valid
                effective = effective / effective.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
            else:
                raise ValueError(f"Unsupported decomposition mode {self.mode!r}")

        context = (tokens * effective.unsqueeze(-1)).sum(dim=1)
        return context, gate, effective


def run_edge_ablation_source(spec, device: torch.device, out_dir: Path, *, force: bool = False) -> dict[str, Any]:
    """Use the CTDG edge-query ablation as the source of truth.

    Path localization asks the same question as the CTDG edge-query adapter
    ablation for this domain. Reusing that row avoids accidental config drift.
    """

    edge_variant = EDGE_VARIANT_BY_PATH_VARIANT[spec.variant]
    edge_config = edge_ablation.load_config(edge_ablation.DEFAULT_CONFIG)
    edge_output_root = REPO_ROOT / str(
        edge_config.get("experiment", {}).get("output_root", "experiments/edge_query_adapter_ablation/outputs")
    )
    edge_out_dir = edge_ablation.output_dir(edge_output_root, edge_variant, spec.seed)
    metrics_path = edge_out_dir / "final_metrics.json"
    if force or not metrics_path.exists():
        edge_row = edge_ablation.run_variant(edge_config, edge_variant, spec.seed, device, edge_out_dir)
    else:
        edge_row = json.loads(metrics_path.read_text(encoding="utf-8"))

    row = {
        "domain": "ctdg",
        "dataset": edge_row.get("dataset", spec.dataset),
        "model": edge_row.get("model", spec.model),
        "variant": spec.variant,
        "seed": spec.seed,
        "status": edge_row.get("status", "completed"),
        "train_loss": edge_row.get("train_loss"),
        "eval_loss": edge_row.get("eval_loss"),
        "primary_metric": edge_row.get("primary_metric"),
        "accuracy": edge_row.get("accuracy"),
        "auc": edge_row.get("auc"),
        "gate_mean": edge_row.get("warrant_gate_mean"),
        "warrant_logit_mean": edge_row.get("warrant_logit_mean"),
        "edge_warrant_gate_mean": edge_row.get("edge_warrant_gate_mean"),
        "edge_warrant_attention_entropy": edge_row.get("edge_warrant_attention_entropy"),
        "warrant_active_blocks": edge_row.get("warrant_active_blocks"),
        "warrant_replacements": edge_row.get("warrant_replacements"),
        "elapsed_sec": edge_row.get("elapsed_sec"),
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
        **path_metadata("ctdg", spec.variant),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    source_metrics = edge_out_dir / "metrics_by_epoch.csv"
    if source_metrics.exists():
        (out_dir / "metrics_by_epoch.csv").write_text(source_metrics.read_text(encoding="utf-8"), encoding="utf-8")
    row = rounded(row)
    with (out_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, sort_keys=True)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": "experiments/edge_query_adapter_ablation",
                "source_variant": edge_variant,
                "source_output_dir": str(edge_out_dir.relative_to(REPO_ROOT)),
                "path_variant": spec.variant,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return row


class PathCTDGRunner(CTDGRunner):
    """CTDG link-prediction path-localization runner.

    The shuffled control keeps source/destination histories fixed but evaluates
    the edge Warrant adapter with a mismatched edge-level query.
    """

    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        if self.spec.variant != "shuffled_pairing":
            return super().evaluate(model)

        model.eval()
        labels_all, scores_all, losses, aux_rows = [], [], [], []
        with torch.no_grad():
            for rows in batches(
                self.eval_idx,
                self.batch_size,
                shuffle=False,
                desc=f"{self.spec.model} {self.spec.variant} eval",
                config=self.config,
            ):
                pos = self.make_batch(rows, negative=False)
                neg = self.make_batch(rows, negative=True)
                combined = tuple(torch.cat([pos_item, neg_item], dim=0) for pos_item, neg_item in zip(pos, neg))
                result = model(*combined)
                labels = torch.cat(
                    [
                        torch.ones(len(rows), device=self.device),
                        torch.zeros(len(rows), device=self.device),
                    ]
                )
                logits = result.logits
                losses.append(float(F.binary_cross_entropy_with_logits(logits, labels).cpu()))
                labels_all.extend(labels.cpu().tolist())
                scores_all.extend(torch.sigmoid(logits).cpu().tolist())
                aux_rows.append(result.aux)
        preds = [1.0 if score >= 0.5 else 0.0 for score in scores_all]
        acc = float(np.mean([pred == label for pred, label in zip(preds, labels_all)])) if labels_all else float("nan")
        auc = binary_auc(labels_all, scores_all)
        return {"eval_loss": finite_mean(losses), "primary_metric": auc, "auc": auc, "accuracy": acc, **gate_stats(aux_rows)}


def configure_model(model: torch.nn.Module, variant: str) -> None:
    if variant in {"open_path_no_gate", "generic_open_path"}:
        if getattr(model, "edge_warrant", None) is None:
            raise ValueError("CTDG open_path_no_gate requires an edge Warrant adapter")
        model.edge_warrant = OpenPathEdgeAdapter(model.edge_warrant)
        return
    if variant == "shuffled_pairing":
        if getattr(model, "edge_warrant", None) is None:
            raise ValueError("CTDG shuffled_pairing requires an edge Warrant adapter")
        model.edge_warrant = edge_ablation.ShuffledEdgeQueryAdapter(model.edge_warrant)
        return
    if variant in {"scalar_gate", "item_only_gate", "normalized_gate", "attention_adapter"}:
        if getattr(model, "edge_warrant", None) is None:
            raise ValueError(f"CTDG {variant} requires an edge Warrant adapter")
        model.edge_warrant = DecomposedEdgeAdapter(model.edge_warrant, variant)
        return
    if variant == "combined_full":
        if getattr(model, "edge_warrant", None) is None:
            raise ValueError("CTDG combined_full requires an edge Warrant adapter")
        return
