from __future__ import annotations

import types

import torch

from experiments.neural_dissection.run import RAGRunner
from experiments.path_localization.common import set_attention_warrant


def _nontrivial_permutation(batch_size: int, device: torch.device) -> torch.Tensor:
    order = torch.arange(batch_size, device=device)
    if batch_size <= 1:
        return order
    return torch.roll(order, shifts=max(1, batch_size // 2))


class PathRAGRunner(RAGRunner):
    """RAG support-selection path-localization runner.

    The shuffled control keeps question, passages, and support labels fixed but
    rolls the permission mass across passage slots. This breaks the
    question-passage permission pairing without changing the retrieval input.
    """

    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        if self.spec.variant != "shuffled_pairing":
            return super().evaluate(model)

        original = model._support_logits
        model._support_logits = types.MethodType(shuffled_support_logits, model)
        try:
            return super().evaluate(model)
        finally:
            model._support_logits = original


def shuffled_support_logits(self, query, memory, context, info, valid_mask):
    mass = info["attention"] * info.get("warrant_gate", torch.ones_like(info["attention"]))
    if mass.shape[1] > 1:
        mass = torch.roll(mass, shifts=max(1, mass.shape[1] // 2), dims=1)
    query_p = query.unsqueeze(1).expand_as(memory)
    context_p = context.unsqueeze(1).expand_as(memory)
    features = torch.cat([query_p, memory, query_p * memory, context_p, mass.unsqueeze(-1)], dim=-1)
    logits = self.passage_scorer(features).squeeze(-1)
    if self.passage_warrant_scale is not None:
        passage_count = valid_mask.to(mass.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
        warrant_bias = 0.5 * torch.sigmoid(self.passage_warrant_scale) * torch.log1p(
            mass.clamp_min(0.0) * passage_count
        )
        logits = logits + warrant_bias
    return logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)


def configure_model(model: torch.nn.Module, variant: str) -> None:
    if variant in {"correct_path_warrant", "shuffled_pairing"}:
        set_attention_warrant(model, True)
        return
    if variant == "generic_qk_warrant":
        original = model._support_logits

        def attention_only_support_logits(self, query, memory, context, info, valid_mask):
            patched = dict(info)
            patched["warrant_gate"] = torch.ones_like(info["attention"])
            return original(query, memory, context, patched, valid_mask)

        model._support_logits = types.MethodType(attention_only_support_logits, model)
