from __future__ import annotations

import types

import numpy as np
import torch

from experiments.neural_dissection.run import RAGRunner, rag_contexts_to_passages, rag_support_labels, tokenize_text
from experiments.path_localization.common import decompose_warrant_block, set_attention_warrant


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

    def prepare(self):
        frame = super().prepare()
        count = len(self.examples)
        self.question_cache = np.zeros((count, self.question_len), dtype=np.int64)
        self.passage_cache = np.zeros((count, self.max_passages, self.passage_len), dtype=np.int64)
        self.mask_cache = np.zeros((count, self.max_passages), dtype=bool)
        self.support_cache = np.zeros((count, self.max_passages), dtype=np.float32)
        for idx, example in enumerate(self.examples):
            self.question_cache[idx] = tokenize_text(example.get("question", ""), self.vocab_size, self.question_len)
            passages = rag_contexts_to_passages(example.get("contexts"), max_passages=self.max_passages)
            for passage_idx, passage in enumerate(passages[: self.max_passages]):
                tokens = tokenize_text(passage["text"], self.vocab_size, self.passage_len)
                self.passage_cache[idx, passage_idx] = tokens
                self.mask_cache[idx, passage_idx] = bool(np.any(tokens))
            self.support_cache[idx] = rag_support_labels(example, passages, max_passages=self.max_passages)
        return frame

    def make_batch(self, rows: np.ndarray):
        return (
            torch.as_tensor(self.question_cache[rows], device=self.device),
            torch.as_tensor(self.passage_cache[rows], device=self.device),
            torch.as_tensor(self.mask_cache[rows], dtype=torch.bool, device=self.device),
            torch.as_tensor(self.support_cache[rows], dtype=torch.float32, device=self.device),
        )

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
    if variant == "open_path_no_gate":
        # Keep the metric-facing passage-mass path created by
        # ``passage_warrant_scale``, but use ordinary attention mass (g = 1).
        # No WarrantBlock is expected to remain active in this control.
        set_attention_warrant(model, False)
        model.warrant_expected = False
        return
    if variant in {"correct_path_warrant", "shuffled_pairing", "combined_full"}:
        set_attention_warrant(model, True)
        return
    if variant in {"scalar_gate", "item_only_gate", "normalized_gate", "attention_adapter"}:
        set_attention_warrant(model, True)
        decompose_warrant_block(model.attn.warrant, variant)
        return
    if variant == "generic_open_path":
        set_attention_warrant(model, True)
        # Keep the metric-facing mass path ungated while generic attention
        # remains warranted.
        original = model._support_logits
        def open_metric_logits(self, query, memory, context, info, valid_mask):
            patched = dict(info)
            patched["warrant_gate"] = torch.ones_like(info["attention"])
            return original(query, memory, context, patched, valid_mask)
        model._support_logits = types.MethodType(open_metric_logits, model)
        return
    if variant == "generic_qk_warrant":
        original = model._support_logits

        def attention_only_support_logits(self, query, memory, context, info, valid_mask):
            patched = dict(info)
            patched["warrant_gate"] = torch.ones_like(info["attention"])
            return original(query, memory, context, patched, valid_mask)

        model._support_logits = types.MethodType(attention_only_support_logits, model)
