#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import normalize_rag_text, read_jsonl  # noqa: E402
from models.bert_warrant import BertHotpotMultiCandidateWarrantSelector  # noqa: E402


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
RESULT_COLUMNS = [
    "dataset",
    "model",
    "variant",
    "seed",
    "status",
    "total_parameters",
    "trainable_parameters",
    "train_loss",
    "eval_loss",
    "support_mrr",
    "clean_support_mrr",
    "hard_support_mrr",
    "recall_at_1",
    "recall_at_2",
    "recall_at_5",
    "clean_recall_at_1",
    "hard_recall_at_1",
    "hard_recall_at_2",
    "evidence_f1",
    "auprc",
    "hard_at_1",
    "hard_at_gold_count",
    "unsupported_at_gold_count",
    "precision_at_gold_count",
    "gold_hard_pair_auc",
    "score_margin_gold_hard",
    "candidate_gate_mean",
    "candidate_gate_std",
    "gold_gate_mean",
    "hard_gate_mean",
    "random_gate_mean",
    "gold_attention_mass",
    "hard_attention_mass",
    "random_attention_mass",
    "gold_effective_mass",
    "hard_effective_mass",
    "random_effective_mass",
    "gold_hard_attention_ratio",
    "gold_hard_effective_ratio",
    "gold_random_attention_ratio",
    "gold_random_effective_ratio",
    "elapsed_sec",
    "output_dir",
]


@dataclass
class Candidate:
    question_id: int
    question: str
    title: str
    sentence: str
    label: int
    is_hard: int


@dataclass
class ListExample:
    question_id: int
    question: str
    candidates: list[Candidate]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HotpotQA Transformer encoder Warrant experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", default=None, help="Variant or comma list. Defaults to config variants.")
    parser.add_argument("--seed", default=None, help="Seed or comma list. Defaults to config seeds.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def selected(raw: str | None, values: list[Any]) -> list[str]:
    if raw is None or raw == "all":
        return [str(v) for v in values]
    return [item.strip() for item in raw.split(",") if item.strip()]


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def support_pairs(example: dict[str, Any]) -> set[tuple[str, int]]:
    facts = example.get("supporting_facts") or example.get("supporting") or {}
    titles = facts.get("title") if isinstance(facts, dict) else []
    sent_ids = facts.get("sent_id") if isinstance(facts, dict) else []
    pairs = set()
    if isinstance(titles, list) and isinstance(sent_ids, list):
        for title, sent_id in zip(titles, sent_ids):
            try:
                pairs.add((normalize_rag_text(title), int(sent_id)))
            except Exception:
                continue
    return pairs


def all_sentences(example: dict[str, Any]) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    contexts = example.get("contexts")
    if not isinstance(contexts, list):
        return rows
    for context in contexts:
        if not isinstance(context, dict):
            continue
        titles = context.get("title")
        sentences = context.get("sentences")
        if isinstance(titles, list) and isinstance(sentences, list):
            for title, passage_sentences in zip(titles, sentences):
                if isinstance(passage_sentences, list):
                    for sent_id, sentence in enumerate(passage_sentences):
                        text = str(sentence).strip()
                        if text:
                            rows.append((str(title), int(sent_id), text))
    return rows


def build_candidates(examples: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[Candidate], dict[int, list[int]]]:
    builder = config["candidate_builder"]
    max_questions = int(builder["max_questions"])
    max_candidates = int(builder["max_candidates_per_question"])
    hard_k = int(builder["hard_distractors"])
    random_k = int(builder["random_distractors"])
    candidates: list[Candidate] = []
    by_question: dict[int, list[int]] = {}
    rng = random.Random(17)
    for qid, example in enumerate(examples[:max_questions]):
        question = str(example.get("question", ""))
        support = support_pairs(example)
        sentence_rows = all_sentences(example)
        if not sentence_rows or not support:
            continue
        positives: list[tuple[str, int, str]] = []
        negatives: list[tuple[float, tuple[str, int, str]]] = []
        q_tokens = token_set(question)
        for title, sent_id, sentence in sentence_rows:
            key = (normalize_rag_text(title), int(sent_id))
            if key in support:
                positives.append((title, sent_id, sentence))
            else:
                s_tokens = token_set(sentence)
                score = len(q_tokens & s_tokens) / max(1, len(q_tokens))
                negatives.append((score, (title, sent_id, sentence)))
        if not positives or not negatives:
            continue
        negatives.sort(key=lambda item: item[0], reverse=True)
        hard = [row for _, row in negatives[:hard_k]]
        rest = [row for _, row in negatives[hard_k:]]
        rng.shuffle(rest)
        selected_rows = [(row, 1, 0) for row in positives[:2]]
        selected_rows.extend((row, 0, 1) for row in hard)
        selected_rows.extend((row, 0, 0) for row in rest[:random_k])
        selected_rows = selected_rows[:max_candidates]
        if not any(label for _, label, _ in selected_rows):
            continue
        rng.shuffle(selected_rows)
        by_question[qid] = []
        for (title, _, sentence), label, is_hard in selected_rows:
            by_question[qid].append(len(candidates))
            candidates.append(Candidate(qid, question, title, sentence, int(label), int(is_hard)))
    return candidates, by_question


def build_list_examples(candidates: list[Candidate], by_question: dict[int, list[int]]) -> list[ListExample]:
    examples = []
    for qid, indices in sorted(by_question.items()):
        group = [candidates[idx] for idx in indices]
        if not group or not any(candidate.label for candidate in group):
            continue
        examples.append(ListExample(qid, group[0].question, group))
    return examples


def split_questions(question_ids: list[int], eval_ratio: float) -> tuple[set[int], set[int]]:
    ids = sorted(question_ids)
    split = max(1, int(len(ids) * (1.0 - eval_ratio)))
    split = min(split, max(1, len(ids) - 1))
    return set(ids[:split]), set(ids[split:])


def batches(indices: list[int], batch_size: int, *, shuffle: bool):
    rows = indices[:]
    if shuffle:
        random.shuffle(rows)
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def encode_batch(tokenizer, candidates: list[Candidate], indices: list[int], *, max_length: int, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    questions = [candidates[idx].question for idx in indices]
    sentences = [candidates[idx].sentence for idx in indices]
    labels = torch.tensor([candidates[idx].label for idx in indices], dtype=torch.float32, device=device)
    encoded = tokenizer(
        questions,
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    candidate_masks = []
    for encoding in encoded.encodings:
        sequence_ids = encoding.sequence_ids
        candidate_masks.append([sid == 1 for sid in sequence_ids])
    model_inputs = {key: value.to(device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
    candidate_mask = torch.tensor(candidate_masks, dtype=torch.bool, device=device)
    return model_inputs, candidate_mask, labels


def encode_list_batch(
    tokenizer,
    examples: list[ListExample],
    indices: list[int],
    *,
    marker_tokens: list[str],
    max_length: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = [examples[idx] for idx in indices]
    texts = []
    labels = torch.zeros((len(rows), len(marker_tokens)), dtype=torch.float32)
    hard = torch.zeros((len(rows), len(marker_tokens)), dtype=torch.bool)
    for row_idx, example in enumerate(rows):
        chunks = [
            f"Question: {example.question}",
            "Instruction: Select only the candidate sentences that are necessary supporting evidence for answering the question. Do not select sentences that are merely related, lexically similar, or about the same entities.",
        ]
        for cand_idx, marker in enumerate(marker_tokens):
            if cand_idx >= len(example.candidates):
                continue
            candidate = example.candidates[cand_idx]
            chunks.append(f"{marker} Candidate evidence: {candidate.sentence}")
            labels[row_idx, cand_idx] = float(candidate.label)
            hard[row_idx, cand_idx] = bool(candidate.is_hard)
        texts.append(" ".join(chunks))
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"]
    marker_ids = tokenizer.convert_tokens_to_ids(marker_tokens)
    special_ids = set(tokenizer.all_special_ids)
    marker_id_set = set(int(marker_id) for marker_id in marker_ids)
    batch, seq_len = input_ids.shape
    marker_mask = torch.zeros((batch, len(marker_tokens), seq_len), dtype=torch.bool)
    token_mask = torch.zeros_like(marker_mask)
    valid = torch.zeros((batch, len(marker_tokens)), dtype=torch.bool)
    for row_idx in range(batch):
        ids = [int(value) for value in input_ids[row_idx].tolist()]
        positions: list[int | None] = []
        for marker_id in marker_ids:
            try:
                positions.append(ids.index(int(marker_id)))
            except ValueError:
                positions.append(None)
        for cand_idx, pos in enumerate(positions):
            if pos is None or cand_idx >= len(rows[row_idx].candidates):
                continue
            marker_mask[row_idx, cand_idx, pos] = True
            next_positions = [p for p in positions[cand_idx + 1 :] if p is not None]
            end = min(next_positions) if next_positions else seq_len
            for token_pos in range(pos + 1, end):
                token_id = ids[token_pos]
                if token_id == tokenizer.pad_token_id or token_id in marker_id_set or token_id in special_ids:
                    continue
                token_mask[row_idx, cand_idx, token_pos] = True
            valid[row_idx, cand_idx] = bool(token_mask[row_idx, cand_idx].any())
    model_inputs = {key: value.to(device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
    return (
        model_inputs,
        marker_mask.to(device),
        token_mask.to(device),
        valid.to(device),
        labels.to(device),
        hard.to(device),
    )


def candidate_gate_scores(result, token_mask: torch.Tensor) -> torch.Tensor | None:
    if not result.aux or "permission" not in result.aux:
        return None
    permission = result.aux["permission"]
    mask = token_mask.to(permission.device, dtype=permission.dtype)
    return (permission * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)


def permission_alignment_loss(
    result,
    labels: torch.Tensor,
    valid: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor | None:
    gate_score = candidate_gate_scores(result, token_mask)
    if gate_score is None:
        return None
    positive = labels.to(torch.bool) & valid
    negative = (~labels.to(torch.bool)) & valid
    losses = []
    for row in range(labels.shape[0]):
        pos = gate_score[row][positive[row]]
        neg = gate_score[row][negative[row]]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        losses.append(F.softplus(neg.unsqueeze(0) - pos.unsqueeze(1) + margin).mean())
    if not losses:
        return None
    return torch.stack(losses).mean()


def gate_grad_norms(model) -> dict[str, float]:
    norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if "gate" not in name or param.grad is None:
            continue
        norms[name] = float(param.grad.detach().norm().cpu())
    return norms


def train_epoch(model, tokenizer, examples, train_indices, marker_tokens, config, device, optimizer) -> float:
    model.train()
    losses = []
    batch_size = int(config["training"]["batch_size"])
    aux_weight = float(config["training"].get("permission_aux_weight", 0.0))
    aux_margin = float(config["training"].get("permission_aux_margin", 0.02))
    debug_gate = bool(config["training"].get("debug_gate", False))
    printed_gate_debug = False
    iterator = tqdm(
        batches(train_indices, batch_size, shuffle=True),
        total=int(math.ceil(len(train_indices) / max(1, batch_size))),
        desc="train",
        unit="batch",
        leave=False,
        disable=not bool(config["training"].get("progress", True)),
    )
    for indices in iterator:
        inputs, marker_mask, token_mask, valid, labels, _ = encode_list_batch(
            tokenizer,
            examples,
            indices,
            marker_tokens=marker_tokens,
            max_length=int(config["model"]["max_length"]),
            device=device,
        )
        result = model(
            **inputs,
            candidate_marker_mask=marker_mask,
            candidate_token_mask=token_mask,
            candidate_valid_mask=valid,
        )
        loss_matrix = F.binary_cross_entropy_with_logits(result.logits, labels, reduction="none")
        loss = (loss_matrix * valid.to(loss_matrix.dtype)).sum() / valid.to(loss_matrix.dtype).sum().clamp_min(1.0)
        aux_loss = None
        if aux_weight > 0.0:
            aux_loss = permission_alignment_loss(
                result,
                labels,
                valid,
                token_mask,
                margin=aux_margin,
            )
            if aux_loss is not None:
                loss = loss + aux_weight * aux_loss
        optimizer.zero_grad()
        loss.backward()
        if debug_gate and not printed_gate_debug:
            norms = gate_grad_norms(model)
            if norms:
                print("gate grad norms:")
                for name, value in sorted(norms.items()):
                    print(f"  {name}: {value:.6e}")
                printed_gate_debug = True
        clip = float(config["training"].get("gradient_clip", 0.0))
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        postfix = {"loss": np.mean(losses)}
        if aux_loss is not None:
            postfix["aux"] = float(aux_loss.detach().cpu())
        gate_score = candidate_gate_scores(result, token_mask)
        if gate_score is not None:
            postfix["gate_std"] = float(gate_score.detach().std().cpu())
        iterator.set_postfix(**postfix)
    return float(np.mean(losses)) if losses else float("nan")


def binary_auprc(labels: list[int], scores: list[float]) -> float:
    order = np.argsort(-np.asarray(scores, dtype=float))
    y = np.asarray(labels, dtype=float)[order]
    positives = y.sum()
    if positives <= 0:
        return float("nan")
    precisions = []
    seen = 0.0
    for rank, label in enumerate(y, start=1):
        if label > 0.5:
            seen += 1.0
            precisions.append(seen / rank)
    return float(np.mean(precisions)) if precisions else float("nan")


def ranking_metrics(values: list[tuple[float, int, int, float, float]]) -> dict[str, float]:
    if not values or not any(label for _, label, _, _, _ in values):
        return {"mrr": float("nan"), "r1": float("nan"), "r2": float("nan"), "r5": float("nan"), "f1": float("nan")}
    order = sorted(values, key=lambda item: item[0], reverse=True)
    labels_ordered = [item[1] for item in order]
    first = next((rank for rank, label in enumerate(labels_ordered, start=1) if label), None)
    gold_count = sum(labels_ordered)
    pred_count = max(1, gold_count)
    predicted = labels_ordered[:pred_count]
    tp = sum(predicted)
    precision = tp / pred_count
    recall = tp / max(1, gold_count)
    f1 = 0.0 if precision + recall <= 0 else 2.0 * precision * recall / (precision + recall)
    if first is None:
        return {"mrr": 0.0, "r1": 0.0, "r2": 0.0, "r5": 0.0, "f1": f1}
    return {
        "mrr": 1.0 / first,
        "r1": float(first <= 1),
        "r2": float(first <= 2),
        "r5": float(first <= 5),
        "f1": f1,
    }


def evaluate(model, tokenizer, examples, eval_indices, marker_tokens, config, device) -> dict[str, float]:
    model.eval()
    batch_size = int(config["training"].get("eval_batch_size", config["training"]["batch_size"]))
    losses = []
    rows: dict[int, list[tuple[float, int, int, float, float]]] = {}
    all_labels, all_scores = [], []
    gold_alpha, hard_alpha, random_alpha = [], [], []
    gold_mass, hard_mass, random_mass = [], [], []
    gold_gate, hard_gate, random_gate = [], [], []
    candidate_gate_values = []
    with torch.no_grad():
        iterator = tqdm(
            batches(eval_indices, batch_size, shuffle=False),
            total=int(math.ceil(len(eval_indices) / max(1, batch_size))),
            desc="eval",
            unit="batch",
            leave=False,
            disable=not bool(config["training"].get("progress", True)),
        )
        for indices in iterator:
            inputs, marker_mask, token_mask, valid, labels, hard = encode_list_batch(
                tokenizer,
                examples,
                indices,
                marker_tokens=marker_tokens,
                max_length=int(config["model"]["max_length"]),
                device=device,
            )
            result = model(
                **inputs,
                candidate_marker_mask=marker_mask,
                candidate_token_mask=token_mask,
                candidate_valid_mask=valid,
            )
            loss_matrix = F.binary_cross_entropy_with_logits(result.logits, labels, reduction="none")
            loss = (loss_matrix * valid.to(loss_matrix.dtype)).sum() / valid.to(loss_matrix.dtype).sum().clamp_min(1.0)
            losses.append(float(loss.cpu()))
            scores = torch.sigmoid(result.logits).detach().cpu()
            labels_cpu = labels.detach().cpu().int()
            hard_cpu = hard.detach().cpu().int()
            valid_cpu = valid.detach().cpu().bool()
            alpha_mass = torch.full_like(scores, float("nan"))
            effective_mass = torch.full_like(scores, float("nan"))
            if {"attention", "permission", "effective_mass"}.issubset(result.aux):
                attention = result.aux["attention"].detach()
                permission = result.aux["permission"].detach()
                effective = result.aux["effective_mass"].detach()
                mask = token_mask.to(attention.device)
                denom_a = mask.to(attention.dtype).sum(dim=-1).clamp_min(1.0)
                alpha_mass = ((attention * mask).sum(dim=-1) / denom_a).cpu()
                effective_mass = ((effective * mask).sum(dim=-1) / denom_a).cpu()
                gate_mass = ((permission * mask).sum(dim=-1) / denom_a).cpu()
                candidate_gate_values.extend(permission[mask].detach().cpu().tolist())
            else:
                gate_mass = torch.full_like(scores, float("nan"))
            for row_pos, example_idx in enumerate(indices):
                example = examples[example_idx]
                rows.setdefault(example.question_id, [])
                for cand_idx, candidate in enumerate(example.candidates[: len(marker_tokens)]):
                    if not bool(valid_cpu[row_pos, cand_idx]):
                        continue
                    score = float(scores[row_pos, cand_idx])
                    label = int(labels_cpu[row_pos, cand_idx])
                    is_hard = int(hard_cpu[row_pos, cand_idx])
                    a_mass = float(alpha_mass[row_pos, cand_idx])
                    e_mass = float(effective_mass[row_pos, cand_idx])
                    g_mass = float(gate_mass[row_pos, cand_idx])
                    rows[example.question_id].append((score, label, is_hard, a_mass, e_mass))
                    all_scores.append(score)
                    all_labels.append(label)
                    if np.isfinite(a_mass):
                        if label:
                            gold_alpha.append(a_mass)
                            gold_mass.append(e_mass)
                            gold_gate.append(g_mass)
                        elif is_hard:
                            hard_alpha.append(a_mass)
                            hard_mass.append(e_mass)
                            hard_gate.append(g_mass)
                        else:
                            random_alpha.append(a_mass)
                            random_mass.append(e_mass)
                            random_gate.append(g_mass)
    rr, clean_rr, hard_rr = [], [], []
    r1, clean_r1, r2, r5, hard_r1, hard_r2, f1s = [], [], [], [], [], [], []
    hard_at_1 = []
    hard_at_gold_count = []
    unsupported_at_gold_count = []
    precision_at_gold_count = []
    gold_hard_pair_auc = []
    score_margin_gold_hard = []
    for qid, values in rows.items():
        metrics = ranking_metrics(values)
        if not np.isfinite(metrics["mrr"]):
            continue

        order = sorted(values, key=lambda item: item[0], reverse=True)
        labels_ordered = [item[1] for item in order]
        gold_count = max(1, sum(labels_ordered))
        top_gold_count = order[:gold_count]

        # Hallucination-style evidence misattribution:
        # hard non-supporting evidence selected as if it were support.
        hard_at_1.append(float(order[0][1] == 0 and order[0][2] == 1))
        hard_at_gold_count.append(
            float(any(item[1] == 0 and item[2] == 1 for item in top_gold_count))
        )
        unsupported_at_gold_count.append(
            float(sum(1 for item in top_gold_count if item[1] == 0) / gold_count)
        )
        precision_at_gold_count.append(
            float(sum(1 for item in top_gold_count if item[1] == 1) / gold_count)
        )

        gold_scores = [item[0] for item in values if item[1] == 1]
        hard_scores = [item[0] for item in values if item[1] == 0 and item[2] == 1]

        if gold_scores and hard_scores:
            wins = []
            margins = []
            for gs in gold_scores:
                for hs in hard_scores:
                    wins.append(float(gs > hs) + 0.5 * float(gs == hs))
                    margins.append(gs - hs)
            gold_hard_pair_auc.append(float(np.mean(wins)))
            score_margin_gold_hard.append(float(np.mean(margins)))

        rr.append(metrics["mrr"])
        r1.append(metrics["r1"])
        r2.append(metrics["r2"])
        r5.append(metrics["r5"])
        f1s.append(metrics["f1"])
        clean_values = [item for item in values if item[1] or not item[2]]
        clean_metrics = ranking_metrics(clean_values)
        if np.isfinite(clean_metrics["mrr"]):
            clean_rr.append(clean_metrics["mrr"])
            clean_r1.append(clean_metrics["r1"])
        hard_values = [item for item in values if item[1] or item[2]]
        hard_metrics = ranking_metrics(hard_values)
        if np.isfinite(hard_metrics["mrr"]):
            hard_rr.append(hard_metrics["mrr"])
            hard_r1.append(hard_metrics["r1"])
            hard_r2.append(hard_metrics["r2"])
    return {
        "eval_loss": float(np.mean(losses)) if losses else float("nan"),
        "support_mrr": float(np.mean(rr)) if rr else float("nan"),
        "clean_support_mrr": float(np.mean(clean_rr)) if clean_rr else float("nan"),
        "hard_support_mrr": float(np.mean(hard_rr)) if hard_rr else float("nan"),
        "recall_at_1": float(np.mean(r1)) if r1 else float("nan"),
        "recall_at_2": float(np.mean(r2)) if r2 else float("nan"),
        "recall_at_5": float(np.mean(r5)) if r5 else float("nan"),
        "clean_recall_at_1": float(np.mean(clean_r1)) if clean_r1 else float("nan"),
        "hard_recall_at_1": float(np.mean(hard_r1)) if hard_r1 else float("nan"),
        "hard_recall_at_2": float(np.mean(hard_r2)) if hard_r2 else float("nan"),
        "evidence_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "auprc": binary_auprc(all_labels, all_scores),
        "hard_at_1": float(np.mean(hard_at_1)) if hard_at_1 else float("nan"),
        "hard_at_gold_count": float(np.mean(hard_at_gold_count)) if hard_at_gold_count else float("nan"),
        "unsupported_at_gold_count": float(np.mean(unsupported_at_gold_count)) if unsupported_at_gold_count else float("nan"),
        "precision_at_gold_count": float(np.mean(precision_at_gold_count)) if precision_at_gold_count else float("nan"),
        "gold_hard_pair_auc": float(np.mean(gold_hard_pair_auc)) if gold_hard_pair_auc else float("nan"),
        "score_margin_gold_hard": float(np.mean(score_margin_gold_hard)) if score_margin_gold_hard else float("nan"),
        "candidate_gate_mean": float(np.mean(candidate_gate_values)) if candidate_gate_values else float("nan"),
        "candidate_gate_std": float(np.std(candidate_gate_values)) if candidate_gate_values else float("nan"),
        "gold_gate_mean": float(np.mean(gold_gate)) if gold_gate else float("nan"),
        "hard_gate_mean": float(np.mean(hard_gate)) if hard_gate else float("nan"),
        "random_gate_mean": float(np.mean(random_gate)) if random_gate else float("nan"),
        "gold_attention_mass": float(np.mean(gold_alpha)) if gold_alpha else float("nan"),
        "hard_attention_mass": float(np.mean(hard_alpha)) if hard_alpha else float("nan"),
        "random_attention_mass": float(np.mean(random_alpha)) if random_alpha else float("nan"),
        "gold_effective_mass": float(np.mean(gold_mass)) if gold_mass else float("nan"),
        "hard_effective_mass": float(np.mean(hard_mass)) if hard_mass else float("nan"),
        "random_effective_mass": float(np.mean(random_mass)) if random_mass else float("nan"),
        "gold_hard_attention_ratio": float(np.mean(gold_alpha) / max(1.0e-8, np.mean(hard_alpha))) if gold_alpha and hard_alpha else float("nan"),
        "gold_hard_effective_ratio": float(np.mean(gold_mass) / max(1.0e-8, np.mean(hard_mass))) if gold_mass and hard_mass else float("nan"),
        "gold_random_attention_ratio": float(np.mean(gold_alpha) / max(1.0e-8, np.mean(random_alpha))) if gold_alpha and random_alpha else float("nan"),
        "gold_random_effective_ratio": float(np.mean(gold_mass) / max(1.0e-8, np.mean(random_mass))) if gold_mass and random_mass else float("nan"),
    }


def output_dir(config: dict[str, Any], variant: str, seed: int) -> Path:
    return REPO_ROOT / str(config["experiment"]["output_root"]) / variant / f"seed_{seed}"


def append_result(path: Path, row: dict[str, Any]) -> None:
    rows = []
    key = (row["variant"], int(row["seed"]))
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            for existing in csv.DictReader(handle):
                if (existing.get("variant"), int(existing.get("seed", -1))) != key:
                    rows.append(existing)
    rows.append({column: row.get(column, "") for column in RESULT_COLUMNS})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run_one(config: dict[str, Any], variant: str, seed: int, *, force: bool, dry_run: bool) -> dict[str, Any]:
    out_dir = output_dir(config, variant, seed)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists() and not force:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    if dry_run:
        print(f"would run {variant} seed={seed} -> {out_dir}")
        return {}
    set_seed(seed)
    start = time.time()
    device = resolve_device(str(config["experiment"].get("device", "auto")))
    raw_examples = read_jsonl(REPO_ROOT / str(config["dataset"]["path"]), int(config["candidate_builder"]["max_questions"]))
    candidates, by_question = build_candidates(raw_examples, config)
    examples = build_list_examples(candidates, by_question)
    train_qids, eval_qids = split_questions([example.question_id for example in examples], float(config["candidate_builder"]["eval_ratio"]))
    train_indices = [idx for idx, example in enumerate(examples) if example.question_id in train_qids]
    eval_indices = [idx for idx, example in enumerate(examples) if example.question_id in eval_qids]
    tokenizer = AutoTokenizer.from_pretrained(str(config["model"]["model_name"]))
    marker_tokens = [f"<C{idx}>" for idx in range(int(config["candidate_builder"]["max_candidates_per_question"]))]
    tokenizer.add_special_tokens({"additional_special_tokens": marker_tokens})
    model = BertHotpotMultiCandidateWarrantSelector(
        model_name=str(config["model"]["model_name"]),
        variant=variant,
        gate_init=float(config["model"].get("gate_init", 0.95)),
        gate_leak=float(config["model"].get("gate_leak", 0.1)),
        dropout=float(config["model"].get("dropout", 0.1)),
        freeze_encoder=bool(config["model"].get("freeze_encoder", False)),
        vocab_size=len(tokenizer),
    ).to(device)
    total_parameters = sum(param.numel() for param in model.parameters())
    trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    base_lr = float(config["training"]["learning_rate"])
    gate_lr = base_lr * float(config["training"].get("gate_learning_rate_multiplier", 1.0))
    gate_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "gate" in name:
            gate_params.append(param)
        else:
            other_params.append(param)
    param_groups = [{"params": other_params, "lr": base_lr}]
    if gate_params:
        param_groups.append({"params": gate_params, "lr": gate_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(config["training"].get("weight_decay", 0.0)))
    train_loss = float("nan")
    for epoch in tqdm(
        range(1, int(config["training"]["epochs"]) + 1),
        desc=f"{variant} seed={seed}",
        unit="epoch",
        disable=not bool(config["training"].get("progress", True)),
    ):
        train_loss = train_epoch(model, tokenizer, examples, train_indices, marker_tokens, config, device, optimizer)
    metrics = evaluate(model, tokenizer, examples, eval_indices, marker_tokens, config, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    if variant == "full_warrant":
        torch.save({"model": model.state_dict(), "config": config, "variant": variant, "seed": seed}, out_dir / "checkpoint.pt")
    row = {
        "dataset": config["dataset"]["name"],
        "model": config["model"]["model_name"],
        "variant": variant,
        "seed": seed,
        "status": "completed",
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "train_loss": train_loss,
        **metrics,
        "elapsed_sec": time.time() - start,
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
    }
    metrics_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    append_result(REPO_ROOT / str(config["experiment"]["output_root"]) / "results.csv", row)
    return row


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    variants = selected(args.variant, list(config["variants"]))
    seeds = [int(seed) for seed in selected(args.seed, list(config["experiment"].get("seeds", [7])))]
    for variant in variants:
        for seed in seeds:
            row = run_one(config, variant, seed, force=args.force, dry_run=args.dry_run)
            if row:
                print(f"{variant} seed={seed}: support_mrr={row.get('support_mrr')}")


if __name__ == "__main__":
    main()
