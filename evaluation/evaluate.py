from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.factory import build_model
from models.attention import WarrantedAttention
from models.official import build_dyglib_inputs
from models.warrant_block import EdgeConditionedWarrantBlock
from models.warrant_patches import (
    WarrantedDyGLibMultiHeadAttention,
    WarrantedEasyTPPMultiHeadAttention,
    WarrantedNeuralSTPPMultiheadAttention,
    WarrantedTorchMultiheadAttention,
)


@dataclass(frozen=True)
class RunSpec:
    domain: str
    dataset: str
    dataset_path: Path
    model: str
    variant: str
    seed: int
    implementation: str
    model_config: dict[str, Any]
    use_warrant: bool


def hash_token(value: Any, vocab_size: int) -> int:
    text = str(value)
    digest = hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % max(1, vocab_size)


def label_encode(values: pd.Series, limit: int | None = None) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(values.astype(str), sort=True)
    if limit is not None:
        codes = np.asarray([int(code % limit) for code in codes], dtype=np.int64)
        return codes, int(min(limit, len(uniques)))
    return codes.astype(np.int64), int(len(uniques))


def chronological_sample(frame: pd.DataFrame, max_examples: int) -> pd.DataFrame:
    if "timestamp" in frame.columns:
        frame = frame.sort_values("timestamp")
    if len(frame) <= max_examples:
        return frame.reset_index(drop=True)
    return frame.iloc[-max_examples:].reset_index(drop=True)


def split_indices(n: int, eval_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    split = max(1, int(n * (1.0 - eval_ratio)))
    split = min(split, max(1, n - 1))
    return np.arange(split), np.arange(split, n)


def batch_iter(indices: np.ndarray, batch_size: int, *, shuffle: bool) -> Iterable[np.ndarray]:
    indices = indices.copy()
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def progress_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("training", {}).get("progress", True))


def epoch_iter(spec: RunSpec, config: dict[str, Any]):
    epochs = int(config["training"]["epochs"])
    return tqdm(
        range(epochs),
        desc=f"{spec.domain}/{spec.dataset} {spec.model} {spec.variant} train",
        unit="epoch",
        leave=True,
        disable=not progress_enabled(config),
    )


def progress_batches(
    indices: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    desc: str,
    config: dict[str, Any],
):
    return tqdm(
        batch_iter(indices, batch_size, shuffle=shuffle),
        total=int(np.ceil(len(indices) / max(1, batch_size))),
        desc=desc,
        unit="batch",
        leave=False,
        disable=not progress_enabled(config),
    )


def resolve_auto_model_config(domain: str, frame: Any, config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    if domain == "ctdg":
        max_node = int(max(frame["src"].max(), frame["dst"].max())) + 1
        if resolved.get("num_nodes") == "auto":
            resolved["num_nodes"] = max_node
    elif domain == "mtpp":
        if resolved.get("num_event_types") == "auto":
            resolved["num_event_types"] = int(frame["mark_id"].max()) + 1
    elif domain == "stpp":
        if resolved.get("num_marks") == "auto":
            resolved["num_marks"] = int(frame["mark_id"].max()) + 1
    elif domain == "tkg":
        if resolved.get("num_entities") == "auto":
            resolved["num_entities"] = int(max(frame["head_id"].max(), frame["tail_id"].max())) + 1
        if resolved.get("num_relations") == "auto":
            resolved["num_relations"] = int(frame["relation_id"].max()) + 1
    return resolved


def build_benchmark_model(spec: RunSpec, frame: Any, config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_config = resolve_auto_model_config(spec.domain, frame, spec.model_config)
    warrant = config.get("warrant", {})
    model_config.setdefault("dropout", config.get("training", {}).get("dropout", 0.1))
    model_config.setdefault("gate_init", float(warrant.get("gate_init", 0.95)))
    model_config.setdefault("gate_leak", float(warrant.get("gate_leak", 0.05)))
    if spec.implementation == "official":
        if device.type == "cuda":
            model_config.setdefault("device", f"cuda:{0 if device.index is None else device.index}")
            model_config.setdefault("gpu", 0 if device.index is None else device.index)
        else:
            model_config.setdefault("device", "cpu")
            model_config.setdefault("gpu", -1)
        if spec.domain == "ctdg":
            model_config.update(
                build_dyglib_inputs(
                    frame,
                    node_feat_dim=int(model_config.get("hidden_dim", 128)),
                    edge_feat_dim=int(model_config.get("edge_feat_dim", 1)),
                    seed=spec.seed,
                )
            )
    model = build_model(
        spec.model,
        implementation=spec.implementation,
        use_warrant=spec.use_warrant,
        **model_config,
    )
    return model.to(device)


def audit_warrant_application(model: torch.nn.Module, spec: RunSpec) -> dict[str, Any]:
    active_wrappers = (
        WarrantedDyGLibMultiHeadAttention,
        WarrantedEasyTPPMultiHeadAttention,
        WarrantedNeuralSTPPMultiheadAttention,
        WarrantedTorchMultiheadAttention,
    )
    active_blocks = 0
    for module in model.modules():
        if isinstance(module, WarrantedAttention) and module.use_warrant:
            active_blocks += 1
        elif isinstance(module, EdgeConditionedWarrantBlock):
            active_blocks += 1
        elif bool(getattr(module, "_warrant_audit_active", False)):
            active_blocks += 1
        elif isinstance(module, active_wrappers):
            active_blocks += 1
        elif hasattr(module, "_warrant_block"):
            active_blocks += 1

    replacements = int(getattr(model, "warrant_replacements", 0))
    effective = getattr(model, "official_name", None)
    if effective is None and hasattr(model, "model"):
        effective = getattr(getattr(model, "model"), "official_name", None)
    effective = effective or model.__class__.__name__

    warrant_expected = bool(getattr(model, "warrant_expected", spec.use_warrant))
    if spec.use_warrant and warrant_expected and active_blocks == 0:
        raise RuntimeError(
            f"use_warrant=True for {spec.model}, but no active WarrantBlock-backed attention was found. "
            "This run would produce a fake warrant result."
        )

    return {
        "effective_implementation": effective,
        "warrant_active_blocks": float(active_blocks),
        "warrant_replacements": float(replacements),
    }


def binary_auc(labels: list[float], scores: list[float]) -> float:
    positives = [(score, label) for score, label in zip(scores, labels) if label > 0.5]
    negatives = [(score, label) for score, label in zip(scores, labels) if label <= 0.5]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for p_score, _ in positives:
        for n_score, _ in negatives:
            wins += 1.0 if p_score > n_score else 0.5 if p_score == n_score else 0.0
    return wins / (len(positives) * len(negatives))


def build_ctdg_node_histories(
    frame: pd.DataFrame,
    history_len: int,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build source-node temporal histories available strictly before each event."""

    nodes = np.zeros((len(frame), history_len), dtype=np.int64)
    times = np.zeros((len(frame), history_len), dtype=np.float32)
    masks = np.zeros((len(frame), history_len), dtype=bool)
    memories: list[deque[tuple[int, float]]] = [deque(maxlen=history_len) for _ in range(num_nodes)]

    src_values = frame["src"].to_numpy(dtype=np.int64)
    dst_values = frame["dst"].to_numpy(dtype=np.int64)
    ts_values = frame["timestamp"].to_numpy(dtype=np.float32)
    for idx, (src, dst, ts) in enumerate(zip(src_values, dst_values, ts_values)):
        history = list(memories[int(src)])
        if history:
            width = min(history_len, len(history))
            selected = history[-width:]
            start = history_len - width
            nodes[idx, start:] = [item[0] for item in selected]
            times[idx, start:] = [item[1] for item in selected]
            masks[idx, start:] = True
        memories[int(src)].append((int(dst), float(ts)))
        if src != dst:
            memories[int(dst)].append((int(src), float(ts)))
    return nodes, times, masks


class CTDGHistoryIndex:
    def __init__(self, frame: pd.DataFrame, history_len: int, num_nodes: int) -> None:
        self.history_len = int(history_len)
        self.node_rows: list[np.ndarray] = []
        self.node_neighbors: list[np.ndarray] = []
        self.node_times: list[np.ndarray] = []
        buckets: list[list[tuple[int, int, float]]] = [[] for _ in range(num_nodes)]
        src_values = frame["src"].to_numpy(dtype=np.int64)
        dst_values = frame["dst"].to_numpy(dtype=np.int64)
        ts_values = frame["timestamp"].to_numpy(dtype=np.float32)
        for row, (src, dst, ts) in enumerate(zip(src_values, dst_values, ts_values)):
            buckets[int(src)].append((row, int(dst), float(ts)))
            if src != dst:
                buckets[int(dst)].append((row, int(src), float(ts)))
        for bucket in buckets:
            bucket.sort(key=lambda item: item[0])
            self.node_rows.append(np.asarray([item[0] for item in bucket], dtype=np.int64))
            self.node_neighbors.append(np.asarray([item[1] for item in bucket], dtype=np.int64))
            self.node_times.append(np.asarray([item[2] for item in bucket], dtype=np.float32))

    def gather(self, node_ids: np.ndarray, row_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nodes = np.zeros((len(node_ids), self.history_len), dtype=np.int64)
        times = np.zeros((len(node_ids), self.history_len), dtype=np.float32)
        masks = np.zeros((len(node_ids), self.history_len), dtype=bool)
        for idx, (node_id, row_id) in enumerate(zip(node_ids, row_ids)):
            if node_id < 0 or node_id >= len(self.node_rows):
                continue
            rows = self.node_rows[int(node_id)]
            end = int(np.searchsorted(rows, int(row_id), side="left"))
            if end <= 0:
                continue
            start = max(0, end - self.history_len)
            width = end - start
            offset = self.history_len - width
            nodes[idx, offset:] = self.node_neighbors[int(node_id)][start:end]
            times[idx, offset:] = self.node_times[int(node_id)][start:end]
            masks[idx, offset:] = True
        return nodes, times, masks


def run_ctdg(spec: RunSpec, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    training = config["training"]
    frame = chronological_sample(pd.read_csv(spec.dataset_path), int(training["max_examples"]))
    frame["src"] = pd.to_numeric(frame["src"], errors="coerce").fillna(0).astype("int64")
    frame["dst"] = pd.to_numeric(frame["dst"], errors="coerce").fillna(0).astype("int64")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(0.0).astype("float32")
    train_idx, eval_idx = split_indices(len(frame), float(training["eval_ratio"]))
    model = build_benchmark_model(spec, frame, config, device)
    model_audit = audit_warrant_application(model, spec)
    opt = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training.get("weight_decay", 0.0)))
    history_len = int(training["history_len"])
    num_nodes = int(max(frame["src"].max(), frame["dst"].max())) + 1
    dst_pool = frame["dst"].drop_duplicates().to_numpy(dtype=np.int64)
    source_hist_nodes, source_hist_times, source_hist_mask = build_ctdg_node_histories(frame, history_len, num_nodes)
    history_index = CTDGHistoryIndex(frame, history_len, num_nodes)

    def make_batch(rows: np.ndarray, negative: bool):
        src = torch.tensor(frame.iloc[rows]["src"].to_numpy(), device=device)
        dst_values = frame.iloc[rows]["dst"].to_numpy().copy()
        if negative:
            src_values = frame.iloc[rows]["src"].to_numpy()
            pos_values = frame.iloc[rows]["dst"].to_numpy()
            dst_values = np.random.choice(dst_pool, size=len(rows), replace=True)
            bad = (dst_values == src_values) | (dst_values == pos_values)
            while bad.any():
                dst_values[bad] = np.random.choice(dst_pool, size=int(bad.sum()), replace=True)
                bad = (dst_values == src_values) | (dst_values == pos_values)
        dst = torch.tensor(dst_values, device=device)
        ts = torch.tensor(frame.iloc[rows]["timestamp"].to_numpy(), dtype=torch.float32, device=device)
        hist_nodes_t = torch.tensor(source_hist_nodes[rows], device=device)
        hist_times_t = torch.tensor(source_hist_times[rows], dtype=torch.float32, device=device)
        edge_feats = torch.zeros((len(rows), history_len, 1), dtype=torch.float32, device=device)
        masks_t = torch.tensor(source_hist_mask[rows], dtype=torch.bool, device=device)
        dst_hist_nodes, dst_hist_times, dst_hist_masks = history_index.gather(dst_values.astype(np.int64), rows.astype(np.int64))
        dst_hist_nodes_t = torch.tensor(dst_hist_nodes, device=device)
        dst_hist_times_t = torch.tensor(dst_hist_times, dtype=torch.float32, device=device)
        dst_edge_feats = torch.zeros((len(rows), history_len, 1), dtype=torch.float32, device=device)
        dst_masks_t = torch.tensor(dst_hist_masks, dtype=torch.bool, device=device)
        return (
            src,
            dst,
            ts,
            hist_nodes_t,
            hist_times_t,
            edge_feats,
            masks_t,
            dst_hist_nodes_t,
            dst_hist_times_t,
            dst_edge_feats,
            dst_masks_t,
        )

    last_train = 0.0
    for epoch in epoch_iter(spec, config):
        model.train()
        losses = []
        for rows in progress_batches(
            train_idx,
            int(training["batch_size"]),
            shuffle=True,
            desc=f"epoch {epoch + 1}/{int(training['epochs'])} batches",
            config=config,
        ):
            pos = make_batch(rows, negative=False)
            neg = make_batch(rows, negative=True)
            pos_logits = model(*pos).logits
            neg_logits = model(*neg).logits
            logits = torch.cat([pos_logits, neg_logits])
            labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        last_train = float(np.mean(losses)) if losses else 0.0
        if progress_enabled(config):
            tqdm.write(f"{spec.model}/{spec.variant} epoch {epoch + 1}: train_loss={last_train:.6f}")

    model.eval()
    labels_all, scores_all, losses = [], [], []
    with torch.no_grad():
        for rows in progress_batches(eval_idx, int(training["batch_size"]), shuffle=False, desc="eval batches", config=config):
            pos = make_batch(rows, negative=False)
            neg = make_batch(rows, negative=True)
            pos_logits = model(*pos).logits
            neg_logits = model(*neg).logits
            logits = torch.cat([pos_logits, neg_logits])
            labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
            losses.append(float(F.binary_cross_entropy_with_logits(logits, labels).cpu()))
            labels_all.extend(labels.cpu().tolist())
            scores_all.extend(torch.sigmoid(logits).cpu().tolist())
    preds = [1.0 if score >= 0.5 else 0.0 for score in scores_all]
    acc = float(np.mean([pred == label for pred, label in zip(preds, labels_all)])) if labels_all else 0.0
    auc = binary_auc(labels_all, scores_all)
    return {
        "train_loss": last_train,
        "eval_loss": float(np.mean(losses)),
        "primary_metric": auc,
        "accuracy": acc,
        "auc": auc,
        "examples": float(len(frame)),
        **model_audit,
    }


def prepare_sequence_frame(path: Path, max_examples: int) -> pd.DataFrame:
    frame = chronological_sample(pd.read_csv(path), max_examples)
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(0.0).astype("float32")
    if "sequence_id" not in frame.columns:
        frame["sequence_id"] = "sequence_0"
    frame["mark_id"], _ = label_encode(frame.get("mark", pd.Series(["event"] * len(frame))))
    return frame


def sequence_windows(frame: pd.DataFrame, history_len: int) -> list[tuple[np.ndarray, np.ndarray, int, float, int]]:
    windows = []
    for _, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values("timestamp")
        marks = group["mark_id"].to_numpy(dtype=np.int64)
        times = group["timestamp"].to_numpy(dtype=np.float32)
        for idx in range(1, len(group)):
            start = max(0, idx - history_len)
            hist_marks = marks[start:idx]
            hist_times = times[start:idx]
            pad = history_len - len(hist_marks)
            windows.append(
                (
                    np.pad(hist_marks, (pad, 0)),
                    np.pad(hist_times, (pad, 0)),
                    int(marks[idx]),
                    float(max(0.0, times[idx] - times[idx - 1])),
                    int(len(hist_marks)),
                )
            )
    return windows


def single_sequence_windows(frame: pd.DataFrame, history_len: int) -> list[tuple[np.ndarray, np.ndarray, int, float, int]]:
    ordered = frame.sort_values("timestamp")
    marks = ordered["mark_id"].to_numpy(dtype=np.int64)
    times = ordered["timestamp"].to_numpy(dtype=np.float32)
    windows = []
    for idx in range(1, len(ordered)):
        start = max(0, idx - history_len)
        hist_marks = marks[start:idx]
        hist_times = times[start:idx]
        pad = history_len - len(hist_marks)
        windows.append(
            (
                np.pad(hist_marks, (pad, 0)),
                np.pad(hist_times, (pad, 0)),
                int(marks[idx]),
                float(max(0.0, times[idx] - times[idx - 1])),
                int(len(hist_marks)),
            )
        )
    return windows


def stpp_warrant_targets(
    coords: torch.Tensor,
    target_coords: torch.Tensor,
    mask: torch.Tensor,
    *,
    top_k: int = 3,
) -> torch.Tensor:
    valid = mask.to(torch.bool)
    if not bool(valid.any()):
        return coords.new_zeros(coords.shape[:2])
    distances = torch.linalg.norm(coords - target_coords.unsqueeze(1), dim=-1)
    max_distance = distances.masked_fill(~valid, 0.0).max(dim=1, keepdim=True).values.clamp_min(1.0)
    spatial_score = 1.0 - (distances / max_distance).clamp(0.0, 1.0)
    positions = torch.linspace(0.0, 1.0, steps=coords.shape[1], device=coords.device, dtype=coords.dtype).unsqueeze(0)
    support_score = 0.8 * spatial_score + 0.2 * positions
    support_score = support_score.masked_fill(~valid, -torch.inf)
    k = min(int(top_k), coords.shape[1])
    indices = torch.topk(support_score, k=k, dim=1).indices
    targets = coords.new_zeros(coords.shape[:2])
    targets.scatter_(1, indices, 1.0)
    return targets * valid.to(targets.dtype)


def stpp_last_query_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        return logits
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand_as(mask)
    indices = torch.where(mask, positions, torch.full_like(positions, -1)).max(dim=1).values.clamp_min(0)
    return logits[torch.arange(logits.shape[0], device=logits.device), indices]


def tkg_warrant_targets(
    head: torch.Tensor,
    relation: torch.Tensor,
    history_head: torch.Tensor,
    history_relation: torch.Tensor,
    history_tail: torch.Tensor,
    target_tail: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask.to(torch.bool)
    target_hit = (history_tail.long() == target_tail.long().unsqueeze(1)).to(torch.float32)
    query_head_touch = (
        (history_head.long() == head.long().unsqueeze(1))
        | (history_tail.long() == head.long().unsqueeze(1))
    ).to(torch.float32)
    relation_match = (history_relation.long() == relation.long().unsqueeze(1)).to(torch.float32)
    support = torch.maximum(target_hit, query_head_touch * relation_match)
    return support * valid.to(support.dtype)


def build_tkg_histories(
    frame: pd.DataFrame,
    history_len: int,
    *,
    strict_time: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute fixed-length TKG histories.

    By default this matches the original benchmark behavior: previous rows in
    the sorted stream are evidence, including earlier rows with the same
    timestamp.  ``strict_time=True`` is available for leakage audits.
    """

    n_rows = len(frame)
    history_len = int(history_len)
    heads = frame["head_id"].to_numpy(dtype=np.int64)
    relations = frame["relation_id"].to_numpy(dtype=np.int64)
    tails = frame["tail_id"].to_numpy(dtype=np.int64)
    times = frame["timestamp_id"].to_numpy(dtype=np.float32)
    hist_heads = np.zeros((n_rows, history_len), dtype=np.int64)
    hist_relations = np.zeros((n_rows, history_len), dtype=np.int64)
    hist_tails = np.zeros((n_rows, history_len), dtype=np.int64)
    hist_times = np.zeros((n_rows, history_len), dtype=np.float32)
    hist_masks = np.zeros((n_rows, history_len), dtype=bool)

    for row, query_time in enumerate(times):
        end = int(np.searchsorted(times, query_time, side="left")) if strict_time else row
        start = max(0, end - history_len)
        length = end - start
        if length <= 0:
            continue
        offset = history_len - length
        hist_heads[row, offset:] = heads[start:end]
        hist_relations[row, offset:] = relations[start:end]
        hist_tails[row, offset:] = tails[start:end]
        hist_times[row, offset:] = times[start:end]
        hist_masks[row, offset:] = True
    return hist_heads, hist_relations, hist_tails, hist_times, hist_masks


def build_strict_tkg_histories(frame: pd.DataFrame, history_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return build_tkg_histories(frame, history_len, strict_time=True)


def stpp_joint_nll(
    result: ModelResult,
    targets: torch.Tensor,
    target_coords: torch.Tensor,
    dtime: torch.Tensor,
    *,
    num_marks: int,
) -> torch.Tensor:
    mark_nll = F.cross_entropy(result.logits, targets) if num_marks > 1 else result.logits.sum() * 0.0
    loc = result.aux["next_location"]
    loc_log_scale = result.aux["location_log_scale"].clamp(-5.0, 5.0)
    loc_var = torch.exp(2.0 * loc_log_scale).view(1, -1)
    location_nll = (
        0.5 * ((target_coords - loc) ** 2 / loc_var).sum(dim=-1)
        + loc_log_scale.sum()
        + math.log(2.0 * math.pi)
    ).mean()

    log_target_time = torch.log1p(dtime.clamp_min(0.0))
    time_log_mean = result.aux["time_log_mean"]
    time_log_scale = result.aux["time_log_scale"].clamp(-5.0, 5.0)
    time_var = torch.exp(2.0 * time_log_scale)
    time_nll = (
        0.5 * ((log_target_time - time_log_mean) ** 2 / time_var)
        + time_log_scale
        + 0.5 * math.log(2.0 * math.pi)
        + torch.log1p(dtime.clamp_min(0.0))
    ).mean()
    return mark_nll + location_nll + time_nll


def run_mtpp(spec: RunSpec, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    training = config["training"]
    frame = prepare_sequence_frame(spec.dataset_path, int(training["max_examples"]))
    windows = sequence_windows(frame, int(training["history_len"]))
    if len(windows) < 2:
        windows = single_sequence_windows(frame, int(training["history_len"]))
    if len(windows) < 2:
        raise ValueError("Not enough MTPP windows")
    train_idx, eval_idx = split_indices(len(windows), float(training["eval_ratio"]))
    model = build_benchmark_model(spec, frame, config, device)
    model_audit = audit_warrant_application(model, spec)
    opt = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training.get("weight_decay", 0.0)))

    def make_batch(rows: np.ndarray):
        batch = [windows[int(row)] for row in rows]
        marks = torch.tensor(np.stack([item[0] for item in batch]), device=device)
        times = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32, device=device)
        targets = torch.tensor([item[2] for item in batch], device=device)
        dtime = torch.tensor([item[3] for item in batch], dtype=torch.float32, device=device)
        mask = torch.tensor(np.stack([np.arange(marks.shape[1]) >= marks.shape[1] - item[4] for item in batch]), dtype=torch.bool, device=device)
        return marks, times, targets, dtime, mask

    last_train = 0.0
    for epoch in epoch_iter(spec, config):
        model.train()
        losses = []
        for rows in progress_batches(
            train_idx,
            int(training["batch_size"]),
            shuffle=True,
            desc=f"epoch {epoch + 1}/{int(training['epochs'])} batches",
            config=config,
        ):
            marks, times, targets, dtime, mask = make_batch(rows)
            result = model(marks, times, mask)
            loss = F.cross_entropy(result.logits, targets) + 0.01 * F.smooth_l1_loss(result.aux["next_time"], dtime)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        last_train = float(np.mean(losses)) if losses else 0.0
        if progress_enabled(config):
            tqdm.write(f"{spec.model}/{spec.variant} epoch {epoch + 1}: train_loss={last_train:.6f}")

    model.eval()
    losses, correct, total, maes, reciprocal_ranks = [], 0, 0, [], []
    with torch.no_grad():
        for rows in progress_batches(eval_idx, int(training["batch_size"]), shuffle=False, desc="eval batches", config=config):
            marks, times, targets, dtime, mask = make_batch(rows)
            result = model(marks, times, mask)
            loss = F.cross_entropy(result.logits, targets) + 0.01 * F.smooth_l1_loss(result.aux["next_time"], dtime)
            losses.append(float(loss.cpu()))
            correct += int((result.logits.argmax(dim=-1) == targets).sum().cpu())
            total += int(targets.numel())
            target_logits = result.logits.gather(1, targets.unsqueeze(1))
            ranks = (result.logits > target_logits).sum(dim=1).to(torch.float32) + 1.0
            reciprocal_ranks.extend((1.0 / ranks).cpu().tolist())
            maes.extend(torch.abs(result.aux["next_time"] - dtime).cpu().tolist())
    acc = correct / max(1, total)
    mark_mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan")
    return {
        "train_loss": last_train,
        "eval_loss": float(np.mean(losses)),
        "primary_metric": mark_mrr,
        "accuracy": acc,
        "mark_mrr": mark_mrr,
        "mae_time": float(np.mean(maes)),
        "examples": float(len(windows)),
        **model_audit,
    }


def run_stpp(spec: RunSpec, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    training = config["training"]
    frame = prepare_sequence_frame(spec.dataset_path, int(training["max_examples"]))
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(0.0).astype("float64")
    frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce").fillna(0.0).astype("float32")
    frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce").fillna(0.0).astype("float32")
    for column in ("lat", "lon"):
        std = float(frame[column].std() or 1.0)
        frame[column] = (frame[column] - float(frame[column].mean())) / std
    time_origin = float(frame["timestamp"].min())
    time_scale = float(frame["timestamp"].std() or 1.0)
    frame["time_value"] = ((frame["timestamp"] - time_origin) / max(time_scale, 1.0)).astype("float32")

    delta_values: list[float] = []
    for _, group in frame.groupby("sequence_id", sort=False):
        ts = group.sort_values("timestamp")["timestamp"].to_numpy(dtype=np.float64)
        if len(ts) > 1:
            delta_values.extend(np.log1p(np.maximum(0.0, np.diff(ts))).tolist())
    if not delta_values:
        ordered_ts = frame.sort_values("timestamp")["timestamp"].to_numpy(dtype=np.float64)
        if len(ordered_ts) > 1:
            delta_values.extend(np.log1p(np.maximum(0.0, np.diff(ordered_ts))).tolist())
    delta_scale = float(np.mean(delta_values) or 1.0)
    delta_scale = max(delta_scale, 1.0)

    windows = []
    history_len = int(training["history_len"])
    for _, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values("timestamp")
        marks = group["mark_id"].to_numpy(dtype=np.int64)
        raw_times = group["timestamp"].to_numpy(dtype=np.float64)
        times = group["time_value"].to_numpy(dtype=np.float32)
        coords = group[["lat", "lon"]].to_numpy(dtype=np.float32)
        for idx in range(1, len(group)):
            start = max(0, idx - history_len)
            pad = history_len - (idx - start)
            delta_target = np.log1p(max(0.0, raw_times[idx] - raw_times[idx - 1])) / delta_scale
            windows.append(
                (
                    np.pad(marks[start:idx], (pad, 0)),
                    np.pad(times[start:idx], (pad, 0)),
                    np.pad(coords[start:idx], ((pad, 0), (0, 0))),
                    int(marks[idx]),
                    coords[idx],
                    float(delta_target),
                    idx - start,
                )
            )
    if len(windows) < 2:
        ordered = frame.sort_values("timestamp")
        marks = ordered["mark_id"].to_numpy(dtype=np.int64)
        raw_times = ordered["timestamp"].to_numpy(dtype=np.float64)
        times = ordered["time_value"].to_numpy(dtype=np.float32)
        coords = ordered[["lat", "lon"]].to_numpy(dtype=np.float32)
        windows = []
        for idx in range(1, len(ordered)):
            start = max(0, idx - history_len)
            pad = history_len - (idx - start)
            delta_target = np.log1p(max(0.0, raw_times[idx] - raw_times[idx - 1])) / delta_scale
            windows.append(
                (
                    np.pad(marks[start:idx], (pad, 0)),
                    np.pad(times[start:idx], (pad, 0)),
                    np.pad(coords[start:idx], ((pad, 0), (0, 0))),
                    int(marks[idx]),
                    coords[idx],
                    float(delta_target),
                    idx - start,
                )
            )
    if len(windows) < 2:
        raise ValueError("Not enough STPP windows")
    train_idx, eval_idx = split_indices(len(windows), float(training["eval_ratio"]))
    model = build_benchmark_model(spec, frame, config, device)
    model_audit = audit_warrant_application(model, spec)
    opt = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training.get("weight_decay", 0.0)))
    num_marks = int(frame["mark_id"].max()) + 1
    gate_loss_weight = float(spec.model_config.get("gate_loss_weight", config.get("stpp", {}).get("gate_loss_weight", 0.05)))

    def make_batch(rows: np.ndarray):
        batch = [windows[int(row)] for row in rows]
        marks = torch.tensor(np.stack([item[0] for item in batch]), device=device)
        times = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32, device=device)
        coords = torch.tensor(np.stack([item[2] for item in batch]), dtype=torch.float32, device=device)
        targets = torch.tensor([item[3] for item in batch], device=device)
        target_coords = torch.tensor(np.stack([item[4] for item in batch]), dtype=torch.float32, device=device)
        dtime = torch.tensor([item[5] for item in batch], dtype=torch.float32, device=device)
        mask = torch.tensor(np.stack([np.arange(marks.shape[1]) >= marks.shape[1] - item[6] for item in batch]), dtype=torch.bool, device=device)
        return times, coords, marks, targets, target_coords, dtime, mask

    last_train = 0.0
    for epoch in epoch_iter(spec, config):
        model.train()
        losses = []
        for rows in progress_batches(
            train_idx,
            int(training["batch_size"]),
            shuffle=True,
            desc=f"epoch {epoch + 1}/{int(training['epochs'])} batches",
            config=config,
        ):
            times, coords, marks, targets, target_coords, dtime, mask = make_batch(rows)
            result = model(times, coords, marks, mask)
            task_loss = stpp_joint_nll(result, targets, target_coords, dtime, num_marks=num_marks)
            loss = task_loss
            if spec.use_warrant and gate_loss_weight > 0.0 and "raw_warrant_logits" in result.aux:
                warrant_targets = stpp_warrant_targets(coords, target_coords, mask)
                warrant_logits = stpp_last_query_logits(result.aux["raw_warrant_logits"], mask)
                loss = loss + gate_loss_weight * masked_gate_bce(warrant_logits, warrant_targets, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(task_loss.detach().cpu()))
        last_train = float(np.mean(losses)) if losses else 0.0
        if progress_enabled(config):
            tqdm.write(f"{spec.model}/{spec.variant} epoch {epoch + 1}: train_loss={last_train:.6f}")

    model.eval()
    losses, correct, total, rmses, time_maes = [], 0, 0, [], []
    with torch.no_grad():
        for rows in progress_batches(eval_idx, int(training["batch_size"]), shuffle=False, desc="eval batches", config=config):
            times, coords, marks, targets, target_coords, dtime, mask = make_batch(rows)
            result = model(times, coords, marks, mask)
            task_loss = stpp_joint_nll(result, targets, target_coords, dtime, num_marks=num_marks)
            losses.append(float(task_loss.cpu()))
            if num_marks > 1:
                correct += int((result.logits.argmax(dim=-1) == targets).sum().cpu())
                total += int(targets.numel())
            rmses.extend(torch.sqrt(((result.aux["next_location"] - target_coords) ** 2).sum(dim=-1)).cpu().tolist())
            time_maes.extend(torch.abs(result.aux["next_time"] - dtime).cpu().tolist())
    acc = correct / max(1, total) if total else float("nan")
    rmse_location = float(np.mean(rmses))
    mae_time = float(np.mean(time_maes))
    joint_nll = float(np.mean(losses))
    primary = rmse_location
    return {
        "train_loss": last_train,
        "eval_loss": joint_nll,
        "primary_metric": primary,
        "joint_nll": joint_nll,
        "accuracy": acc,
        "mae_time": mae_time,
        "rmse_location": rmse_location,
        "examples": float(len(windows)),
        **model_audit,
    }


def run_tkg(spec: RunSpec, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    training = config["training"]
    raw_frame = pd.read_csv(spec.dataset_path)
    raw_frame["timestamp"] = pd.to_numeric(raw_frame["timestamp"], errors="coerce").fillna(0.0)
    if "split" in raw_frame.columns:
        raw_frame["split"] = raw_frame["split"].astype(str).str.lower()
    else:
        raw_frame["split"] = "train"
    entity_values = pd.concat([raw_frame["head"], raw_frame["tail"]], ignore_index=True)
    entity_map = {value: idx for idx, value in enumerate(sorted(entity_values.astype(str).unique()))}
    raw_frame["head_id"] = raw_frame["head"].astype(str).map(entity_map).astype("int64")
    raw_frame["tail_id"] = raw_frame["tail"].astype(str).map(entity_map).astype("int64")
    raw_frame["relation_id"], _ = label_encode(raw_frame["relation"])
    raw_frame["timestamp_id"], _ = label_encode(raw_frame["timestamp"])

    max_examples = int(training["max_examples"])
    eval_ratio = float(training["eval_ratio"])
    train_budget = max(1, int(max_examples * (1.0 - eval_ratio)))
    eval_budget = max(1, max_examples - train_budget)
    if "train" in set(raw_frame["split"]):
        train_frame = raw_frame[raw_frame["split"].eq("train")].sort_values("timestamp").tail(train_budget)
        eval_candidates = raw_frame[raw_frame["split"].isin(["test", "val", "valid"])].sort_values("timestamp")
        test_frame = eval_candidates[eval_candidates["split"].eq("test")]
        eval_frame = (test_frame if len(test_frame) else eval_candidates).head(eval_budget)
        frame = pd.concat([train_frame, eval_frame], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        train_idx = frame.index[frame["split"].eq("train")].to_numpy()
        eval_idx = frame.index[~frame["split"].eq("train")].to_numpy()
        if len(train_idx) == 0 or len(eval_idx) == 0:
            frame = chronological_sample(raw_frame, max_examples)
            train_idx, eval_idx = split_indices(len(frame), eval_ratio)
    else:
        frame = chronological_sample(raw_frame, max_examples)
        train_idx, eval_idx = split_indices(len(frame), eval_ratio)
    model = build_benchmark_model(spec, frame, config, device)
    model_audit = audit_warrant_application(model, spec)
    lr_key = "warrant_learning_rate" if spec.use_warrant else "learning_rate"
    learning_rate = float(spec.model_config.get(lr_key, spec.model_config.get("learning_rate", training["learning_rate"])))
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=float(training.get("weight_decay", 0.0)))
    history_len = int(training["history_len"])
    gate_loss_weight = float(spec.model_config.get("gate_loss_weight", config.get("tkg", {}).get("gate_loss_weight", 0.05)))
    gradient_clip = float(spec.model_config.get("gradient_clip", config.get("tkg", {}).get("gradient_clip", 1.0)))
    head_values = frame["head_id"].to_numpy(dtype=np.int64)
    relation_values = frame["relation_id"].to_numpy(dtype=np.int64)
    time_values = frame["timestamp_id"].to_numpy(dtype=np.float32)
    tail_values = frame["tail_id"].to_numpy(dtype=np.int64)
    strict_time = bool(config.get("tkg", {}).get("strict_time_history", False))
    hist_heads, hist_relations, hist_tails, hist_times, hist_masks = build_tkg_histories(
        frame,
        history_len,
        strict_time=strict_time,
    )

    def make_batch(rows: np.ndarray):
        rows = np.asarray(rows, dtype=np.int64)
        return (
            torch.as_tensor(head_values[rows], device=device),
            torch.as_tensor(relation_values[rows], device=device),
            torch.as_tensor(time_values[rows], dtype=torch.float32, device=device),
            torch.as_tensor(hist_heads[rows], device=device),
            torch.as_tensor(hist_relations[rows], device=device),
            torch.as_tensor(hist_tails[rows], device=device),
            torch.as_tensor(hist_times[rows], dtype=torch.float32, device=device),
            torch.as_tensor(hist_masks[rows], dtype=torch.bool, device=device),
            torch.as_tensor(tail_values[rows], device=device),
        )

    last_train = 0.0
    for epoch in epoch_iter(spec, config):
        model.train()
        losses = []
        for rows in progress_batches(
            train_idx,
            int(training["batch_size"]),
            shuffle=True,
            desc=f"epoch {epoch + 1}/{int(training['epochs'])} batches",
            config=config,
        ):
            *inputs, target = make_batch(rows)
            result = model(*inputs)
            loss = F.cross_entropy(result.logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite TKG train loss for {spec.model}/{spec.dataset}/{spec.variant}")
            if "relation_logits" in result.aux:
                loss = loss + 0.05 * F.cross_entropy(result.aux["relation_logits"], inputs[1])
            if spec.use_warrant and gate_loss_weight > 0.0 and "raw_warrant_logits" in result.aux:
                warrant_targets = tkg_warrant_targets(inputs[0], inputs[1], inputs[3], inputs[4], inputs[5], target, inputs[7])
                loss = loss + gate_loss_weight * masked_gate_bce(result.aux["raw_warrant_logits"], warrant_targets, inputs[7])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite TKG total train loss for {spec.model}/{spec.dataset}/{spec.variant}")
            opt.zero_grad()
            loss.backward()
            if gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        last_train = float(np.mean(losses)) if losses else 0.0
        if progress_enabled(config):
            tqdm.write(f"{spec.model}/{spec.variant} epoch {epoch + 1}: train_loss={last_train:.6f}")

    model.eval()
    losses, correct, total, reciprocal_ranks = [], 0, 0, []
    with torch.no_grad():
        for rows in progress_batches(eval_idx, int(training["batch_size"]), shuffle=False, desc="eval batches", config=config):
            *inputs, target = make_batch(rows)
            result = model(*inputs)
            if not torch.isfinite(result.logits).all():
                raise FloatingPointError(f"non-finite TKG eval logits for {spec.model}/{spec.dataset}/{spec.variant}")
            eval_loss = F.cross_entropy(result.logits, target)
            if not torch.isfinite(eval_loss):
                raise FloatingPointError(f"non-finite TKG eval loss for {spec.model}/{spec.dataset}/{spec.variant}")
            losses.append(float(eval_loss.cpu()))
            correct += int((result.logits.argmax(dim=-1) == target).sum().cpu())
            total += int(target.numel())
            target_logits = result.logits.gather(1, target.unsqueeze(1))
            if not torch.isfinite(target_logits).all():
                raise FloatingPointError(f"non-finite TKG target logits for {spec.model}/{spec.dataset}/{spec.variant}")
            ranks = (result.logits > target_logits).sum(dim=1).to(torch.float32) + 1.0
            reciprocal_ranks.extend((1.0 / ranks).cpu().tolist())
    acc = correct / max(1, total)
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan")
    return {
        "train_loss": last_train,
        "eval_loss": float(np.mean(losses)),
        "primary_metric": mrr,
        "accuracy": acc,
        "mrr": mrr,
        "examples": float(len(frame)),
        **model_audit,
    }


def read_jsonl(path: Path, max_examples: int) -> list[dict[str, Any]]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(json.loads(line))
            if len(examples) >= max_examples:
                break
    return examples


def tokenize_text(text: Any, vocab_size: int, max_len: int) -> np.ndarray:
    words = str(text).lower().split()[:max_len]
    ids = [hash_token(word, vocab_size - 1) + 1 for word in words]
    return np.pad(np.asarray(ids, dtype=np.int64), (0, max_len - len(ids)))[:max_len]


def rag_context_to_text(context: Any) -> str:
    if isinstance(context, str):
        return context
    if isinstance(context, dict):
        parts: list[str] = []
        title = context.get("title")
        if isinstance(title, list):
            parts.extend(str(item) for item in title)
        elif title is not None:
            parts.append(str(title))
        sentences = context.get("sentences")
        if isinstance(sentences, list):
            for item in sentences:
                if isinstance(item, list):
                    parts.extend(str(sentence) for sentence in item)
                else:
                    parts.append(str(item))
        elif sentences is not None:
            parts.append(str(sentences))
        if parts:
            return " ".join(parts)
    return str(context)


def rag_contexts_to_passages(contexts: Any, *, max_passages: int) -> list[dict[str, str]]:
    if not isinstance(contexts, list):
        contexts = [contexts] if contexts is not None else []
    passages: list[dict[str, str]] = []
    for context in contexts:
        if isinstance(context, dict):
            titles = context.get("title")
            sentences = context.get("sentences")
            if isinstance(titles, list) and isinstance(sentences, list) and len(titles) == len(sentences):
                for title, passage_sentences in zip(titles, sentences):
                    parts = [str(title)]
                    if isinstance(passage_sentences, list):
                        parts.extend(str(sentence) for sentence in passage_sentences)
                    elif passage_sentences is not None:
                        parts.append(str(passage_sentences))
                    text = " ".join(part for part in parts if part)
                    if text:
                        passages.append({"title": str(title), "text": text})
                    if len(passages) >= max_passages:
                        return passages
                continue
        text = rag_context_to_text(context)
        if text.strip():
            title = context.get("title", "") if isinstance(context, dict) else ""
            passages.append({"title": str(title), "text": text})
        if len(passages) >= max_passages:
            break
    return passages[:max_passages]


def rag_contexts_to_passage_texts(contexts: Any, *, max_passages: int) -> list[str]:
    return [passage["text"] for passage in rag_contexts_to_passages(contexts, max_passages=max_passages)]


def normalize_rag_text(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def rag_support_titles(example: dict[str, Any]) -> set[str]:
    facts = example.get("supporting_facts") or example.get("supporting")
    titles: set[str] = set()
    if isinstance(facts, dict):
        raw_titles = facts.get("title") or facts.get("titles")
        if isinstance(raw_titles, list):
            titles.update(normalize_rag_text(title) for title in raw_titles)
        elif raw_titles is not None:
            titles.add(normalize_rag_text(raw_titles))
    elif isinstance(facts, list):
        for fact in facts:
            if isinstance(fact, (list, tuple)) and fact:
                titles.add(normalize_rag_text(fact[0]))
            elif isinstance(fact, dict) and fact.get("title") is not None:
                titles.add(normalize_rag_text(fact["title"]))
            elif isinstance(fact, str):
                titles.add(normalize_rag_text(fact))
    return {title for title in titles if title}


def rag_support_labels(example: dict[str, Any], passages: list[dict[str, str]], *, max_passages: int) -> np.ndarray:
    labels = np.zeros(max_passages, dtype=np.float32)
    support_titles = rag_support_titles(example)
    if support_titles:
        for idx, passage in enumerate(passages[:max_passages]):
            if normalize_rag_text(passage.get("title", "")) in support_titles:
                labels[idx] = 1.0
        if labels.any():
            return labels

    answers = example.get("answers") or [example.get("answer", "")]
    if not isinstance(answers, list):
        answers = [answers]
    normalized_answers = [normalize_rag_text(answer) for answer in answers if normalize_rag_text(answer)]
    for idx, passage in enumerate(passages[:max_passages]):
        normalized_text = normalize_rag_text(passage.get("text", ""))
        if normalized_text and any(answer and answer in normalized_text for answer in normalized_answers):
            labels[idx] = 1.0
    if labels.any() or not passages:
        return labels

    question_tokens = set(normalize_rag_text(example.get("question", "")).split())
    best_idx, best_score = 0, -1.0
    for idx, passage in enumerate(passages[:max_passages]):
        passage_tokens = set(normalize_rag_text(passage.get("text", "")).split())
        if not passage_tokens:
            continue
        score = len(question_tokens & passage_tokens) / max(1, len(question_tokens))
        if score > best_score:
            best_idx, best_score = idx, score
    if best_score > 0.0:
        labels[best_idx] = 1.0
    return labels


def masked_support_nll(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    usable_examples = mask.any(dim=1) & labels.bool().any(dim=1)
    if not bool(usable_examples.any()):
        return (logits * 0.0).sum()
    selected_logits = logits[usable_examples].masked_fill(~mask[usable_examples], torch.finfo(logits.dtype).min)
    selected_labels = labels[usable_examples] * mask[usable_examples].to(labels.dtype)
    target_dist = selected_labels / selected_labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    log_probs = F.log_softmax(selected_logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1).mean()


def masked_gate_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    usable = mask & labels.bool().any(dim=1, keepdim=True)
    if not bool(usable.any()):
        return (logits * 0.0).sum()
    selected_labels = labels[usable]
    positives = selected_labels.sum().clamp_min(1.0)
    negatives = (1.0 - selected_labels).sum().clamp_min(1.0)
    pos_weight = (negatives / positives).detach()
    return F.binary_cross_entropy_with_logits(logits[usable], selected_labels, pos_weight=pos_weight)


def support_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    warrant_mass: torch.Tensor | None = None,
) -> dict[str, float]:
    usable_examples = mask.any(dim=1) & labels.bool().any(dim=1)
    if not bool(usable_examples.any()):
        return {
            "support_precision": float("nan"),
            "support_recall": float("nan"),
            "support_f1": float("nan"),
            "support_recall_at_1": float("nan"),
            "support_recall_at_2": float("nan"),
            "support_recall_at_5": float("nan"),
            "support_mrr": float("nan"),
            "support_ap": float("nan"),
            "support_auc": float("nan"),
            "support_warrant_mass": float("nan"),
            "distractor_warrant_mass": float("nan"),
            "support_mass_ratio": float("nan"),
        }

    logits = logits.detach().float().cpu()
    labels = labels.detach().float().cpu()
    mask = mask.detach().bool().cpu()
    if warrant_mass is None:
        warrant_mass = logits.exp()
    warrant_mass = warrant_mass.detach().float().cpu()
    recall_hits = {1: 0.0, 2: 0.0, 5: 0.0}
    set_precisions = []
    set_recalls = []
    set_f1s = []
    reciprocal_ranks = []
    average_precisions = []
    auc_labels: list[float] = []
    auc_scores: list[float] = []
    support_masses = []
    distractor_masses = []
    mass_ratios = []
    for row_idx, (row_logits, row_labels, row_mask) in enumerate(zip(logits, labels, mask)):
        if not bool(row_mask.any()) or not bool(row_labels.bool().any()):
            continue
        scores = row_logits.masked_fill(~row_mask, -float("inf"))
        order = torch.argsort(scores, descending=True)
        gold_count = int((row_labels > 0.5).sum())
        pred_count = min(max(1, gold_count), int(row_mask.sum()))
        predicted = order[:pred_count]
        true_positives = float((row_labels[predicted] > 0.5).sum())
        precision = true_positives / max(1, pred_count)
        recall = true_positives / max(1, gold_count)
        set_precisions.append(precision)
        set_recalls.append(recall)
        set_f1s.append(0.0 if precision + recall <= 0.0 else 2.0 * precision * recall / (precision + recall))
        for k in recall_hits:
            topk = order[: min(k, int(row_mask.sum()))]
            recall_hits[k] += float(bool((row_labels[topk] > 0.5).any()))
        positive_ranks = torch.nonzero(row_labels[order] > 0.5, as_tuple=False)
        if positive_ranks.numel():
            reciprocal_ranks.append(1.0 / float(int(positive_ranks[0, 0]) + 1))
            precisions = []
            positives_seen = 0.0
            for rank, idx in enumerate(order.tolist(), start=1):
                if not bool(row_mask[idx]):
                    continue
                if row_labels[idx] > 0.5:
                    positives_seen += 1.0
                    precisions.append(positives_seen / float(rank))
            if precisions:
                average_precisions.append(float(np.mean(precisions)))
        row_mass = warrant_mass[row_idx] if warrant_mass.ndim == 2 else row_logits.exp()
        mass = row_mass * row_mask.to(row_mass.dtype)
        support_mass = float((mass * row_labels).sum())
        distractor_mass = float((mass * (1.0 - row_labels) * row_mask.to(row_labels.dtype)).sum())
        support_masses.append(support_mass)
        distractor_masses.append(distractor_mass)
        mass_ratios.append(support_mass / max(1.0e-8, support_mass + distractor_mass))
        for score, label, valid in zip(row_logits.tolist(), row_labels.tolist(), row_mask.tolist()):
            if valid:
                auc_scores.append(float(score))
                auc_labels.append(float(label))
    examples = int(usable_examples.sum().cpu())
    return {
        "support_precision": float(np.mean(set_precisions)) if set_precisions else float("nan"),
        "support_recall": float(np.mean(set_recalls)) if set_recalls else float("nan"),
        "support_f1": float(np.mean(set_f1s)) if set_f1s else float("nan"),
        "support_recall_at_1": recall_hits[1] / max(1, examples),
        "support_recall_at_2": recall_hits[2] / max(1, examples),
        "support_recall_at_5": recall_hits[5] / max(1, examples),
        "support_mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan"),
        "support_ap": float(np.mean(average_precisions)) if average_precisions else float("nan"),
        "support_auc": binary_auc(auc_labels, auc_scores),
        "support_warrant_mass": float(np.mean(support_masses)) if support_masses else float("nan"),
        "distractor_warrant_mass": float(np.mean(distractor_masses)) if distractor_masses else float("nan"),
        "support_mass_ratio": float(np.mean(mass_ratios)) if mass_ratios else float("nan"),
    }


def run_rag(spec: RunSpec, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    training = config["training"]
    examples = read_jsonl(spec.dataset_path, int(training["max_examples"]))
    if len(examples) < 2:
        raise ValueError("Not enough RAG examples")
    vocab_size = int(spec.model_config.get("vocab_size", 8192))
    train_idx, eval_idx = split_indices(len(examples), float(training["eval_ratio"]))
    frame_stub = {"vocab_size": vocab_size}
    model = build_benchmark_model(spec, frame_stub, config, device)
    model_audit = audit_warrant_application(model, spec)
    opt = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training.get("weight_decay", 0.0)))

    max_passages = int(spec.model_config.get("max_passages", 4))
    question_len = int(spec.model_config.get("question_len", 32))
    passage_len = int(spec.model_config.get("passage_len", 64))
    batch_size = int(spec.model_config.get("batch_size", config.get("rag", {}).get("batch_size", training["batch_size"])))
    eval_batch_size = int(spec.model_config.get("eval_batch_size", config.get("rag", {}).get("eval_batch_size", batch_size)))
    support_loss_weight = float(spec.model_config.get("support_loss_weight", config.get("rag", {}).get("support_loss_weight", 1.0)))
    gate_loss_weight = float(spec.model_config.get("gate_loss_weight", config.get("rag", {}).get("gate_loss_weight", 0.0)))

    question_cache = np.zeros((len(examples), question_len), dtype=np.int64)
    passage_cache = np.zeros((len(examples), max_passages, passage_len), dtype=np.int64)
    mask_cache = np.zeros((len(examples), max_passages), dtype=bool)
    support_cache = np.zeros((len(examples), max_passages), dtype=np.float32)
    for idx, example in enumerate(examples):
        question_cache[idx] = tokenize_text(example.get("question", ""), vocab_size, question_len)
        passage_records = rag_contexts_to_passages(example.get("contexts"), max_passages=max_passages)
        for passage_idx, passage in enumerate(passage_records[:max_passages]):
            tokens = tokenize_text(passage["text"], vocab_size, passage_len)
            passage_cache[idx, passage_idx] = tokens
            mask_cache[idx, passage_idx] = bool(np.any(tokens))
        support_cache[idx] = rag_support_labels(example, passage_records, max_passages=max_passages)

    def make_batch(rows: np.ndarray):
        return (
            torch.as_tensor(question_cache[rows], device=device),
            torch.as_tensor(passage_cache[rows], device=device),
            torch.as_tensor(mask_cache[rows], dtype=torch.bool, device=device),
            torch.as_tensor(support_cache[rows], dtype=torch.float32, device=device),
        )

    last_train = 0.0
    for epoch in epoch_iter(spec, config):
        model.train()
        losses = []
        for rows in progress_batches(
            train_idx,
            batch_size,
            shuffle=True,
            desc=f"epoch {epoch + 1}/{int(training['epochs'])} batches",
            config=config,
        ):
            question, passages, mask, support_target = make_batch(rows)
            result = model(question, passages, mask)
            support_loss = masked_support_nll(result.aux["passage_logits"], support_target, mask)
            loss = support_loss_weight * support_loss
            if spec.use_warrant and gate_loss_weight > 0.0:
                loss = loss + gate_loss_weight * masked_gate_bce(result.aux["raw_warrant_logits"], support_target, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        last_train = float(np.mean(losses)) if losses else 0.0
        if progress_enabled(config):
            tqdm.write(f"{spec.model}/{spec.variant} epoch {epoch + 1}: train_loss={last_train:.6f}")

    model.eval()
    losses = []
    support_metric_rows: list[dict[str, float]] = []
    with torch.no_grad():
        for rows in progress_batches(eval_idx, eval_batch_size, shuffle=False, desc="eval batches", config=config):
            question, passages, mask, support_target = make_batch(rows)
            result = model(question, passages, mask)
            support_loss = masked_support_nll(result.aux["passage_logits"], support_target, mask)
            loss = support_loss_weight * support_loss
            if spec.use_warrant and gate_loss_weight > 0.0:
                loss = loss + gate_loss_weight * masked_gate_bce(result.aux["raw_warrant_logits"], support_target, mask)
            losses.append(float(loss.cpu()))
            support_metric_rows.append(
                support_metrics(
                    result.aux["passage_logits"],
                    support_target,
                    mask,
                    result.aux.get("warrant_mass"),
                )
            )

    def finite_mean(values: list[float]) -> float:
        finite = [value for value in values if not np.isnan(value)]
        return float(np.mean(finite)) if finite else float("nan")

    support_precision = finite_mean([row["support_precision"] for row in support_metric_rows])
    support_recall = finite_mean([row["support_recall"] for row in support_metric_rows])
    support_f1 = finite_mean([row["support_f1"] for row in support_metric_rows])
    support_recall_at_1 = finite_mean([row["support_recall_at_1"] for row in support_metric_rows])
    support_recall_at_2 = finite_mean([row["support_recall_at_2"] for row in support_metric_rows])
    support_recall_at_5 = finite_mean([row["support_recall_at_5"] for row in support_metric_rows])
    support_mrr = finite_mean([row["support_mrr"] for row in support_metric_rows])
    support_ap = finite_mean([row["support_ap"] for row in support_metric_rows])
    support_auc = finite_mean([row["support_auc"] for row in support_metric_rows])
    support_mass = finite_mean([row["support_warrant_mass"] for row in support_metric_rows])
    distractor_mass = finite_mean([row["distractor_warrant_mass"] for row in support_metric_rows])
    support_mass_ratio = finite_mean([row["support_mass_ratio"] for row in support_metric_rows])
    primary = support_mrr
    return {
        "train_loss": last_train,
        "eval_loss": float(np.mean(losses)),
        "primary_metric": primary,
        "accuracy": float("nan"),
        "answer_accuracy": float("nan"),
        "support_precision": support_precision,
        "support_recall": support_recall,
        "support_f1": support_f1,
        "support_recall_at_1": support_recall_at_1,
        "support_recall_at_2": support_recall_at_2,
        "support_recall_at_5": support_recall_at_5,
        "support_mrr": support_mrr,
        "support_ap": support_ap,
        "support_auc": support_auc,
        "support_warrant_mass": support_mass,
        "distractor_warrant_mass": distractor_mass,
        "support_mass_ratio": support_mass_ratio,
        "examples": float(len(examples)),
        **model_audit,
    }


RUNNERS = {
    "ctdg": run_ctdg,
    "mtpp": run_mtpp,
    "stpp": run_stpp,
    "tkg": run_tkg,
    "rag": run_rag,
}


def evaluate_run(spec: RunSpec, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Train/evaluate one already-selected benchmark run and return metrics."""

    if not spec.dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {spec.dataset_path}")
    return RUNNERS[spec.domain](spec, config, device)
