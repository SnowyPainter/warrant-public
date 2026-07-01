#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import (  # noqa: E402
    CTDGHistoryIndex,
    RunSpec,
    audit_warrant_application,
    binary_auc,
    build_benchmark_model,
    build_ctdg_node_histories,
    build_tkg_histories,
    chronological_sample,
    label_encode,
    masked_support_nll,
    prepare_sequence_frame,
    rag_contexts_to_passages,
    rag_support_labels,
    read_jsonl,
    sequence_windows,
    single_sequence_windows,
    split_indices,
    stpp_joint_nll,
    support_metrics,
    tokenize_text,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
VARIANTS = {"base": False, "warrant": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small Warrant neural dissection experiments.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--domain", default=None, help="Optional comma-separated domain filter.")
    parser.add_argument("--model", default=None, help="Optional comma-separated model filter.")
    parser.add_argument("--variant", default=None, help="Optional comma-separated variant filter: base,warrant.")
    parser.add_argument("--force", action="store_true", help="Overwrite completed run folders.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_config(config: dict[str, Any]) -> torch.device:
    name = str(config.get("experiment", {}).get("device", "cuda:0"))
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def progress_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("training", {}).get("progress", True))


def batches(indices: np.ndarray, batch_size: int, *, shuffle: bool, desc: str, config: dict[str, Any]):
    idx = indices.copy()
    if shuffle:
        np.random.shuffle(idx)
    iterator = (idx[start : start + batch_size] for start in range(0, len(idx), batch_size))
    return tqdm(
        iterator,
        total=int(np.ceil(len(idx) / max(1, batch_size))),
        desc=desc,
        unit="batch",
        leave=False,
        disable=not progress_enabled(config),
    )


def finite_mean(values: list[float]) -> float:
    values = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


SCALAR_AUX_KEYS = (
    "warrant_gate_mean",
    "warrant_logit_mean",
    "edge_warrant_gate_mean",
    "edge_warrant_attention_entropy",
    "warrant_tail_mass_mean",
    "copy_gate_mean",
)


def aux_scalar(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        tensor = value.detach().float()
        if tensor.numel() == 0:
            return float("nan")
        return float(tensor.mean().cpu())
    return float(value)


def gate_stats(aux_rows: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    values: dict[str, list[float]] = {key: [] for key in SCALAR_AUX_KEYS}
    for aux in aux_rows:
        for key in SCALAR_AUX_KEYS:
            if key in aux:
                values[key].append(aux_scalar(aux[key]))
    stats = {
        "gate_mean": finite_mean(values["warrant_gate_mean"]),
        "warrant_logit_mean": finite_mean(values["warrant_logit_mean"]),
    }
    for key in SCALAR_AUX_KEYS:
        if key in ("warrant_gate_mean", "warrant_logit_mean"):
            continue
        mean = finite_mean(values[key])
        if np.isfinite(mean):
            stats[key] = mean
    return stats


class DomainRunner:
    primary_name = "primary_metric"
    higher_is_better = True

    def __init__(self, spec: RunSpec, config: dict[str, Any], device: torch.device) -> None:
        self.spec = spec
        self.config = config
        self.training = config["training"]
        self.device = device
        self.batch_size = int(self.training["batch_size"])

    def prepare(self) -> Any:
        raise NotImplementedError

    def train_epoch(self, model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int) -> dict[str, float]:
        raise NotImplementedError

    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        raise NotImplementedError


class CTDGRunner(DomainRunner):
    primary_name = "auc"

    def prepare(self) -> Any:
        frame = chronological_sample(pd.read_csv(self.spec.dataset_path), int(self.training["max_examples"]))
        frame["src"] = pd.to_numeric(frame["src"], errors="coerce").fillna(0).astype("int64")
        frame["dst"] = pd.to_numeric(frame["dst"], errors="coerce").fillna(0).astype("int64")
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(0.0).astype("float32")
        self.frame = frame
        self.train_idx, self.eval_idx = split_indices(len(frame), float(self.training["eval_ratio"]))
        self.history_len = int(self.training["history_len"])
        self.num_nodes = int(max(frame["src"].max(), frame["dst"].max())) + 1
        self.dst_pool = frame["dst"].drop_duplicates().to_numpy(dtype=np.int64)
        self.src_hist_nodes, self.src_hist_times, self.src_hist_mask = build_ctdg_node_histories(
            frame,
            self.history_len,
            self.num_nodes,
        )
        self.history_index = CTDGHistoryIndex(frame, self.history_len, self.num_nodes)
        return frame

    def make_batch(self, rows: np.ndarray, *, negative: bool):
        frame = self.frame
        src = torch.tensor(frame.iloc[rows]["src"].to_numpy(), device=self.device)
        dst_values = frame.iloc[rows]["dst"].to_numpy().copy()
        if negative:
            src_values = frame.iloc[rows]["src"].to_numpy()
            pos_values = frame.iloc[rows]["dst"].to_numpy()
            dst_values = np.random.choice(self.dst_pool, size=len(rows), replace=True)
            bad = (dst_values == src_values) | (dst_values == pos_values)
            while bad.any():
                dst_values[bad] = np.random.choice(self.dst_pool, size=int(bad.sum()), replace=True)
                bad = (dst_values == src_values) | (dst_values == pos_values)
        dst = torch.tensor(dst_values, device=self.device)
        ts = torch.tensor(frame.iloc[rows]["timestamp"].to_numpy(), dtype=torch.float32, device=self.device)
        hist_nodes = torch.tensor(self.src_hist_nodes[rows], device=self.device)
        hist_times = torch.tensor(self.src_hist_times[rows], dtype=torch.float32, device=self.device)
        edge_feats = torch.zeros((len(rows), self.history_len, 1), dtype=torch.float32, device=self.device)
        masks = torch.tensor(self.src_hist_mask[rows], dtype=torch.bool, device=self.device)
        dst_hist_nodes, dst_hist_times, dst_hist_masks = self.history_index.gather(dst_values.astype(np.int64), rows.astype(np.int64))
        return (
            src,
            dst,
            ts,
            hist_nodes,
            hist_times,
            edge_feats,
            masks,
            torch.tensor(dst_hist_nodes, device=self.device),
            torch.tensor(dst_hist_times, dtype=torch.float32, device=self.device),
            torch.zeros((len(rows), self.history_len, 1), dtype=torch.float32, device=self.device),
            torch.tensor(dst_hist_masks, dtype=torch.bool, device=self.device),
        )

    def train_epoch(self, model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int) -> dict[str, float]:
        model.train()
        losses, aux_rows = [], []
        for rows in batches(self.train_idx, self.batch_size, shuffle=True, desc=f"{self.spec.model} {self.spec.variant} train e{epoch}", config=self.config):
            pos = self.make_batch(rows, negative=False)
            neg = self.make_batch(rows, negative=True)
            pos_result = model(*pos)
            neg_result = model(*neg)
            logits = torch.cat([pos_result.logits, neg_result.logits])
            labels = torch.cat([torch.ones_like(pos_result.logits), torch.zeros_like(neg_result.logits)])
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            aux_rows.extend([pos_result.aux, neg_result.aux])
        return {"train_loss": finite_mean(losses), **gate_stats(aux_rows)}

    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        model.eval()
        labels_all, scores_all, losses, aux_rows = [], [], [], []
        with torch.no_grad():
            for rows in batches(self.eval_idx, self.batch_size, shuffle=False, desc=f"{self.spec.model} {self.spec.variant} eval", config=self.config):
                pos = self.make_batch(rows, negative=False)
                neg = self.make_batch(rows, negative=True)
                pos_result = model(*pos)
                neg_result = model(*neg)
                logits = torch.cat([pos_result.logits, neg_result.logits])
                labels = torch.cat([torch.ones_like(pos_result.logits), torch.zeros_like(neg_result.logits)])
                losses.append(float(F.binary_cross_entropy_with_logits(logits, labels).cpu()))
                labels_all.extend(labels.cpu().tolist())
                scores_all.extend(torch.sigmoid(logits).cpu().tolist())
                aux_rows.extend([pos_result.aux, neg_result.aux])
        preds = [1.0 if score >= 0.5 else 0.0 for score in scores_all]
        acc = float(np.mean([pred == label for pred, label in zip(preds, labels_all)])) if labels_all else float("nan")
        auc = binary_auc(labels_all, scores_all)
        return {"eval_loss": finite_mean(losses), "primary_metric": auc, "auc": auc, "accuracy": acc, **gate_stats(aux_rows)}


class MTPPRunner(DomainRunner):
    primary_name = "mark_mrr"

    def prepare(self) -> Any:
        frame = prepare_sequence_frame(self.spec.dataset_path, int(self.training["max_examples"]))
        windows = sequence_windows(frame, int(self.training["history_len"]))
        if len(windows) < 2:
            windows = single_sequence_windows(frame, int(self.training["history_len"]))
        if len(windows) < 2:
            raise ValueError("Not enough MTPP windows")
        self.frame = frame
        self.windows = windows
        self.train_idx, self.eval_idx = split_indices(len(windows), float(self.training["eval_ratio"]))
        return frame

    def make_batch(self, rows: np.ndarray):
        batch = [self.windows[int(row)] for row in rows]
        marks = torch.tensor(np.stack([item[0] for item in batch]), device=self.device)
        times = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32, device=self.device)
        targets = torch.tensor([item[2] for item in batch], device=self.device)
        dtime = torch.tensor([item[3] for item in batch], dtype=torch.float32, device=self.device)
        mask = torch.tensor(
            np.stack([np.arange(marks.shape[1]) >= marks.shape[1] - item[4] for item in batch]),
            dtype=torch.bool,
            device=self.device,
        )
        return marks, times, targets, dtime, mask

    def train_epoch(self, model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int) -> dict[str, float]:
        model.train()
        losses, aux_rows = [], []
        for rows in batches(self.train_idx, self.batch_size, shuffle=True, desc=f"{self.spec.model} {self.spec.variant} train e{epoch}", config=self.config):
            marks, times, targets, dtime, mask = self.make_batch(rows)
            result = model(marks, times, mask)
            loss = F.cross_entropy(result.logits, targets) + 0.01 * F.smooth_l1_loss(result.aux["next_time"], dtime)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            aux_rows.append(result.aux)
        return {"train_loss": finite_mean(losses), **gate_stats(aux_rows)}

    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        model.eval()
        losses, correct, total, rr, maes, aux_rows = [], 0, 0, [], [], []
        with torch.no_grad():
            for rows in batches(self.eval_idx, self.batch_size, shuffle=False, desc=f"{self.spec.model} {self.spec.variant} eval", config=self.config):
                marks, times, targets, dtime, mask = self.make_batch(rows)
                result = model(marks, times, mask)
                loss = F.cross_entropy(result.logits, targets) + 0.01 * F.smooth_l1_loss(result.aux["next_time"], dtime)
                target_logits = result.logits.gather(1, targets.unsqueeze(1))
                ranks = (result.logits > target_logits).sum(dim=1).float() + 1.0
                rr.extend((1.0 / ranks).cpu().tolist())
                correct += int((result.logits.argmax(dim=-1) == targets).sum().cpu())
                total += int(targets.numel())
                maes.extend(torch.abs(result.aux["next_time"] - dtime).cpu().tolist())
                losses.append(float(loss.cpu()))
                aux_rows.append(result.aux)
        mark_mrr = finite_mean(rr)
        return {
            "eval_loss": finite_mean(losses),
            "primary_metric": mark_mrr,
            "mark_mrr": mark_mrr,
            "accuracy": correct / max(1, total),
            "mae_time": finite_mean(maes),
            **gate_stats(aux_rows),
        }


class STPPRunner(DomainRunner):
    primary_name = "rmse_location"
    higher_is_better = False

    def prepare(self) -> Any:
        frame = prepare_sequence_frame(self.spec.dataset_path, int(self.training["max_examples"]))
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(0.0).astype("float64")
        frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce").fillna(0.0).astype("float32")
        frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce").fillna(0.0).astype("float32")
        for column in ("lat", "lon"):
            std = float(frame[column].std() or 1.0)
            frame[column] = (frame[column] - float(frame[column].mean())) / std
        time_origin = float(frame["timestamp"].min())
        time_scale = max(float(frame["timestamp"].std() or 1.0), 1.0)
        frame["time_value"] = ((frame["timestamp"] - time_origin) / time_scale).astype("float32")
        deltas = []
        for _, group in frame.groupby("sequence_id", sort=False):
            ts = group.sort_values("timestamp")["timestamp"].to_numpy(dtype=np.float64)
            if len(ts) > 1:
                deltas.extend(np.log1p(np.maximum(0.0, np.diff(ts))).tolist())
        self.delta_scale = max(float(np.mean(deltas) or 1.0), 1.0)
        self.frame = frame
        self.windows = self.make_windows(frame)
        if len(self.windows) < 2:
            raise ValueError("Not enough STPP windows")
        self.train_idx, self.eval_idx = split_indices(len(self.windows), float(self.training["eval_ratio"]))
        self.num_marks = int(frame["mark_id"].max()) + 1
        return frame

    def make_windows(self, frame: pd.DataFrame):
        windows = []
        history_len = int(self.training["history_len"])
        groups = list(frame.groupby("sequence_id", sort=False))
        if len(groups) == 1 and len(groups[0][1]) < 2:
            groups = [("sequence_0", frame.sort_values("timestamp"))]
        for _, group in groups:
            group = group.sort_values("timestamp")
            marks = group["mark_id"].to_numpy(dtype=np.int64)
            raw_times = group["timestamp"].to_numpy(dtype=np.float64)
            times = group["time_value"].to_numpy(dtype=np.float32)
            coords = group[["lat", "lon"]].to_numpy(dtype=np.float32)
            for idx in range(1, len(group)):
                start = max(0, idx - history_len)
                pad = history_len - (idx - start)
                dtime = np.log1p(max(0.0, raw_times[idx] - raw_times[idx - 1])) / self.delta_scale
                windows.append((np.pad(marks[start:idx], (pad, 0)), np.pad(times[start:idx], (pad, 0)), np.pad(coords[start:idx], ((pad, 0), (0, 0))), int(marks[idx]), coords[idx], float(dtime), idx - start))
        return windows

    def make_batch(self, rows: np.ndarray):
        batch = [self.windows[int(row)] for row in rows]
        marks = torch.tensor(np.stack([item[0] for item in batch]), device=self.device)
        times = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32, device=self.device)
        coords = torch.tensor(np.stack([item[2] for item in batch]), dtype=torch.float32, device=self.device)
        targets = torch.tensor([item[3] for item in batch], device=self.device)
        target_coords = torch.tensor(np.stack([item[4] for item in batch]), dtype=torch.float32, device=self.device)
        dtime = torch.tensor([item[5] for item in batch], dtype=torch.float32, device=self.device)
        mask = torch.tensor(np.stack([np.arange(marks.shape[1]) >= marks.shape[1] - item[6] for item in batch]), dtype=torch.bool, device=self.device)
        return times, coords, marks, targets, target_coords, dtime, mask

    def train_epoch(self, model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int) -> dict[str, float]:
        model.train()
        losses, aux_rows = [], []
        for rows in batches(self.train_idx, self.batch_size, shuffle=True, desc=f"{self.spec.model} {self.spec.variant} train e{epoch}", config=self.config):
            times, coords, marks, targets, target_coords, dtime, mask = self.make_batch(rows)
            result = model(times, coords, marks, mask)
            loss = stpp_joint_nll(result, targets, target_coords, dtime, num_marks=self.num_marks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            aux_rows.append(result.aux)
        return {"train_loss": finite_mean(losses), **gate_stats(aux_rows)}

    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        model.eval()
        losses, correct, total, rmses, maes, aux_rows = [], 0, 0, [], [], []
        with torch.no_grad():
            for rows in batches(self.eval_idx, self.batch_size, shuffle=False, desc=f"{self.spec.model} {self.spec.variant} eval", config=self.config):
                times, coords, marks, targets, target_coords, dtime, mask = self.make_batch(rows)
                result = model(times, coords, marks, mask)
                loss = stpp_joint_nll(result, targets, target_coords, dtime, num_marks=self.num_marks)
                losses.append(float(loss.cpu()))
                if self.num_marks > 1:
                    correct += int((result.logits.argmax(dim=-1) == targets).sum().cpu())
                    total += int(targets.numel())
                rmses.extend(torch.sqrt(((result.aux["next_location"] - target_coords) ** 2).sum(dim=-1)).cpu().tolist())
                maes.extend(torch.abs(result.aux["next_time"] - dtime).cpu().tolist())
                aux_rows.append(result.aux)
        joint_nll = finite_mean(losses)
        rmse_location = finite_mean(rmses)
        return {"eval_loss": joint_nll, "primary_metric": rmse_location, "joint_nll": joint_nll, "accuracy": correct / max(1, total) if total else float("nan"), "rmse_location": rmse_location, "mae_time": finite_mean(maes), **gate_stats(aux_rows)}


class TKGRunner(DomainRunner):
    primary_name = "mrr"

    def prepare(self) -> Any:
        raw = pd.read_csv(self.spec.dataset_path)
        raw["timestamp"] = pd.to_numeric(raw["timestamp"], errors="coerce").fillna(0.0)
        raw["split"] = raw["split"].astype(str).str.lower() if "split" in raw.columns else "train"
        entity_values = pd.concat([raw["head"], raw["tail"]], ignore_index=True)
        entity_map = {value: idx for idx, value in enumerate(sorted(entity_values.astype(str).unique()))}
        raw["head_id"] = raw["head"].astype(str).map(entity_map).astype("int64")
        raw["tail_id"] = raw["tail"].astype(str).map(entity_map).astype("int64")
        raw["relation_id"], _ = label_encode(raw["relation"])
        raw["timestamp_id"], _ = label_encode(raw["timestamp"])
        max_examples = int(self.training["max_examples"])
        train_budget = max(1, int(max_examples * (1.0 - float(self.training["eval_ratio"]))))
        eval_budget = max(1, max_examples - train_budget)
        train = raw[raw["split"].eq("train")].sort_values("timestamp").tail(train_budget)
        eval_candidates = raw[raw["split"].isin(["test", "val", "valid"])].sort_values("timestamp")
        test = eval_candidates[eval_candidates["split"].eq("test")]
        eval_frame = (test if len(test) else eval_candidates).head(eval_budget)
        frame = pd.concat([train, eval_frame], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        self.frame = frame
        self.train_idx = frame.index[frame["split"].eq("train")].to_numpy()
        self.eval_idx = frame.index[~frame["split"].eq("train")].to_numpy()
        self.history_len = int(self.training["history_len"])
        self.head_values = frame["head_id"].to_numpy(dtype=np.int64)
        self.relation_values = frame["relation_id"].to_numpy(dtype=np.int64)
        self.time_values = frame["timestamp_id"].to_numpy(dtype=np.float32)
        self.tail_values = frame["tail_id"].to_numpy(dtype=np.int64)
        strict_time = bool(self.config.get("tkg", {}).get("strict_time_history", False))
        self.hist_heads, self.hist_relations, self.hist_tails, self.hist_times, self.hist_masks = build_tkg_histories(
            frame,
            self.history_len,
            strict_time=strict_time,
        )
        return frame

    def make_batch(self, rows: np.ndarray):
        rows = np.asarray(rows, dtype=np.int64)
        return (
            torch.as_tensor(self.head_values[rows], device=self.device),
            torch.as_tensor(self.relation_values[rows], device=self.device),
            torch.as_tensor(self.time_values[rows], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.hist_heads[rows], device=self.device),
            torch.as_tensor(self.hist_relations[rows], device=self.device),
            torch.as_tensor(self.hist_tails[rows], device=self.device),
            torch.as_tensor(self.hist_times[rows], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.hist_masks[rows], dtype=torch.bool, device=self.device),
            torch.as_tensor(self.tail_values[rows], device=self.device),
        )

    def train_epoch(self, model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int) -> dict[str, float]:
        model.train()
        losses, aux_rows = [], []
        for rows in batches(self.train_idx, self.batch_size, shuffle=True, desc=f"{self.spec.model} {self.spec.variant} train e{epoch}", config=self.config):
            *inputs, target = self.make_batch(rows)
            result = model(*inputs)
            loss = F.cross_entropy(result.logits, target)
            if "relation_logits" in result.aux:
                loss = loss + 0.05 * F.cross_entropy(result.aux["relation_logits"], inputs[1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            aux_rows.append(result.aux)
        return {"train_loss": finite_mean(losses), **gate_stats(aux_rows)}

    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        model.eval()
        losses, correct, total, rr, aux_rows = [], 0, 0, [], []
        with torch.no_grad():
            for rows in batches(self.eval_idx, self.batch_size, shuffle=False, desc=f"{self.spec.model} {self.spec.variant} eval", config=self.config):
                *inputs, target = self.make_batch(rows)
                result = model(*inputs)
                losses.append(float(F.cross_entropy(result.logits, target).cpu()))
                correct += int((result.logits.argmax(dim=-1) == target).sum().cpu())
                total += int(target.numel())
                target_logits = result.logits.gather(1, target.unsqueeze(1))
                ranks = (result.logits > target_logits).sum(dim=1).float() + 1.0
                rr.extend((1.0 / ranks).cpu().tolist())
                aux_rows.append(result.aux)
        mrr = finite_mean(rr)
        return {"eval_loss": finite_mean(losses), "primary_metric": mrr, "mrr": mrr, "accuracy": correct / max(1, total), **gate_stats(aux_rows)}


class RAGRunner(DomainRunner):
    primary_name = "support_mrr"

    def prepare(self) -> Any:
        self.examples = read_jsonl(self.spec.dataset_path, int(self.training["max_examples"]))
        if len(self.examples) < 2:
            raise ValueError("Not enough RAG examples")
        self.vocab_size = int(self.spec.model_config.get("vocab_size", 8192))
        self.train_idx, self.eval_idx = split_indices(len(self.examples), float(self.training["eval_ratio"]))
        self.max_passages = int(self.spec.model_config.get("max_passages", 4))
        self.question_len = int(self.spec.model_config.get("question_len", 32))
        self.passage_len = int(self.spec.model_config.get("passage_len", 64))
        self.batch_size = int(self.spec.model_config.get("batch_size", self.batch_size))
        self.eval_batch_size = int(self.spec.model_config.get("eval_batch_size", self.batch_size))
        return {"vocab_size": self.vocab_size}

    def make_batch(self, rows: np.ndarray):
        qs, ps, masks, support_targets = [], [], [], []
        for row in rows:
            example = self.examples[int(row)]
            qs.append(tokenize_text(example.get("question", ""), self.vocab_size, self.question_len))
            passages = rag_contexts_to_passages(example.get("contexts"), max_passages=self.max_passages)
            arrays = [tokenize_text(p["text"], self.vocab_size, self.passage_len) for p in passages]
            passage_mask = [bool(np.any(p)) for p in arrays]
            while len(arrays) < self.max_passages:
                arrays.append(np.zeros(self.passage_len, dtype=np.int64))
                passage_mask.append(False)
            ps.append(np.stack(arrays[: self.max_passages]))
            masks.append(np.asarray(passage_mask[: self.max_passages], dtype=bool))
            support_targets.append(rag_support_labels(example, passages, max_passages=self.max_passages))
        return (
            torch.tensor(np.stack(qs), device=self.device),
            torch.tensor(np.stack(ps), device=self.device),
            torch.tensor(np.stack(masks), dtype=torch.bool, device=self.device),
            torch.tensor(np.stack(support_targets), dtype=torch.float32, device=self.device),
        )

    def train_epoch(self, model: torch.nn.Module, opt: torch.optim.Optimizer, epoch: int) -> dict[str, float]:
        model.train()
        losses, aux_rows = [], []
        for rows in batches(self.train_idx, self.batch_size, shuffle=True, desc=f"{self.spec.model} {self.spec.variant} train e{epoch}", config=self.config):
            question, passages, mask, target = self.make_batch(rows)
            result = model(question, passages, mask)
            loss = masked_support_nll(result.aux["passage_logits"], target, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            aux_rows.append(result.aux)
        return {"train_loss": finite_mean(losses), **gate_stats(aux_rows)}

    def evaluate(self, model: torch.nn.Module) -> dict[str, float]:
        model.eval()
        losses, metric_rows, attention_rows, aux_rows = [], [], [], []
        with torch.no_grad():
            for rows in batches(self.eval_idx, self.eval_batch_size, shuffle=False, desc=f"{self.spec.model} {self.spec.variant} eval", config=self.config):
                question, passages, mask, target = self.make_batch(rows)
                result = model(question, passages, mask)
                losses.append(float(masked_support_nll(result.aux["passage_logits"], target, mask).cpu()))
                metric_rows.append(support_metrics(result.aux["passage_logits"], target, mask, result.aux.get("warrant_mass")))
                if "attention_mass" in result.aux:
                    attention_rows.append(support_metrics(result.aux["passage_logits"], target, mask, result.aux["attention_mass"]))
                aux_rows.append(result.aux)
        support_mrr = finite_mean([row["support_mrr"] for row in metric_rows])
        metrics = {
            "eval_loss": finite_mean(losses),
            "primary_metric": support_mrr,
            "support_precision": finite_mean([row["support_precision"] for row in metric_rows]),
            "support_recall": finite_mean([row["support_recall"] for row in metric_rows]),
            "support_f1": finite_mean([row["support_f1"] for row in metric_rows]),
            "support_recall_at_1": finite_mean([row["support_recall_at_1"] for row in metric_rows]),
            "support_mrr": support_mrr,
            "support_recall_at_2": finite_mean([row["support_recall_at_2"] for row in metric_rows]),
            "support_recall_at_5": finite_mean([row["support_recall_at_5"] for row in metric_rows]),
            "support_ap": finite_mean([row["support_ap"] for row in metric_rows]),
            "support_auc": finite_mean([row["support_auc"] for row in metric_rows]),
            "support_warrant_mass": finite_mean([row["support_warrant_mass"] for row in metric_rows]),
            "distractor_warrant_mass": finite_mean([row["distractor_warrant_mass"] for row in metric_rows]),
            "support_mass_ratio": finite_mean([row["support_mass_ratio"] for row in metric_rows]),
            **gate_stats(aux_rows),
        }
        if attention_rows:
            metrics.update(
                {
                    "support_attention_mass": finite_mean([row["support_warrant_mass"] for row in attention_rows]),
                    "distractor_attention_mass": finite_mean([row["distractor_warrant_mass"] for row in attention_rows]),
                    "support_attention_ratio": finite_mean([row["support_mass_ratio"] for row in attention_rows]),
                }
            )
        return metrics


RUNNER_BY_DOMAIN: dict[str, type[DomainRunner]] = {
    "ctdg": CTDGRunner,
    "mtpp": MTPPRunner,
    "stpp": STPPRunner,
    "tkg": TKGRunner,
    "rag": RAGRunner,
}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def collect_existing_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.glob("*/*/*/*/metrics_by_epoch.csv")):
        try:
            rows.extend(pd.read_csv(path).to_dict("records"))
        except Exception as exc:
            print(f"warning: failed to read {path}: {exc}")
    return rows


def maybe_plot_metrics(rows: list[dict[str, Any]], out_dir: Path, runner: DomainRunner) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for variant, subset in frame.groupby("variant"):
        axes[0].plot(subset["epoch"], subset["primary_metric"], marker="o", label=variant)
        if "gate_mean" in subset:
            axes[1].plot(subset["epoch"], subset["gate_mean"], marker="o", label=variant)
    axes[0].set_title(f"Primary metric ({runner.primary_name})")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel(runner.primary_name)
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Warrant gate mean")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("gate")
    axes[1].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=220)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, *, float_digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._\n"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.{float_digits}f}")
    columns = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in display.columns) + " |")
    return "\n".join(lines) + "\n"


def metric_direction(domain: str) -> str:
    return "higher" if RUNNER_BY_DOMAIN[str(domain)].higher_is_better else "lower"


def primary_name(domain: str) -> str:
    return RUNNER_BY_DOMAIN[str(domain)].primary_name


def regime_label(final_gate: float, slope: float) -> str:
    if not np.isfinite(final_gate):
        return "not-observed"
    if final_gate >= 0.9:
        base = "open / near-identity"
    elif final_gate <= 0.35:
        base = "suppressive"
    else:
        base = "selective"
    if np.isfinite(slope):
        if slope > 0.03:
            return f"{base}, opening"
        if slope < -0.03:
            return f"{base}, closing"
    return base


def final_delta_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (domain, dataset, model), subset in frame.groupby(["domain", "dataset", "model"]):
        final = subset.sort_values("epoch").groupby("variant", as_index=False).tail(1)
        base = final[final["variant"].eq("base")]
        warrant = final[final["variant"].eq("warrant")]
        if base.empty or warrant.empty:
            continue
        base_row = base.iloc[0]
        warrant_row = warrant.iloc[0]
        direction = metric_direction(str(domain))
        delta = float(warrant_row["primary_metric"]) - float(base_row["primary_metric"])
        improvement = -delta if direction == "lower" else delta
        rows.append(
            {
                "domain": domain,
                "dataset": dataset,
                "model": model,
                "primary_metric": primary_name(str(domain)),
                "direction": direction,
                "base": float(base_row["primary_metric"]),
                "warrant": float(warrant_row["primary_metric"]),
                "delta_warrant_minus_base": delta,
                "improvement": improvement,
            }
        )
    return rows


def gate_regime_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "gate_mean" not in frame.columns:
        return rows
    warrant = frame[frame["variant"].eq("warrant")].copy()
    for (domain, dataset, model), subset in warrant.groupby(["domain", "dataset", "model"]):
        subset = subset.sort_values("epoch")
        gate = pd.to_numeric(subset["gate_mean"], errors="coerce")
        logits = pd.to_numeric(subset.get("warrant_logit_mean", pd.Series(dtype=float)), errors="coerce")
        metric = pd.to_numeric(subset["primary_metric"], errors="coerce")
        final_gate = float(gate.dropna().iloc[-1]) if gate.notna().any() else float("nan")
        first_gate = float(gate.dropna().iloc[0]) if gate.notna().any() else float("nan")
        slope = final_gate - first_gate if np.isfinite(final_gate) and np.isfinite(first_gate) else float("nan")
        final_metric = float(metric.dropna().iloc[-1]) if metric.notna().any() else float("nan")
        final_logit = float(logits.dropna().iloc[-1]) if logits.notna().any() else float("nan")
        rows.append(
            {
                "domain": domain,
                "dataset": dataset,
                "model": model,
                "primary_metric": primary_name(str(domain)),
                "direction": metric_direction(str(domain)),
                "final_primary": final_metric,
                "final_gate_mean": final_gate,
                "gate_slope": slope,
                "final_warrant_logit_mean": final_logit,
                "regime": regime_label(final_gate, slope),
            }
        )
    return rows


def mass_diagnostic_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "support_attention_mass",
        "distractor_attention_mass",
        "support_attention_ratio",
        "support_warrant_mass",
        "distractor_warrant_mass",
        "support_mass_ratio",
        "warrant_tail_mass_mean",
        "edge_warrant_gate_mean",
        "edge_warrant_attention_entropy",
    ]
    available = [column for column in columns if column in frame.columns]
    if not available:
        return []
    rows: list[dict[str, Any]] = []
    final = frame.sort_values("epoch").groupby(["domain", "dataset", "model", "variant"], as_index=False).tail(1)
    for _, item in final.iterrows():
        row: dict[str, Any] = {
            "domain": item["domain"],
            "dataset": item["dataset"],
            "model": item["model"],
            "variant": item["variant"],
        }
        has_value = False
        for column in available:
            value = pd.to_numeric(pd.Series([item.get(column)]), errors="coerce").iloc[0]
            if pd.notna(value):
                row[column] = float(value)
                has_value = True
        if has_value:
            if "support_attention_ratio" in row and "support_mass_ratio" in row:
                row["ratio_gain_warrant_minus_attention"] = row["support_mass_ratio"] - row["support_attention_ratio"]
            rows.append(row)
    return rows


def plot_line_by_domain(frame: pd.DataFrame, column: str, out_path: Path, title: str, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if column not in frame.columns:
        return
    subset = frame[frame["variant"].eq("warrant")].copy()
    subset[column] = pd.to_numeric(subset[column], errors="coerce")
    subset = subset.dropna(subset=[column])
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for (domain, dataset, model), group in subset.groupby(["domain", "dataset", "model"]):
        group = group.sort_values("epoch")
        ax.plot(group["epoch"], group[column], marker="o", label=f"{domain}/{dataset}/{model}")
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_metric_vs_gate(frame: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if "gate_mean" not in frame.columns:
        return
    warrant = frame[frame["variant"].eq("warrant")].copy()
    for (domain, dataset, model), subset in warrant.groupby(["domain", "dataset", "model"]):
        subset = subset.sort_values("epoch")
        metric = pd.to_numeric(subset["primary_metric"], errors="coerce")
        gate = pd.to_numeric(subset["gate_mean"], errors="coerce")
        if metric.notna().sum() < 2 or gate.notna().sum() < 2:
            continue
        fig, ax_metric = plt.subplots(figsize=(7, 4))
        ax_gate = ax_metric.twinx()
        ax_metric.plot(subset["epoch"], metric, marker="o", color="#2563eb", label=primary_name(str(domain)))
        ax_gate.plot(subset["epoch"], gate, marker="s", color="#dc2626", label="gate_mean")
        ax_metric.set_title(f"{domain}/{dataset}/{model}: metric vs Warrant gate")
        ax_metric.set_xlabel("epoch")
        ax_metric.set_ylabel(f"{primary_name(str(domain))} ({metric_direction(str(domain))} is better)", color="#2563eb")
        ax_gate.set_ylabel("gate_mean", color="#dc2626")
        ax_metric.grid(alpha=0.25)
        fig.tight_layout()
        path = out_dir / f"{domain}_{dataset}_{model}_metric_vs_gate.png".replace("/", "_")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220)
        plt.close(fig)


def write_global_diagnostics(rows: list[dict[str, Any]], output_root: Path) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    analysis_dir = output_root / "analysis"
    tables_dir = analysis_dir / "tables"
    plots_dir = analysis_dir / "plots"
    metric_gate_dir = plots_dir / "metric_vs_gate"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    final_delta = pd.DataFrame(final_delta_rows(frame))
    final_delta.to_csv(tables_dir / "final_base_vs_warrant.csv", index=False)
    (tables_dir / "final_base_vs_warrant.md").write_text(markdown_table(final_delta), encoding="utf-8")

    regimes = pd.DataFrame(gate_regime_rows(frame))
    regimes.to_csv(tables_dir / "warrant_gate_regimes.csv", index=False)
    (tables_dir / "warrant_gate_regimes.md").write_text(markdown_table(regimes), encoding="utf-8")

    mass = pd.DataFrame(mass_diagnostic_rows(frame))
    mass.to_csv(tables_dir / "mass_diagnostics.csv", index=False)
    (tables_dir / "mass_diagnostics.md").write_text(markdown_table(mass), encoding="utf-8")

    plot_line_by_domain(frame, "gate_mean", plots_dir / "gate_mean_by_domain.png", "Warrant gate mean over epochs", "gate_mean")
    plot_line_by_domain(frame, "warrant_logit_mean", plots_dir / "warrant_logit_mean_by_domain.png", "Warrant logit mean over epochs", "warrant_logit_mean")
    plot_metric_vs_gate(frame, metric_gate_dir)

    report = [
        "# Neural Dissection Report",
        "",
        "This report is generated from per-epoch dissection runs. It is meant to show whether the Warrant operation changes internal evidence contribution, not only final benchmark scores.",
        "",
        "## Generated Artifacts",
        "",
        "- `analysis/tables/final_base_vs_warrant.csv`: final primary metric delta for each selected domain/model.",
        "- `analysis/tables/warrant_gate_regimes.csv`: final gate level, gate slope, Warrant logit mean, and a coarse regime label.",
        "- `analysis/tables/mass_diagnostics.csv`: available contribution diagnostics such as RAG support/distractor mass and TKG copy-tail mass.",
        "- `analysis/plots/gate_mean_by_domain.png`: Warrant gate trajectory by domain.",
        "- `analysis/plots/warrant_logit_mean_by_domain.png`: Warrant logit trajectory by domain.",
        "- `analysis/plots/metric_vs_gate/*.png`: dual-axis primary metric and gate trajectory for each Warrant run.",
        "",
        "## Final Base vs Warrant",
        "",
        markdown_table(final_delta),
        "## Warrant Gate Regimes",
        "",
        markdown_table(regimes),
        "## Mass Diagnostics",
        "",
        markdown_table(mass),
    ]
    (analysis_dir / "dissection_report.md").write_text("\n".join(report), encoding="utf-8")


def run_one(spec: RunSpec, config: dict[str, Any], device: torch.device, out_dir: Path) -> list[dict[str, Any]]:
    runner_cls = RUNNER_BY_DOMAIN[spec.domain]
    runner = runner_cls(spec, config, device)
    frame_or_stub = runner.prepare()
    model = build_benchmark_model(spec, frame_or_stub, config, device)
    audit = audit_warrant_application(model, spec)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    rows: list[dict[str, Any]] = []
    epochs = int(config["training"]["epochs"])
    iterator = tqdm(range(1, epochs + 1), desc=f"{spec.domain}/{spec.model}/{spec.variant}", unit="epoch", disable=not progress_enabled(config))
    start = time.time()
    for epoch in iterator:
        train_metrics = runner.train_epoch(model, opt, epoch)
        eval_metrics = runner.evaluate(model)
        row = {
            "domain": spec.domain,
            "dataset": spec.dataset,
            "model": spec.model,
            "variant": spec.variant,
            "seed": spec.seed,
            "epoch": epoch,
            "elapsed_sec": round(time.time() - start, 3),
            **audit,
            **{("train_loss" if k == "train_loss" else f"train_{k}"): v for k, v in train_metrics.items()},
            **eval_metrics,
        }
        rows.append(row)
        iterator.set_postfix(primary=f"{row.get('primary_metric', float('nan')):.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "metrics_by_epoch.csv", rows)
    final = rows[-1] if rows else {}
    (out_dir / "final_metrics.json").write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    return rows


def selected(value: str, filt: set[str] | None) -> bool:
    return filt is None or value in filt


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["experiment"].get("seed", 7))
    set_seed(seed)
    device = device_from_config(config)
    output_root = REPO_ROOT / str(config["experiment"].get("output_root", "experiments/neural_dissection/outputs"))
    domain_filter = set(args.domain.split(",")) if args.domain else None
    model_filter = set(args.model.split(",")) if args.model else None
    variant_filter = set(args.variant.split(",")) if args.variant else None

    all_rows: list[dict[str, Any]] = []
    for run in config["runs"]:
        domain = str(run["domain"])
        model_name = str(run["model"])
        if not selected(domain, domain_filter) or not selected(model_name, model_filter):
            continue
        for variant, use_warrant in VARIANTS.items():
            if not selected(variant, variant_filter):
                continue
            out_dir = output_root / domain / str(run["dataset"]) / model_name / variant
            if out_dir.joinpath("final_metrics.json").exists() and not args.force:
                print(f"skip completed: {out_dir}")
                existing = pd.read_csv(out_dir / "metrics_by_epoch.csv").to_dict("records")
                all_rows.extend(existing)
                continue
            spec = RunSpec(
                domain=domain,
                dataset=str(run["dataset"]),
                dataset_path=REPO_ROOT / str(run["path"]),
                model=model_name,
                variant=variant,
                seed=seed,
                implementation=str(config["training"].get("implementation", "reference")),
                model_config=dict(config["models"][domain][model_name]),
                use_warrant=use_warrant,
            )
            set_seed(seed)
            rows = run_one(spec, config, device, out_dir)
            all_rows.extend(rows)

    aggregate_rows = collect_existing_rows(output_root)
    write_rows(output_root / "aggregate_metrics.csv", aggregate_rows)
    if aggregate_rows:
        frame = pd.DataFrame(aggregate_rows)
        for (domain, dataset, model), subset in frame.groupby(["domain", "dataset", "model"]):
            out_dir = output_root / str(domain) / str(dataset) / str(model)
            runner = RUNNER_BY_DOMAIN[str(domain)](
                RunSpec(str(domain), str(dataset), Path("."), str(model), "warrant", seed, "reference", {}, True),
                config,
                device,
            )
            maybe_plot_metrics(subset.to_dict("records"), out_dir, runner)
        write_global_diagnostics(aggregate_rows, output_root)
    print(f"wrote neural dissection outputs: {output_root}")


if __name__ == "__main__":
    main()
