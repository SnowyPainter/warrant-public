#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import (  # noqa: E402
    RunSpec,
    build_benchmark_model,
    masked_support_nll,
    rag_contexts_to_passages,
    rag_support_labels,
    read_jsonl,
    split_indices,
    support_metrics,
    tokenize_text,
)
from plots.attention_permission import plot_matrix_comparison  # noqa: E402
from plots.common import save_figure  # noqa: E402


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small RAG Warrant model and export real attention/permission case studies.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true", help="Overwrite checkpoint and case-study outputs.")
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


def batch_iter(indices: np.ndarray, batch_size: int, *, shuffle: bool):
    rows = indices.copy()
    if shuffle:
        np.random.shuffle(rows)
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def prepare_batch(
    examples: list[dict[str, Any]],
    rows: np.ndarray,
    *,
    vocab_size: int,
    max_passages: int,
    question_len: int,
    passage_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    questions, passages, masks, labels = [], [], [], []
    for row in rows:
        example = examples[int(row)]
        passage_items = rag_contexts_to_passages(example.get("contexts"), max_passages=max_passages)
        question_ids = tokenize_text(example.get("question", ""), vocab_size, question_len)
        passage_ids = [tokenize_text(item["text"], vocab_size, passage_len) for item in passage_items]
        passage_mask = [bool(np.any(item)) for item in passage_ids]
        while len(passage_ids) < max_passages:
            passage_ids.append(np.zeros(passage_len, dtype=np.int64))
            passage_mask.append(False)
        questions.append(question_ids)
        passages.append(np.stack(passage_ids[:max_passages]))
        masks.append(np.asarray(passage_mask[:max_passages], dtype=bool))
        labels.append(rag_support_labels(example, passage_items, max_passages=max_passages))
    return (
        torch.tensor(np.stack(questions), device=device),
        torch.tensor(np.stack(passages), device=device),
        torch.tensor(np.stack(masks), dtype=torch.bool, device=device),
        torch.tensor(np.stack(labels), dtype=torch.float32, device=device),
    )


def passage_table(example: dict[str, Any], *, max_passages: int) -> list[dict[str, str]]:
    passages = rag_contexts_to_passages(example.get("contexts"), max_passages=max_passages)
    while len(passages) < max_passages:
        passages.append({"title": "", "text": ""})
    return passages[:max_passages]


def finite_mean(values: list[float]) -> float:
    values = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def train_or_load(
    config: dict[str, Any],
    examples: list[dict[str, Any]],
    train_idx: np.ndarray,
    device: torch.device,
    output_root: Path,
    *,
    force: bool,
) -> torch.nn.Module:
    model_cfg = dict(config["model"])
    model_name = str(model_cfg.pop("name"))
    implementation = str(model_cfg.pop("implementation", "reference"))
    seed = int(config["experiment"]["seed"])
    spec = RunSpec(
        domain="rag",
        dataset=str(config["dataset"]["name"]),
        dataset_path=REPO_ROOT / str(config["dataset"]["path"]),
        model=model_name,
        variant="warrant",
        seed=seed,
        implementation=implementation,
        model_config=model_cfg,
        use_warrant=True,
    )
    model = build_benchmark_model(spec, {"vocab_size": int(model_cfg["vocab_size"])}, {"training": config["training"], "warrant": model_cfg}, device)
    checkpoint_path = output_root / "checkpoint.pt"
    if checkpoint_path.exists() and not force:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model"])
        return model

    training = config["training"]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    batch_size = int(training["batch_size"])
    max_passages = int(model_cfg["max_passages"])
    question_len = int(model_cfg["question_len"])
    passage_len = int(model_cfg["passage_len"])
    vocab_size = int(model_cfg["vocab_size"])
    progress = bool(training.get("progress", True))
    for epoch in tqdm(range(1, int(training["epochs"]) + 1), desc="case-study train", unit="epoch", disable=not progress):
        model.train()
        losses: list[float] = []
        iterator = tqdm(
            batch_iter(train_idx, batch_size, shuffle=True),
            total=int(np.ceil(len(train_idx) / max(1, batch_size))),
            desc=f"epoch {epoch}",
            unit="batch",
            leave=False,
            disable=not progress,
        )
        for rows in iterator:
            question, passage_ids, mask, labels = prepare_batch(
                examples,
                rows,
                vocab_size=vocab_size,
                max_passages=max_passages,
                question_len=question_len,
                passage_len=passage_len,
                device=device,
            )
            result = model(question, passage_ids, mask)
            loss = masked_support_nll(result.aux["passage_logits"], labels, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            iterator.set_postfix(loss=finite_mean(losses))
    if bool(training.get("save_checkpoint", True)):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": config}, checkpoint_path)
    return model


def rank_of_first_support(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> int:
    valid_logits = np.where(mask, logits, -np.inf)
    order = np.argsort(-valid_logits)
    for rank, idx in enumerate(order, start=1):
        if labels[idx] > 0.5:
            return rank
    return 999


def select_case(
    model: torch.nn.Module,
    config: dict[str, Any],
    examples: list[dict[str, Any]],
    eval_idx: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    model_cfg = config["model"]
    max_passages = int(model_cfg["max_passages"])
    batch_size = int(config["training"].get("eval_batch_size", config["training"]["batch_size"]))
    vocab_size = int(model_cfg["vocab_size"])
    question_len = int(model_cfg["question_len"])
    passage_len = int(model_cfg["passage_len"])
    best: dict[str, Any] | None = None
    model.eval()
    with torch.no_grad():
        for rows in tqdm(
            batch_iter(eval_idx, batch_size, shuffle=False),
            total=int(np.ceil(len(eval_idx) / max(1, batch_size))),
            desc="case-study scan",
            unit="batch",
            disable=not bool(config["training"].get("progress", True)),
        ):
            question, passage_ids, mask, labels = prepare_batch(
                examples,
                rows,
                vocab_size=vocab_size,
                max_passages=max_passages,
                question_len=question_len,
                passage_len=passage_len,
                device=device,
            )
            result = model(question, passage_ids, mask)
            logits = result.aux["passage_logits"].detach().float().cpu().numpy()
            attention = result.aux["attention_mass"].detach().float().cpu().numpy()
            warrant_mass = result.aux["warrant_mass"].detach().float().cpu().numpy()
            labels_np = labels.detach().float().cpu().numpy()
            mask_np = mask.detach().bool().cpu().numpy()
            gate = np.divide(warrant_mass, np.maximum(attention, 1.0e-8))
            for local_idx, row in enumerate(rows):
                valid = mask_np[local_idx]
                support = labels_np[local_idx] > 0.5
                if not valid.any() or not support.any() or not ((~support) & valid).any():
                    continue
                attn = attention[local_idx]
                perm = gate[local_idx]
                mass = warrant_mass[local_idx]
                support_ratio = float(mass[support].sum() / max(1.0e-8, mass[valid].sum()))
                attention_support_ratio = float(attn[support].sum() / max(1.0e-8, attn[valid].sum()))
                high_attention_distractor = float((attn[(~support) & valid] * (1.0 - perm[(~support) & valid])).max())
                rank = rank_of_first_support(logits[local_idx], labels_np[local_idx], valid)
                score = (
                    3.0 * (support_ratio - attention_support_ratio)
                    + high_attention_distractor
                    + (1.0 / max(1, rank))
                )
                candidate = {
                    "row": int(row),
                    "local_idx": int(local_idx),
                    "score": float(score),
                    "rank": int(rank),
                    "attention": attn,
                    "permission": perm,
                    "mass": mass,
                    "logits": logits[local_idx],
                    "labels": labels_np[local_idx],
                    "mask": valid,
                    "support_ratio": support_ratio,
                    "attention_support_ratio": attention_support_ratio,
                    "high_attention_distractor": high_attention_distractor,
                }
                if best is None or candidate["score"] > best["score"]:
                    best = candidate
    if best is None:
        raise RuntimeError("No usable HotpotQA case with both support and distractor passages was found.")
    return best


def collect_cases(
    model: torch.nn.Module,
    config: dict[str, Any],
    examples: list[dict[str, Any]],
    eval_idx: np.ndarray,
    device: torch.device,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    model_cfg = config["model"]
    max_passages = int(model_cfg["max_passages"])
    batch_size = int(config["training"].get("eval_batch_size", config["training"]["batch_size"]))
    vocab_size = int(model_cfg["vocab_size"])
    question_len = int(model_cfg["question_len"])
    passage_len = int(model_cfg["passage_len"])
    candidates: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for rows in tqdm(
            batch_iter(eval_idx, batch_size, shuffle=False),
            total=int(np.ceil(len(eval_idx) / max(1, batch_size))),
            desc="case-map scan",
            unit="batch",
            disable=not bool(config["training"].get("progress", True)),
        ):
            question, passage_ids, mask, labels = prepare_batch(
                examples,
                rows,
                vocab_size=vocab_size,
                max_passages=max_passages,
                question_len=question_len,
                passage_len=passage_len,
                device=device,
            )
            result = model(question, passage_ids, mask)
            logits = result.aux["passage_logits"].detach().float().cpu().numpy()
            attention = result.aux["attention_mass"].detach().float().cpu().numpy()
            warrant_mass = result.aux["warrant_mass"].detach().float().cpu().numpy()
            labels_np = labels.detach().float().cpu().numpy()
            mask_np = mask.detach().bool().cpu().numpy()
            gate = np.divide(warrant_mass, np.maximum(attention, 1.0e-8))
            for local_idx, row in enumerate(rows):
                valid = mask_np[local_idx]
                support = labels_np[local_idx] > 0.5
                if not valid.any() or not support.any() or not ((~support) & valid).any():
                    continue
                attn = attention[local_idx]
                perm = gate[local_idx]
                mass = warrant_mass[local_idx]
                support_ratio = float(mass[support].sum() / max(1.0e-8, mass[valid].sum()))
                attention_support_ratio = float(attn[support].sum() / max(1.0e-8, attn[valid].sum()))
                high_attention_distractor = float((attn[(~support) & valid] * (1.0 - perm[(~support) & valid])).max())
                rank = rank_of_first_support(logits[local_idx], labels_np[local_idx], valid)
                score = (
                    3.0 * (support_ratio - attention_support_ratio)
                    + high_attention_distractor
                    + (1.0 / max(1, rank))
                )
                candidates.append(
                    {
                        "row": int(row),
                        "score": float(score),
                        "rank": int(rank),
                        "attention": attn,
                        "permission": perm,
                        "mass": mass,
                        "logits": logits[local_idx],
                        "labels": labels_np[local_idx],
                        "mask": valid,
                        "support_ratio": support_ratio,
                        "attention_support_ratio": attention_support_ratio,
                    }
                )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[: max(1, int(limit))]


def write_map_outputs(config: dict[str, Any], examples: list[dict[str, Any]], cases: list[dict[str, Any]], output_root: Path) -> None:
    if not cases:
        return
    max_passages = int(config["model"]["max_passages"])
    attention = np.stack([case["attention"][:max_passages] for case in cases])
    permission = np.stack([np.clip(case["permission"][:max_passages], 0.0, 1.0) for case in cases])
    mass = np.stack([case["mass"][:max_passages] for case in cases])
    labels = np.stack([case["labels"][:max_passages] for case in cases])
    mask = np.stack([case["mask"][:max_passages] for case in cases])
    rows = np.asarray([case["row"] for case in cases], dtype=np.int64)
    matrix_path = output_root / "attention_permission_cases.npz"
    np.savez(
        matrix_path,
        attention=attention,
        permission=permission,
        effective_mass=mass,
        labels=labels,
        mask=mask,
        rows=rows,
    )
    row_labels = [f"Q{idx + 1}" for idx in range(len(cases))]
    column_labels = [f"P{idx + 1}" for idx in range(max_passages)]
    fig = plot_matrix_comparison(
        attention,
        permission,
        title="",
        row_labels=row_labels,
        column_labels=column_labels,
        support_labels=labels,
    )
    png_path, pdf_path = save_figure(fig, "attention_permission_hotpotqa_map", output_dir=output_root)
    lines = [
        "# Attention-Permission Map Case Set",
        "",
        f"- Matrix: `{matrix_path}`",
        f"- Figure PNG: `{png_path}`",
        f"- Figure PDF: `{pdf_path}`",
        "",
        "| Row | Example row | First support rank | Support attention ratio | Support mass ratio | Question |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for idx, case in enumerate(cases):
        question = str(examples[int(case["row"])].get("question", "")).replace("|", "\\|")
        lines.append(
            f"| Q{idx + 1} | {int(case['row'])} | {int(case['rank'])} | "
            f"{float(case['attention_support_ratio']):.4f} | {float(case['support_ratio']):.4f} | {question[:120]} |"
        )
    (output_root / "case_map_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_case_outputs(config: dict[str, Any], examples: list[dict[str, Any]], case: dict[str, Any], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    example = examples[int(case["row"])]
    max_passages = int(config["model"]["max_passages"])
    passages = passage_table(example, max_passages=max_passages)
    valid = case["mask"].astype(bool)
    attention = case["attention"][:max_passages]
    permission = np.clip(case["permission"][:max_passages], 0.0, 1.0)
    mass = case["mass"][:max_passages]
    labels = case["labels"][:max_passages]
    logits = case["logits"][:max_passages]
    column_labels = [
        f"P{idx + 1}{'*' if labels[idx] > 0.5 else ''}" if valid[idx] else f"P{idx + 1}"
        for idx in range(max_passages)
    ]
    matrix_path = output_root / "attention_permission_case.npz"
    np.savez(
        matrix_path,
        attention=attention.reshape(1, -1),
        permission=permission.reshape(1, -1),
        effective_mass=mass.reshape(1, -1),
        labels=labels.reshape(1, -1),
        logits=logits.reshape(1, -1),
        mask=valid.reshape(1, -1),
    )
    with (output_root / "case.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": config["dataset"]["name"],
                "model": config["model"]["name"],
                "row": int(case["row"]),
                "question": example.get("question", ""),
                "answers": example.get("answers") or [example.get("answer", "")],
                "rank_of_first_support": int(case["rank"]),
                "support_attention_ratio": float(case["attention_support_ratio"]),
                "support_mass_ratio": float(case["support_ratio"]),
                "passages": [
                    {
                        "index": idx + 1,
                        "title": passages[idx].get("title", ""),
                        "is_support": bool(labels[idx] > 0.5),
                        "attention": float(attention[idx]),
                        "permission": float(permission[idx]),
                        "effective_mass": float(mass[idx]),
                        "support_logit": float(logits[idx]),
                        "snippet": passages[idx].get("text", "")[:360],
                    }
                    for idx in range(max_passages)
                    if valid[idx]
                ],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    fig = plot_matrix_comparison(
        attention.reshape(1, -1),
        permission.reshape(1, -1),
        title="HotpotQA supporting passage selection",
        row_labels=["Question"],
        column_labels=column_labels,
    )
    png_path, pdf_path = save_figure(fig, "attention_permission_hotpotqa_case", output_dir=output_root)
    report_lines = [
        "# Attention-Permission Case Study",
        "",
        f"- Dataset: `{config['dataset']['name']}`",
        f"- Model: `{config['model']['name']}` with Warrant",
        f"- Example row: `{int(case['row'])}`",
        f"- Rank of first supporting passage: `{int(case['rank'])}`",
        f"- Support attention ratio: `{float(case['attention_support_ratio']):.4f}`",
        f"- Support warranted-mass ratio: `{float(case['support_ratio']):.4f}`",
        f"- Matrix: `{matrix_path}`",
        f"- Figure PNG: `{png_path}`",
        f"- Figure PDF: `{pdf_path}`",
        "",
        "## Question",
        "",
        str(example.get("question", "")),
        "",
        "## Passage-Level Values",
        "",
        "| Passage | Support | Attention | Permission | Effective mass | Logit | Title |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for idx in range(max_passages):
        if not valid[idx]:
            continue
        report_lines.append(
            f"| P{idx + 1} | {bool(labels[idx] > 0.5)} | {attention[idx]:.4f} | {permission[idx]:.4f} | "
            f"{mass[idx]:.4f} | {logits[idx]:.4f} | {passages[idx].get('title', '')} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "별표가 붙은 passage는 HotpotQA supporting fact title과 일치하는 supporting evidence passage다. "
            "그림은 raw attention, Warrant permission, 그리고 두 값을 곱한 effective mass를 같은 실제 샘플에서 비교한다. "
            "따라서 이 case study는 Warrant가 attention relevance를 단순히 다시 그리는 것이 아니라, "
            "support passage ranking score로 들어가는 weighted value term의 permission을 별도로 조절한다는 정성적 증거로 사용된다.",
        ]
    )
    (output_root / "case_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    device = resolve_device(str(config["experiment"].get("device", "auto")))
    output_root = REPO_ROOT / str(config["experiment"]["output_root"])
    examples = read_jsonl(REPO_ROOT / str(config["dataset"]["path"]), int(config["training"]["max_examples"]))
    train_idx, eval_idx = split_indices(len(examples), float(config["training"]["eval_ratio"]))
    model = train_or_load(config, examples, train_idx, device, output_root, force=args.force)
    case = select_case(model, config, examples, eval_idx, device)
    write_case_outputs(config, examples, case, output_root)
    cases = collect_cases(
        model,
        config,
        examples,
        eval_idx,
        device,
        limit=int(config.get("selection", {}).get("num_map_cases", 12)),
    )
    write_map_outputs(config, examples, cases, output_root)
    print(f"wrote case study to {output_root}")


if __name__ == "__main__":
    main()
