#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate import build_benchmark_model
from experiments.neural_dissection.run import set_seed
from experiments.path_localization.common import build_spec, configure_variant, resolve_device
from experiments.path_localization.run import CONFIGURE_BY_DOMAIN, RUNNER_BY_DOMAIN

DEFAULT_CONFIG = Path(__file__).with_name("frozen_controls.yaml")


def train_epochs(runner, model, optimizer, epochs: int, *, description: str, progress: bool, eval_every: int) -> list[dict]:
    rows = []
    last_eval = {}
    for epoch in tqdm(range(1, epochs + 1), desc=description, unit="epoch", disable=not progress):
        train = runner.train_epoch(model, optimizer, epoch)
        if epoch == epochs or epoch % eval_every == 0:
            last_eval = runner.evaluate(model)
            rows.append({"epoch": epoch, **train, **last_eval})
        else:
            rows.append({"epoch": epoch, **train})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def permission_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(False)
        lowered = name.lower()
        if "warrant" in lowered and any(token in lowered for token in ("query_proj", "key_proj", "scorer")):
            parameter.requires_grad_(True)
            selected.append(parameter)
    if not selected:
        raise RuntimeError("No permission/adapter scorer parameters were selected")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--domain", default="all")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = resolve_device(config)
    domains = list(config["tasks"]) if args.domain == "all" else [x.strip() for x in args.domain.split(",")]
    seeds = [int(x) for x in config["experiment"]["seeds"]]
    epochs = int(config["training"]["epochs"])
    lr = float(config["training"]["learning_rate"])
    progress = bool(config["training"].get("progress", True))
    eval_every = max(1, int(config["training"].get("eval_every", 1)))
    root = ROOT / config["experiment"]["output_root"]

    for domain in domains:
        for seed in seeds:
            set_seed(seed)
            open_spec = build_spec(config, domain, "open_path_no_gate", seed)
            runner = RUNNER_BY_DOMAIN[domain](open_spec, config, device)
            frame = runner.prepare()
            open_model = build_benchmark_model(open_spec, frame, config, device)
            configure_variant(open_model, open_spec)
            CONFIGURE_BY_DOMAIN[domain](open_model, open_spec.variant)
            open_dir = root / domain / open_spec.dataset / open_spec.model / "open_path_pretrain" / f"seed_{seed}"
            checkpoint = open_dir / "model_state.pt"
            if checkpoint.exists():
                open_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            else:
                optimizer = torch.optim.AdamW(open_model.parameters(), lr=lr, weight_decay=float(config["training"].get("weight_decay", 0.0)))
                rows = train_epochs(runner, open_model, optimizer, epochs, description=f"{domain} OpenPath seed={seed}", progress=progress, eval_every=eval_every)
                open_dir.mkdir(parents=True, exist_ok=True)
                torch.save(open_model.state_dict(), checkpoint)
                write_csv(open_dir / "metrics_by_epoch.csv", rows)
                (open_dir / "final_metrics.json").write_text(json.dumps(rows[-1], indent=2), encoding="utf-8")

            pretrained = open_model.state_dict()
            for label, target_variant in (("frozen_gate", "correct_path_warrant"), ("frozen_attention_adapter", "attention_adapter")):
                out_dir = root / domain / open_spec.dataset / open_spec.model / label / f"seed_{seed}"
                if (out_dir / "final_metrics.json").exists():
                    continue
                set_seed(seed)
                spec = build_spec(config, domain, target_variant, seed)
                model = build_benchmark_model(spec, frame, config, device)
                # Load the identical OpenPath-trained state before changing the
                # forward semantics; monkey-patched controls preserve state keys.
                model.load_state_dict(pretrained, strict=True)
                configure_variant(model, spec)
                CONFIGURE_BY_DOMAIN[domain](model, target_variant)
                trainable = permission_parameters(model)
                optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
                rows = train_epochs(runner, model, optimizer, epochs, description=f"{domain} {label} seed={seed}", progress=progress, eval_every=eval_every)
                final = {
                    **rows[-1], "domain": domain, "dataset": spec.dataset,
                    "model": spec.model, "variant": label, "seed": seed,
                    "trainable_parameters": sum(p.numel() for p in trainable),
                    "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
                }
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out_dir / "model_state.pt")
                write_csv(out_dir / "metrics_by_epoch.csv", rows)
                (out_dir / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
                print(domain, seed, label, final.get("primary_metric"), flush=True)


if __name__ == "__main__":
    main()
