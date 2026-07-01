#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.bert_hotpotqa_warrant.run import run_one, selected  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run novelty-control HotpotQA Warrant package.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", default=None, help="Variant or comma list. Defaults to config variants.")
    parser.add_argument("--seed", default=None, help="Seed or comma list. Defaults to config seeds.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    variants = selected(args.variant, list(config["variants"]))
    seeds = [int(seed) for seed in selected(args.seed, list(config["experiment"].get("seeds", [7])))]
    for variant in variants:
        for seed in seeds:
            row = run_one(config, variant, seed, force=args.force, dry_run=args.dry_run)
            if row:
                print(f"{variant} seed={seed}: support_mrr={row.get('support_mrr')}")


if __name__ == "__main__":
    main()
