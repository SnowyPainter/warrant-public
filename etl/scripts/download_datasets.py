#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw"
DEFAULT_EXTERNAL_ROOT = REPO_ROOT / "external"


@dataclass(frozen=True)
class Source:
    kind: str
    url: str
    name: str
    patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetSpec:
    domain: str
    name: str
    sources: tuple[Source, ...]
    note: str = ""


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "ctdg",
        "wikipedia",
        (Source("url", "https://snap.stanford.edu/jodie/wikipedia.csv", "wikipedia.csv"),),
    ),
    DatasetSpec(
        "ctdg",
        "mooc",
        (Source("url", "https://snap.stanford.edu/jodie/mooc.csv", "mooc.csv"),),
    ),
    DatasetSpec(
        "ctdg",
        "lastfm",
        (Source("url", "https://snap.stanford.edu/jodie/lastfm.csv", "lastfm.csv"),),
    ),
    DatasetSpec("ctdg", "starcraft", (), "Destructive CTDG setting; place raw files manually if not public."),
    DatasetSpec("ctdg", "lanl", (), "LANL often requires separate access; place raw files manually."),
    DatasetSpec("ctdg", "actmooc", (), "ACTMOOC source varies; place raw files manually."),
    DatasetSpec(
        "mtpp",
        "stackoverflow",
        (
            Source(
                "huggingface",
                "easytpp/stackoverflow",
                "easytpp_stackoverflow",
            ),
        ),
    ),
    DatasetSpec(
        "mtpp",
        "retweets",
        (
            Source(
                "huggingface",
                "easytpp/retweet",
                "easytpp_retweet",
            ),
        ),
    ),
    DatasetSpec(
        "stpp",
        "earthquake",
        (
            Source(
                "git",
                "https://github.com/ss15859/EarthquakeNPP.git",
                "EarthquakeNPP",
                (
                    "Datasets/ComCat/ComCat_catalog.csv",
                    "Datasets/QTM/SanJac_catalog.csv",
                    "Datasets/QTM/SaltonSea_catalog.csv",
                    "Datasets/SCEDC/SCEDC_catalog.csv",
                    "Datasets/WHITE/WHITE_catalog.csv",
                ),
            ),
        ),
    ),
    DatasetSpec(
        "stpp",
        "gowalla",
        (
            Source(
                "url",
                "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz",
                "loc-gowalla_totalCheckins.txt.gz",
            ),
        ),
    ),
    DatasetSpec(
        "tkg",
        "icews14",
        (Source("git", "https://github.com/INK-USC/RE-Net.git", "RE-Net", ("data/ICEWS14/*",)),),
    ),
    DatasetSpec(
        "tkg",
        "icews18",
        (Source("git", "https://github.com/INK-USC/RE-Net.git", "RE-Net", ("data/ICEWS18/*",)),),
    ),
    DatasetSpec(
        "tkg",
        "gdelt",
        (Source("git", "https://github.com/INK-USC/RE-Net.git", "RE-Net", ("data/GDELT/*",)),),
    ),
    DatasetSpec(
        "rag",
        "natural_questions",
        (
            Source(
                "git",
                "https://github.com/google-research-datasets/natural-questions.git",
                "natural-questions",
                ("**/*.jsonl*", "**/*.json*", "**/*.gz"),
            ),
        ),
        "Natural Questions is large; official download may require the upstream instructions.",
    ),
    DatasetSpec(
        "rag",
        "hotpotqa",
        (
            Source(
                "huggingface",
                "hotpotqa/hotpot_qa",
                "hotpotqa_hotpot_qa",
            ),
        ),
        "Uses HuggingFace datasets when installed; otherwise prints the manual source.",
    ),
)


def run(cmd: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    print("+", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def clone_or_pull(source: Source, external_root: Path, *, dry_run: bool) -> Path:
    target = external_root / source.name
    if target.exists():
        run(["git", "-C", str(target), "pull", "--ff-only"], dry_run=dry_run)
    else:
        run(["git", "clone", "--depth", "1", source.url, str(target)], dry_run=dry_run)
    return target


def download_url(source: Source, raw_dir: Path, *, dry_run: bool) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / source.name
    if target.exists():
        print(f"exists: {target}")
        return 1
    print(f"download: {source.url} -> {target}")
    if not dry_run:
        urllib.request.urlretrieve(source.url, target)
    return 1


def copy_matching_files(source_dir: Path, raw_dir: Path, patterns: tuple[str, ...], *, dry_run: bool) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    seen: set[Path] = set()
    for pattern in patterns or ("**/*",):
        for path in source_dir.glob(pattern):
            if path in seen or not path.is_file():
                continue
            if any(part.startswith(".git") for part in path.parts):
                continue
            rel = path.relative_to(source_dir)
            target = raw_dir / rel
            print(f"copy: {path} -> {target}")
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            seen.add(path)
            copied += 1
    return copied


def download_huggingface(source: Source, raw_dir: Path, *, dry_run: bool) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)
    marker = raw_dir / "HUGGINGFACE_DATASET.txt"
    marker.write_text(f"{source.url}\n", encoding="utf-8")
    try:
        import datasets  # type: ignore
    except Exception:
        print(f"HuggingFace datasets is not installed. Manual dataset id: {source.url}")
        return 0
    if dry_run:
        print(f"would load HuggingFace dataset: {source.url}")
        return 1
    if source.url == "hotpotqa/hotpot_qa":
        ds = datasets.load_dataset(source.url, "distractor")
    else:
        ds = datasets.load_dataset(source.url)
    for split, table in ds.items():
        out = raw_dir / f"{split}.jsonl"
        table.to_json(str(out), lines=True, force_ascii=False)
        print(f"wrote: {out}")
    return len(ds)


def write_manifest(spec: DatasetSpec, raw_dir: Path, copied: int) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "domain": spec.domain,
        "dataset": spec.name,
        "sources": [asdict(source) for source in spec.sources],
        "note": spec.note,
        "copied_files": copied,
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def has_payload_files(raw_dir: Path) -> bool:
    ignored = {"manifest.json", "HUGGINGFACE_DATASET.txt"}
    return raw_dir.exists() and any(path.is_file() and path.name not in ignored for path in raw_dir.rglob("*"))


def materialize(domain: str, dataset: str, *, raw_root: Path, processed_root: Path, dry_run: bool) -> None:
    cmd = [
        sys.executable,
        "-m",
        f"etl.materialize.{domain}",
        "--dataset",
        dataset,
        "--raw-root",
        str(raw_root),
        "--processed-root",
        str(processed_root),
    ]
    run(cmd, dry_run=dry_run)


def selected_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    specs = list(DATASETS)
    if args.domain != "all":
        specs = [spec for spec in specs if spec.domain == args.domain]
    if args.dataset != "all":
        wanted = {name.strip().lower() for name in args.dataset.split(",")}
        specs = [spec for spec in specs if spec.name.lower() in wanted]
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw datasets for the Warrant research plan.")
    parser.add_argument("--domain", default="all", choices=["all", "ctdg", "mtpp", "stpp", "tkg", "rag"])
    parser.add_argument("--dataset", default="all", help="Dataset name, comma list, or all.")
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--processed-root", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument("--materialize", action="store_true", help="Run etl.materialize after download.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = selected_specs(args)
    if not specs:
        raise SystemExit("No matching datasets.")

    for spec in specs:
        raw_dir = args.raw_root / spec.domain / spec.name
        copied = 0
        print(f"\n== {spec.domain}/{spec.name} ==")
        if spec.note:
            print(f"note: {spec.note}")
        for source in spec.sources:
            if source.kind == "git":
                repo = clone_or_pull(source, args.external_root, dry_run=args.dry_run)
                copied += copy_matching_files(repo, raw_dir, source.patterns, dry_run=args.dry_run)
            elif source.kind == "url":
                copied += download_url(source, raw_dir, dry_run=args.dry_run)
            elif source.kind == "huggingface":
                copied += download_huggingface(source, raw_dir, dry_run=args.dry_run)
            else:
                raise ValueError(f"unknown source kind: {source.kind}")
        if not spec.sources:
            raw_dir.mkdir(parents=True, exist_ok=True)
        if not args.dry_run:
            write_manifest(spec, raw_dir, copied)
        if args.materialize and (copied > 0 or has_payload_files(raw_dir)):
            try:
                materialize(
                    spec.domain,
                    spec.name,
                    raw_root=args.raw_root,
                    processed_root=args.processed_root,
                    dry_run=args.dry_run,
                )
            except subprocess.CalledProcessError as exc:
                if len(specs) == 1:
                    raise
                print(f"skip materialize: {spec.domain}/{spec.name} failed with exit code {exc.returncode}")
        elif args.materialize:
            print(f"skip materialize: no raw files for {spec.domain}/{spec.name}")


if __name__ == "__main__":
    main()
