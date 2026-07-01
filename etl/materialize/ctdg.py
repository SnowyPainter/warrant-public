from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import (
    chronological_split,
    common_parser,
    first_existing_column,
    iter_files,
    processed_dir,
    raw_dir,
    read_table,
    require_files,
    write_metadata,
)


DOMAIN = "ctdg"


def read_ctdg_table(path: Path) -> pd.DataFrame:
    frame = read_table(path)
    if "comma_separated_list_of_features" not in frame.columns:
        return frame
    # JODIE-style CSVs have a short logical header followed by a variable-width
    # comma-separated feature vector.  pandas' default parser treats the first
    # field as an index when data rows are wider than the header, shifting
    # timestamp/label columns and silently corrupting the event stream.
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
        for line in handle:
            if line.strip():
                rows.append(line.rstrip("\n\r").split(","))
    if not rows or min(len(row) for row in rows) < 4:
        return frame
    width = max(len(row) for row in rows)
    columns = ["user_id", "item_id", "timestamp", "state_label"] + [f"feature_{idx}" for idx in range(max(0, width - 4))]
    padded = [row + [""] * (width - len(row)) for row in rows]
    return pd.DataFrame(padded, columns=columns)


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    src = first_existing_column(frame, ("src", "source", "source_id", "u", "user", "user_id", "from"))
    dst = first_existing_column(frame, ("dst", "destination", "target", "target_id", "i", "item", "item_id", "to"))
    ts = first_existing_column(frame, ("timestamp", "time", "ts", "t"))
    label = first_existing_column(frame, ("label", "state_label", "y"))
    if src is None or dst is None or ts is None:
        if frame.shape[1] < 3:
            raise ValueError("CTDG table needs at least source, destination, timestamp columns")
        src, dst, ts = frame.columns[:3]
    src_values = "src:" + frame[src].astype(str)
    dst_values = "dst:" + frame[dst].astype(str)
    node_index = pd.Index(pd.concat([src_values, dst_values], ignore_index=True).unique())
    node_map = pd.Series(range(len(node_index)), index=node_index)
    out = pd.DataFrame(
        {
            "src": src_values.map(node_map).astype("int64"),
            "dst": dst_values.map(node_map).astype("int64"),
            "timestamp": pd.to_numeric(frame[ts], errors="coerce"),
        }
    )
    out["label"] = pd.to_numeric(frame[label], errors="coerce").fillna(1.0) if label else 1.0
    feature_cols = [column for column in frame.columns if column not in {src, dst, ts, label}]
    for idx, column in enumerate(feature_cols):
        out[f"feat_{idx}"] = pd.to_numeric(frame[column], errors="coerce")
    return out.dropna(subset=["src", "dst", "timestamp"])


def materialize(dataset: str, raw_root: Path, processed_root: Path) -> None:
    raw = raw_dir(raw_root, DOMAIN, dataset)
    files = list(iter_files(raw, (".csv", ".txt", ".tsv", ".csv.gz", ".txt.gz", ".tsv.gz")))
    require_files(files, raw)
    frames = []
    used = []
    for file in files:
        try:
            frames.append(normalize_frame(read_ctdg_table(file)))
            used.append(file)
        except Exception as exc:
            print(f"skip {file}: {exc}")
    require_files(used, raw)
    events = chronological_split(pd.concat(frames, ignore_index=True))
    out = processed_dir(processed_root, DOMAIN, dataset)
    events.to_csv(out / "events.csv", index=False)
    write_metadata(
        out,
        domain=DOMAIN,
        dataset=dataset,
        source_files=used,
        rows=len(events),
        schema={"task": "ctdg_link_prediction", "primary": "events.csv"},
    )
    print(f"wrote {out / 'events.csv'} ({len(events)} rows)")


def main() -> None:
    args = common_parser("Materialize CTDG link-prediction datasets.").parse_args()
    materialize(args.dataset, args.raw_root, args.processed_root)


if __name__ == "__main__":
    main()
