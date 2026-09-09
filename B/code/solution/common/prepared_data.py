"""Convert competition CSV files into compact, memory-mapped arrays.

The conversion is deliberately separate from model code: CSV parsing and raw
identifier handling are non-core data concerns, while every downstream model
sees the same typed arrays.  For bipartite graphs only destinations observed in
training receive learned embedding rows; unseen test candidates are represented
by ``-1`` and are scored by statistical/cold-start channels instead of random,
untrained embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from solution.common.datasets import count_csv_rows, get_dataset, validate_headers


ROOT = Path(__file__).resolve().parents[3]
PREPARATION_SCHEMA_VERSION = 2


def open_array(
    path: Path, shape: tuple[int, ...], dtype, resume: bool = False
) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume and path.is_file():
        array = np.load(path, mmap_mode="r+")
        if array.shape == shape and array.dtype == np.dtype(dtype):
            return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def collect_train_domains(path: Path, chunksize: int) -> tuple[np.ndarray, np.ndarray]:
    source_blocks: list[np.ndarray] = []
    destination_blocks: list[np.ndarray] = []
    for frame in pd.read_csv(
        path,
        usecols=["src", "dst"],
        dtype={"src": np.int32, "dst": np.int32},
        chunksize=chunksize,
    ):
        source_blocks.append(np.unique(frame["src"].to_numpy(np.int32)))
        destination_blocks.append(np.unique(frame["dst"].to_numpy(np.int32)))
    return (
        np.unique(np.concatenate(source_blocks)),
        np.unique(np.concatenate(destination_blocks)),
    )


def compact(values: np.ndarray, domain: np.ndarray) -> np.ndarray:
    """Map values into a sorted domain, returning -1 for unseen identifiers."""
    flat = np.asarray(values, dtype=np.int32).reshape(-1)
    positions = np.searchsorted(domain, flat)
    output = np.full(len(flat), -1, dtype=np.int32)
    valid = positions < len(domain)
    output[valid] = positions[valid]
    valid &= domain[np.minimum(positions, len(domain) - 1)] == flat
    output[~valid] = -1
    return output.reshape(values.shape)


def prepare(dataset: str, output_dir: Path, chunksize: int) -> Path:
    spec = get_dataset(dataset)
    validate_headers(ROOT, [dataset])
    train_path = spec.train_path(ROOT)
    test_path = spec.test_path(ROOT)
    train_rows = count_csv_rows(train_path)
    test_rows = count_csv_rows(test_path)
    if test_rows != spec.expected_test_rows:
        raise ValueError(
            f"{dataset}: expected {spec.expected_test_rows} test rows, got {test_rows}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_signature = {
        "preparation_schema_version": PREPARATION_SCHEMA_VERSION,
        "preparation_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "dataset": dataset,
        "train_size": train_path.stat().st_size,
        "train_mtime_ns": train_path.stat().st_mtime_ns,
        "test_size": test_path.stat().st_size,
        "test_mtime_ns": test_path.stat().st_mtime_ns,
        "train_rows": train_rows,
        "test_rows": test_rows,
    }
    prepared_names = (
        "train_src.npy", "train_dst.npy", "train_time.npy", "train_split.npy",
        "test_src.npy", "test_time.npy", "test_candidates_raw.npy",
        "test_candidates.npy",
    )
    metadata_path = output_dir / "metadata.json"
    source_path = output_dir / "source_ids.npy"
    destination_path = output_dir / "destination_ids.npy"
    complete_files = (
        source_path, destination_path,
        *(output_dir / name for name in prepared_names),
    )
    if metadata_path.is_file() and all(path.is_file() for path in complete_files):
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_metadata.get("source_signature") == source_signature:
            print(f"resume {dataset}: prepared arrays already complete", flush=True)
            return metadata_path

    domain_marker = output_dir / "domains.json"
    if (
        domain_marker.is_file() and source_path.is_file() and destination_path.is_file()
        and json.loads(domain_marker.read_text(encoding="utf-8")) == source_signature
    ):
        source_ids = np.load(source_path)
        destination_ids = np.load(destination_path)
        print(f"resume {dataset}: reuse compact ID domains", flush=True)
    elif spec.graph_type == "bipartite":
        source_ids, destination_ids = collect_train_domains(train_path, chunksize)
    else:
        # Dataset1/3 identifiers are small enough for an identity mapping.  It
        # preserves directed/reverse structural semantics and avoids a costly
        # scan over all test candidates solely to build a shared vocabulary.
        maximum = -1
        for frame in pd.read_csv(
            train_path,
            usecols=["src", "dst"],
            dtype={"src": np.int32, "dst": np.int32},
            chunksize=chunksize,
        ):
            maximum = max(maximum, int(frame[["src", "dst"]].to_numpy().max()))
        for frame in pd.read_csv(
            test_path,
            dtype=np.int32,
            chunksize=max(10_000, chunksize // 4),
        ):
            maximum = max(maximum, int(frame.drop(columns=["time"]).to_numpy().max()))
        source_ids = destination_ids = np.arange(maximum + 1, dtype=np.int32)

    np.save(source_path, source_ids)
    np.save(destination_path, destination_ids)
    domain_marker.write_text(
        json.dumps(source_signature, indent=2) + "\n", encoding="utf-8"
    )

    progress_path = output_dir / "prepare_progress.json"
    progress = {"signature": source_signature, "train_rows": 0, "test_rows": 0}
    if progress_path.is_file():
        loaded_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if loaded_progress.get("signature") == source_signature:
            progress = loaded_progress
    if (int(progress["train_rows"]) > 0 or int(progress["test_rows"]) > 0) and not all(
        (output_dir / name).is_file() for name in prepared_names
    ):
        print(f"restart {dataset}: incomplete prepared-array set", flush=True)
        progress = {"signature": source_signature, "train_rows": 0, "test_rows": 0}
    resume_arrays = int(progress["train_rows"]) > 0 or int(progress["test_rows"]) > 0
    train_src = open_array(output_dir / "train_src.npy", (train_rows,), np.int32, resume_arrays)
    train_dst = open_array(output_dir / "train_dst.npy", (train_rows,), np.int32, resume_arrays)
    train_time = open_array(output_dir / "train_time.npy", (train_rows,), np.int64, resume_arrays)
    train_split = open_array(output_dir / "train_split.npy", (train_rows,), np.uint8, resume_arrays)
    offset = int(progress["train_rows"])
    parsed = 0
    for frame in pd.read_csv(train_path, chunksize=chunksize):
        frame_start = parsed
        parsed += len(frame)
        if parsed <= offset:
            continue
        if frame_start < offset:
            frame = frame.iloc[offset - frame_start :]
        stop = offset + len(frame)
        raw_src = frame["src"].to_numpy(np.int32)
        raw_dst = frame["dst"].to_numpy(np.int32)
        train_src[offset:stop] = compact(raw_src, source_ids)
        train_dst[offset:stop] = compact(raw_dst, destination_ids)
        train_time[offset:stop] = frame["time"].to_numpy(np.int64)
        train_split[offset:stop] = (
            frame["split"].to_numpy(np.uint8) if "split" in frame else 0
        )
        for array in (train_src, train_dst, train_time, train_split):
            array.flush()
        offset = stop
        progress["train_rows"] = offset
        temporary = progress_path.with_name(progress_path.name + ".tmp")
        temporary.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        temporary.replace(progress_path)
        print(f"prepared {dataset} train {stop:,}/{train_rows:,}", flush=True)

    test_src = open_array(output_dir / "test_src.npy", (test_rows,), np.int32, resume_arrays)
    test_time = open_array(output_dir / "test_time.npy", (test_rows,), np.int64, resume_arrays)
    test_candidates_raw = open_array(
        output_dir / "test_candidates_raw.npy",
        (test_rows, spec.candidate_width),
        np.int32, resume_arrays,
    )
    test_candidates = open_array(
        output_dir / "test_candidates.npy",
        (test_rows, spec.candidate_width),
        np.int32, resume_arrays,
    )
    offset = int(progress["test_rows"])
    parsed = 0
    test_chunksize = max(10_000, chunksize // 4)
    for frame in pd.read_csv(test_path, chunksize=test_chunksize):
        frame_start = parsed
        parsed += len(frame)
        if parsed <= offset:
            continue
        if frame_start < offset:
            frame = frame.iloc[offset - frame_start :]
        stop = offset + len(frame)
        raw_src = frame["src"].to_numpy(np.int32)
        raw_candidates = frame.iloc[:, 2:].to_numpy(np.int32)
        test_src[offset:stop] = compact(raw_src, source_ids)
        test_time[offset:stop] = frame["time"].to_numpy(np.int64)
        test_candidates_raw[offset:stop] = raw_candidates
        test_candidates[offset:stop] = compact(raw_candidates, destination_ids)
        for array in (test_src, test_time, test_candidates_raw, test_candidates):
            array.flush()
        offset = stop
        progress["test_rows"] = offset
        temporary = progress_path.with_name(progress_path.name + ".tmp")
        temporary.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
        temporary.replace(progress_path)
        print(f"prepared {dataset} test {stop:,}/{test_rows:,}", flush=True)

    for array in (
        train_src, train_dst, train_time, train_split,
        test_src, test_time, test_candidates_raw, test_candidates,
    ):
        array.flush()
    unseen_candidate_slots = 0
    for start in range(0, test_rows, test_chunksize):
        unseen_candidate_slots += int(
            (np.asarray(test_candidates[start:start + test_chunksize]) < 0).sum()
        )
    metadata = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "dataset": dataset,
        "graph_type": spec.graph_type,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "candidate_width": spec.candidate_width,
        "num_sources": int(len(source_ids)),
        "num_destinations": int(len(destination_ids)),
        "unseen_test_sources": int((np.asarray(test_src) < 0).sum()),
        "unseen_test_candidate_slots": unseen_candidate_slots,
        "source_signature": source_signature,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    progress_path.unlink(missing_ok=True)
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(sorted(("dataset3", "dataset4"))), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    output_dir = args.output_dir or ROOT / "outputs" / args.dataset / "prepared"
    print(prepare(args.dataset, output_dir, args.chunksize))


if __name__ == "__main__":
    main()
