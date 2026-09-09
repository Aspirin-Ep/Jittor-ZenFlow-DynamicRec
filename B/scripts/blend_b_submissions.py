#!/usr/bin/env python3
"""Stream a rank blend of two Dataset4 predictions into a B-board ZIP."""

from __future__ import annotations

import argparse
import io
import itertools
import zipfile
from pathlib import Path

import numpy as np


WIDTH = 100


def read_block(handle, rows: int) -> np.ndarray:
    lines = list(itertools.islice(handle, rows))
    if not lines:
        return np.empty((0, WIDTH), dtype=np.float32)
    payload = b",".join(line.strip() for line in lines)
    values = np.fromstring(payload.decode("ascii"), sep=",", dtype=np.float32)
    if values.size != len(lines) * WIDTH:
        raise ValueError("prediction block does not contain 100 scores per row")
    return values.reshape(len(lines), WIDTH)


def rank_scores(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, axis=1, kind="stable")
    ranks = np.empty(order.shape, dtype=np.int16)
    np.put_along_axis(
        ranks, order, np.arange(WIDTH, dtype=np.int16)[None, :], axis=1
    )
    return (1.0 - ranks.astype(np.float32) / (WIDTH - 1)).astype(np.float32)


def blend(
    baseline_zip: Path,
    candidate_csv: Path,
    dataset3_csv: Path,
    output_zip: Path,
    candidate_weight: float,
    block_rows: int,
) -> int:
    if not 0.0 <= candidate_weight <= 1.0:
        raise ValueError("candidate weight must be inside [0, 1]")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with (
        zipfile.ZipFile(baseline_zip) as baseline,
        baseline.open("dataset4.csv") as baseline_handle,
        candidate_csv.open("rb") as candidate_handle,
        zipfile.ZipFile(
            output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as output,
    ):
        output.write(dataset3_csv, "dataset3.csv")
        # Dataset4 is ~2.55 GB before compression, so ZipFile cannot infer the
        # final member size early enough to enable ZIP64 automatically.
        with output.open("dataset4.csv", "w", force_zip64=True) as raw_output:
            text_output = io.TextIOWrapper(raw_output, encoding="ascii", newline="")
            while True:
                old = read_block(baseline_handle, block_rows)
                new = read_block(candidate_handle, block_rows)
                if len(old) != len(new):
                    raise ValueError("baseline and candidate Dataset4 row counts differ")
                if not len(old):
                    break
                scores = (
                    (1.0 - candidate_weight) * rank_scores(old)
                    + candidate_weight * rank_scores(new)
                )
                np.savetxt(text_output, scores, delimiter=",", fmt="%.8f")
                rows += len(scores)
            text_output.flush()
            # Detach so TextIOWrapper does not try to close an already-managed
            # ZipExtFile during context teardown.
            text_output.detach()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-zip", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--dataset3-csv", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--candidate-weight", type=float, required=True)
    parser.add_argument("--block-rows", type=int, default=2_500)
    args = parser.parse_args()
    if args.block_rows <= 0:
        parser.error("--block-rows must be positive")
    rows = blend(
        args.baseline_zip,
        args.candidate_csv,
        args.dataset3_csv,
        args.output_zip,
        args.candidate_weight,
        args.block_rows,
    )
    print(f"wrote {args.output_zip}: Dataset4 rows={rows:,}")


if __name__ == "__main__":
    main()
