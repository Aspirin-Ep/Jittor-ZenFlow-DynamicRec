"""Streaming prediction output and validation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    low = values.min(axis=1, keepdims=True)
    high = values.max(axis=1, keepdims=True)
    return (values - low) / np.maximum(high - low, 1e-12)


def write_score_rows(handle, values: np.ndarray) -> None:
    np.savetxt(handle, normalize_rows(values), delimiter=",", fmt="%.8f")


def validate_score_csv(path: Path, rows: int, width: int = 100) -> dict[str, object]:
    count = 0
    low = float("inf")
    high = float("-inf")
    with path.open("r", encoding="utf-8") as handle:
        for count, line in enumerate(handle, start=1):
            values = np.fromstring(line, sep=",", dtype=np.float64)
            if len(values) != width or not np.isfinite(values).all():
                raise ValueError(f"{path}: invalid row {count}")
            low = min(low, float(values.min()))
            high = max(high, float(values.max()))
    if count != rows:
        raise ValueError(f"{path}: expected {rows} rows, got {count}")
    if low < 0.0 or high > 1.0:
        raise ValueError(f"{path}: scores outside [0,1]: [{low}, {high}]")
    return {"valid": True, "rows": count, "width": width, "min": low, "max": high}
