"""Dataset registry and schema inspection.

The algorithms consume :class:`DatasetSpec` instead of embedding scene names,
row counts, or node limits in model code.  Dataset1/3 share a homogeneous node
space; Dataset2/4 are typed bipartite graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


GraphType = Literal["homogeneous", "bipartite"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    graph_type: GraphType
    has_split: bool
    expected_test_rows: int
    candidate_width: int = 100

    def train_path(self, root: Path) -> Path:
        return root / self.name / "train.csv"

    def test_path(self, root: Path) -> Path:
        return root / self.name / "test.csv"

    def output_path(self, root: Path) -> Path:
        return root / "submissions" / "prediction" / f"{self.name}.csv"


DATASETS: dict[str, DatasetSpec] = {
    "dataset1": DatasetSpec("dataset1", "homogeneous", False, 61_051),
    "dataset2": DatasetSpec("dataset2", "bipartite", True, 153_420),
    "dataset3": DatasetSpec("dataset3", "homogeneous", True, 157_670),
    "dataset4": DatasetSpec("dataset4", "bipartite", True, 2_322_538),
}


def get_dataset(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown dataset {name!r}; expected one of {sorted(DATASETS)}"
        ) from error


def validate_headers(root: Path, names: list[str]) -> None:
    """Validate CSV headers without materialising either large input file."""
    import csv

    for name in names:
        spec = get_dataset(name)
        paths = (spec.train_path(root), spec.test_path(root))
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "missing competition data:\n"
                + "\n".join(f"  - {path}" for path in missing)
            )
        with spec.train_path(root).open("r", encoding="utf-8", newline="") as handle:
            train_header = next(csv.reader(handle))
        required = ["src", "dst", "time"] + (["split"] if spec.has_split else [])
        if train_header != required:
            raise ValueError(
                f"{spec.train_path(root)} header {train_header} != {required}"
            )
        with spec.test_path(root).open("r", encoding="utf-8", newline="") as handle:
            test_header = next(csv.reader(handle))
        expected = ["src", "time"] + [
            f"c{index}" for index in range(1, spec.candidate_width + 1)
        ]
        if test_header != expected:
            raise ValueError(
                f"{spec.test_path(root)} must contain src,time,c1..c{spec.candidate_width}"
            )


def count_csv_rows(path: Path, block_size: int = 8 * 1024 * 1024) -> int:
    """Count data rows with bounded memory (the first line is the header)."""
    newlines = 0
    last = b""
    with path.open("rb", buffering=block_size) as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            newlines += block.count(b"\n")
            last = block[-1:]
    logical_lines = newlines + int(bool(last) and last != b"\n")
    return max(logical_lines - 1, 0)
