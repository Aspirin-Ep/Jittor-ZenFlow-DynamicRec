"""Verify the uncompressed B-board code-review directory before packaging."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "提交说明文档.pdf",
    "run.py",
    "requirements.txt",
    "scripts/run_checkpoint_inference.py",
    "code/solution/dataset3/train_ranker.py",
    "code/solution/dataset4/model.py",
    "code/solution/dataset4/pipeline.py",
    "docs/B榜最优结果复现说明.md",
    "docs/A榜到B榜算法改动说明.md",
    "checkpoints/dataset3/ranker.txt",
    "checkpoints/dataset4/graph_mf/graph_mf.pkl",
    "checkpoints/dataset4/craft/craft_best.pkl",
    "checkpoints/dataset4/final_ranker.txt",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    missing = [relative for relative in REQUIRED if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError("missing required files: " + ", ".join(missing))

    forbidden_directories = [
        name for name in ("dataset1", "dataset2", "dataset3", "dataset4")
        if (ROOT / name).exists()
    ]
    forbidden_files = [
        path.relative_to(ROOT) for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".zip", ".npy"}
    ]
    if forbidden_directories or forbidden_files:
        raise ValueError(
            f"raw/prediction payload found: dirs={forbidden_directories}, "
            f"files={forbidden_files}"
        )

    checksum_root = ROOT / "checkpoints"
    checksum_file = checksum_root / "SHA256SUMS"
    checked = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = checksum_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"checksum mismatch: {relative}")
        checked += 1

    files = [path for path in ROOT.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    print({
        "valid": True,
        "files": len(files),
        "bytes": total_bytes,
        "checkpoint_checksums": checked,
        "raw_datasets": 0,
        "prediction_files": 0,
    })


if __name__ == "__main__":
    main()
