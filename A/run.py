"""Reproduce the complete competition submission from raw CSV files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "code"
PYTHON = sys.executable
STATE_ROOT = ROOT / "outputs/state"
DATASET1_INFERENCE_DIR = ROOT / "outputs/dataset1/inference"
DATASET_SHAPES = {
    "dataset1": (61_051, 100),
    "dataset2": (153_420, 100),
}
FINAL_DIR = ROOT / "submissions/prediction"
FINAL_ZIP = FINAL_DIR / "result.zip"


def python_environment() -> dict[str, str]:
    environment = os.environ.copy()
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(SOURCE_ROOT)
        if not previous
        else str(SOURCE_ROOT) + os.pathsep + previous
    )
    return environment


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = [ROOT / "run.py", ROOT / "requirements.txt"]
    paths.extend(sorted(SOURCE_ROOT.rglob("*.py")))
    paths.extend(sorted(SOURCE_ROOT.rglob("*.json")))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(arguments: list[object]) -> None:
    command = [str(argument) for argument in arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=ROOT,
        env=python_environment(),
        check=True,
    )


def run_stage(
    name: str,
    commands: list[list[object]],
    fingerprint: str,
    force: set[str],
) -> None:
    state_path = STATE_ROOT / f"{name}.json"
    serialized_commands = [
        [str(value) for value in command] for command in commands
    ]
    if state_path.is_file() and name not in force:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("source_fingerprint") == fingerprint
            and state.get("commands") == serialized_commands
        ):
            print(f"[skip] {name}", flush=True)
            return
    print(f"[stage] {name}", flush=True)
    for command in commands:
        run_command(command)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "stage": name,
                "source_fingerprint": fingerprint,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "commands": serialized_commands,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def require_raw_data() -> None:
    import pandas as pd

    required = [
        ROOT / dataset / filename
        for dataset in DATASET_SHAPES
        for filename in ("train.csv", "test.csv")
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少比赛原始数据：\n"
            + "\n".join(f"  - {path.relative_to(ROOT)}" for path in missing)
        )
    expected_headers = {
        "dataset1/train.csv": ("src", "dst", "time"),
        "dataset1/test.csv": ("src", "time"),
        "dataset2/train.csv": ("src", "dst", "time", "split"),
        "dataset2/test.csv": ("src", "time"),
    }
    for relative, prefix in expected_headers.items():
        columns = tuple(pd.read_csv(ROOT / relative, nrows=0).columns)
        if columns[: len(prefix)] != prefix:
            raise ValueError(
                f"{relative} 表头错误：期望前缀 {prefix}，实际 {columns}"
            )
        if relative.endswith("test.csv") and len(columns) != 102:
            raise ValueError(f"{relative} 应包含 100 个候选列")


def normalize_rows(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    low = values.min(axis=1, keepdims=True)
    high = values.max(axis=1, keepdims=True)
    return ((values - low) / np.maximum(high - low, 1e-12)).astype(
        np.float32
    )


def verify_csv(path: Path, shape: tuple[int, int]) -> None:
    import numpy as np

    values = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if values.shape != shape:
        raise ValueError(f"{path}: 期望 {shape}，实际 {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: 存在非有限数值")
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError(f"{path}: 分数不在 [0, 1] 内")


def build_dataset1(fingerprint: str, force: set[str], seed: int) -> Path:
    import numpy as np

    score_path = DATASET1_INFERENCE_DIR / "dataset1_scores.npy"
    csv_path = DATASET1_INFERENCE_DIR / "dataset1.csv"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    run_stage(
        "dataset1_ranker",
        [
            [
                PYTHON,
                "-m",
                "solution.dataset1.train_ranker",
                "--real",
                "--out",
                score_path,
                "--tag",
                "dataset1_ranker",
                "--seed",
                seed,
            ]
        ],
        fingerprint,
        force,
    )
    scores = normalize_rows(np.load(score_path))
    if scores.shape != DATASET_SHAPES["dataset1"]:
        raise ValueError(f"Dataset1 score shape mismatch: {scores.shape}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(csv_path, scores, delimiter=",", fmt="%.8f")
    verify_csv(csv_path, DATASET_SHAPES["dataset1"])
    return csv_path


def build_validation_features(
    fingerprint: str,
    force: set[str],
    seed: int,
) -> None:
    run_stage(
        "validation_benchmarks",
        [[PYTHON, "-m", "training.dataset2.build_benchmarks", "--seed", seed]],
        fingerprint,
        force,
    )
    for fold in range(3):
        run_stage(
            f"validation_oof_fold_{fold}",
            [
                [
                    PYTHON,
                    "-m",
                    "training.dataset2.build_oof_features",
                    "--benchmark",
                    f"outputs/validation_protocol/data/fold_{fold}/benchmark.npz",
                    "--artifacts",
                    f"outputs/validation_protocol/artifacts/fold_{fold}",
                    "--stage",
                    "all",
                    "--seed",
                    seed,
                ]
            ],
            fingerprint,
            force,
        )
    run_stage(
        "validation_temporal_stack",
        [
            [
                PYTHON,
                "-m",
                "solution.dataset2.temporal_stack",
                "--cold-weight",
                "0.15",
                "--seed",
                seed,
            ]
        ],
        fingerprint,
        force,
    )
    for rows in (30_000, 60_000):
        run_stage(
            f"validation_anchor_{rows}",
            [
                [
                    PYTHON,
                    "-m",
                    "training.dataset2.anchor_ranker",
                    "--rows-per-fold",
                    rows,
                    "--output-dir",
                    f"outputs/dataset2/validation/anchor_{rows}",
                    "--seed",
                    seed,
                ]
            ],
            fingerprint,
            force,
        )
    for fold in range(3):
        run_stage(
            f"validation_group_features_{fold}",
            [
                [
                    PYTHON,
                    "-m",
                    "solution.dataset2.group_features",
                    "validate",
                    "--fold",
                    fold,
                    "--with-cooc",
                ]
            ],
            fingerprint,
            force,
        )
    for fold in range(3):
        run_stage(
            f"validation_group_ranker_{fold}",
            [
                [
                    PYTHON,
                    "-m",
                    "solution.dataset2.group_ranker",
                    "--target-fold",
                    fold,
                    "--rows-per-fold",
                    60_000,
                    "--seed",
                    seed,
                ]
            ],
            fingerprint,
            force,
        )
    for fold in range(3):
        run_stage(
            f"validation_graph_ranker_{fold}",
            [
                [
                    PYTHON,
                    "-m",
                    "solution.dataset2.graph_ranker",
                    "validate",
                    "--target-fold",
                    fold,
                    "--seed",
                    seed,
                ]
            ],
            fingerprint,
            force,
        )


def build_production_features(
    fingerprint: str,
    force: set[str],
    seed: int,
) -> None:
    run_stage(
        "base_model_training",
        [[PYTHON, "-m", "training.base_models.train", "--seed", seed]],
        fingerprint,
        force,
    )
    run_stage(
        "base_model_inference",
        [[PYTHON, "-m", "solution.base.dataset2", "--seed", seed]],
        fingerprint,
        force,
    )
    run_stage(
        "basket_features",
        [
            [
                PYTHON,
                "-m",
                "solution.dataset2.build_basket_scores",
                "--base",
                "outputs/dataset2/production/base/dataset2.csv",
                "--output-dir",
                "outputs/dataset2/production/basket",
            ]
        ],
        fingerprint,
        force,
    )
    run_stage(
        "test_temporal_features",
        [
            [
                PYTHON,
                "-m",
                "solution.dataset2.build_temporal_features",
                "--output",
                "outputs/dataset2/production/temporal_features.npy",
            ]
        ],
        fingerprint,
        force,
    )
    run_stage(
        "candidate_distribution_calibration",
        [[PYTHON, "-m", "solution.dataset2.candidate_distribution", "--seed", seed]],
        fingerprint,
        force,
    )
    run_stage(
        "marginal_calibration",
        [[PYTHON, "-m", "solution.dataset2.marginal_calibration", "--seed", seed]],
        fingerprint,
        force,
    )
    run_stage(
        "item_collaborative_calibration",
        [[PYTHON, "-m", "solution.dataset2.collaborative_features", "--seed", seed]],
        fingerprint,
        force,
    )
    run_stage(
        "multiscale_collaborative_calibration",
        [[PYTHON, "-m", "solution.dataset2.multiscale_collaborative", "--seed", seed]],
        fingerprint,
        force,
    )
    run_stage(
        "user_collaborative_calibration",
        [[PYTHON, "-m", "solution.dataset2.user_collaborative", "--seed", seed]],
        fingerprint,
        force,
    )
    run_stage(
        "distribution_ranker_training",
        [[PYTHON, "-m", "training.dataset2.train_distribution_ranker", "--seed", seed]],
        fingerprint,
        force,
    )
    run_stage(
        "distribution_ranker_inference",
        [[PYTHON, "-m", "training.dataset2.apply_distribution_ranker"]],
        fingerprint,
        force,
    )
    run_stage(
        "collaborative_ranker",
        [[PYTHON, "-m", "training.dataset2.build_collaborative_stack", "--seed", seed]],
        fingerprint,
        force,
    )
    run_stage(
        "temporal_stack_inference",
        [[PYTHON, "-m", "training.dataset2.apply_temporal_stack", "--seed", seed]],
        fingerprint,
        force,
    )
    run_stage(
        "anchor_ensemble_inference",
        [[PYTHON, "-m", "training.dataset2.apply_anchor_ensemble", "--seed", seed]],
        fingerprint,
        force,
    )
    run_stage(
        "group_ranker_inference",
        [[PYTHON, "-m", "solution.dataset2.group_inference", "--seed", seed]],
        fingerprint,
        force,
    )


def build_dataset2(fingerprint: str, force: set[str], seed: int) -> Path:
    build_validation_features(fingerprint, force, seed)
    build_production_features(fingerprint, force, seed)
    run_stage(
        "final_candidate_ranker",
        [[
            PYTHON,
            "-m",
            "solution.dataset2.final_ranker",
            "apply",
            "--seed",
            seed,
        ]],
        fingerprint,
        force,
    )
    dataset2 = FINAL_DIR / "dataset2.csv"
    verify_csv(dataset2, DATASET_SHAPES["dataset2"])
    return dataset2


def package_submission(dataset1: Path, dataset2: Path) -> Path:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    final_dataset1 = FINAL_DIR / "dataset1.csv"
    if dataset1.resolve() != final_dataset1.resolve():
        final_dataset1.write_bytes(dataset1.read_bytes())
    final_dataset2 = FINAL_DIR / "dataset2.csv"
    if dataset2.resolve() != final_dataset2.resolve():
        final_dataset2.write_bytes(dataset2.read_bytes())
    verify_csv(final_dataset1, DATASET_SHAPES["dataset1"])
    verify_csv(final_dataset2, DATASET_SHAPES["dataset2"])
    with zipfile.ZipFile(
        FINAL_ZIP,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.write(final_dataset1, "dataset1.csv")
        archive.write(final_dataset2, "dataset2.csv")
    manifest = {
        "method": "temporal_graph_collaborative_candidate_ranker",
        "dataset1_sha256": file_digest(final_dataset1),
        "dataset2_sha256": file_digest(final_dataset2),
        "zip_sha256": file_digest(FINAL_ZIP),
    }
    (FINAL_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return FINAL_ZIP


def list_stages() -> None:
    stages = [
        "dataset1_ranker",
        "validation_benchmarks",
        "validation_oof_fold_0..2",
        "validation_temporal_stack",
        "validation_anchor_30000/60000",
        "validation_group_features_0..2",
        "validation_group_ranker_0..2",
        "validation_graph_ranker_0..2",
        "base_model_training",
        "base_model_inference",
        "basket_features",
        "test_temporal_features",
        "candidate_distribution_calibration",
        "marginal_calibration",
        "item/multiscale/user_collaborative_calibration",
        "distribution_ranker_training/inference",
        "collaborative_ranker",
        "temporal_stack_inference",
        "anchor_ensemble_inference",
        "group_ranker_inference",
        "final_candidate_ranker",
    ]
    print("\n".join(stages))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=("dataset1", "dataset2", "all", "package"),
        default="all",
    )
    parser.add_argument(
        "--force-stage",
        action="append",
        default=[],
        help="忽略指定阶段的完成状态；可以重复传入。",
    )
    parser.add_argument("--list-stages", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=20260724,
        help="完整流水线统一随机种子。",
    )
    args = parser.parse_args()
    if args.list_stages:
        list_stages()
        return

    require_raw_data()
    fingerprint = source_fingerprint()
    force = set(args.force_stage)
    dataset1 = DATASET1_INFERENCE_DIR / "dataset1.csv"
    dataset2 = FINAL_DIR / "dataset2.csv"
    if args.target in {"dataset1", "dataset2", "all"}:
        dataset1 = build_dataset1(fingerprint, force, args.seed)
    if args.target in {"dataset2", "all"}:
        dataset2 = build_dataset2(fingerprint, force, args.seed)
    if args.target == "dataset1":
        print(dataset1)
        return
    if args.target == "package":
        if not dataset1.is_file() or not dataset2.is_file():
            raise FileNotFoundError("请先完成 dataset1 和 dataset2 生成")
    print(package_submission(dataset1, dataset2))


if __name__ == "__main__":
    main()
