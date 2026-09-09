"""Reproduce the complete competition submission from raw CSV files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "code"
PYTHON = sys.executable
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
STATE_ROOT = ROOT / "outputs/state"
DATASET1_INFERENCE_DIR = ROOT / "outputs/dataset1/inference"
DATASET_SHAPES = {
    "dataset1": (61_051, 100),
    "dataset2": (153_420, 100),
    "dataset3": (157_670, 100),
    "dataset4": (2_322_538, 100),
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


def source_fingerprint(datasets: tuple[str, ...] | None = None) -> str:
    """Hash only code and raw data relevant to the selected family."""
    digest = hashlib.sha256()
    paths = [ROOT / "run.py", ROOT / "requirements.txt"]
    if datasets is None:
        paths.extend(sorted(SOURCE_ROOT.rglob("*.py")))
        paths.extend(sorted(SOURCE_ROOT.rglob("*.json")))
        data_scopes = tuple(DATASET_SHAPES)
    else:
        directories = [SOURCE_ROOT / "solution/common"]
        if "dataset3" in datasets:
            directories.extend([
                SOURCE_ROOT / "solution/dataset1",
                SOURCE_ROOT / "solution/dataset3",
            ])
        if "dataset4" in datasets:
            directories.append(SOURCE_ROOT / "solution/dataset4")
        for directory in directories:
            paths.extend(sorted(directory.rglob("*.py")))
            paths.extend(sorted(directory.rglob("*.json")))
        data_scopes = datasets
    paths = sorted(set(paths))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    for dataset in data_scopes:
        for filename in ("train.csv", "test.csv"):
            path = ROOT / dataset / filename
            if path.is_file():
                stat = path.stat()
                digest.update(
                    f"{dataset}/{filename}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
                )
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
    outputs: tuple[Path, ...] = (),
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
            and all(path.is_file() for path in outputs)
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


def require_raw_data(datasets: list[str]) -> None:
    from solution.common.datasets import validate_headers
    required = [
        ROOT / dataset / filename
        for dataset in datasets
        for filename in ("train.csv", "test.csv")
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "缺少比赛原始数据：\n"
            + "\n".join(f"  - {path.relative_to(ROOT)}" for path in missing)
        )
    validate_headers(ROOT, datasets)


def normalize_rows(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    low = values.min(axis=1, keepdims=True)
    high = values.max(axis=1, keepdims=True)
    return ((values - low) / np.maximum(high - low, 1e-12)).astype(
        np.float32
    )


def verify_csv(path: Path, shape: tuple[int, int]) -> None:
    from solution.common.scoring import validate_score_csv
    validate_score_csv(path, shape[0], shape[1])


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


def build_dataset3(
    fingerprint: str, force: set[str], seed: int, workers: int
) -> Path:
    dataset3 = FINAL_DIR / "dataset3.csv"
    run_stage(
        "dataset3_ranker",
        [[
            PYTHON, "-m", "solution.dataset3.train_ranker",
            "--output", dataset3,
            "--model", "outputs/dataset3/ranker.txt",
            "--threads", workers,
            "--feature-workers", max(1, min(workers, 4)),
            "--seed", seed,
        ]],
        fingerprint,
        force,
        outputs=(dataset3,),
    )
    verify_csv(dataset3, DATASET_SHAPES["dataset3"])
    return dataset3


def build_dataset4(
    fingerprint: str,
    force: set[str],
    seed: int,
    workers: int,
    chunk_rows: int,
    cf_dim: int,
) -> Path:
    prepared = ROOT / "outputs/dataset4/prepared"
    metadata = prepared / "metadata.json"
    run_stage(
        "dataset4_prepare",
        [[
            PYTHON, "-m", "solution.common.prepared_data",
            "--dataset", "dataset4", "--output-dir", prepared,
        ]],
        fingerprint,
        force,
        outputs=tuple(
            [metadata]
            + [prepared / f"{name}.npy" for name in (
                "source_ids", "destination_ids", "train_src", "train_dst",
                "train_time", "train_split", "test_src", "test_time",
                "test_candidates", "test_candidates_raw",
            )]
        ),
    )
    dataset4 = FINAL_DIR / "dataset4.csv"
    run_stage(
        "dataset4_ranker",
        [[
            PYTHON, "-m", "solution.dataset4.pipeline",
            "--prepared", prepared,
            "--output", dataset4,
            "--workers", workers,
            "--chunk-rows", chunk_rows,
            "--cf-dim", cf_dim,
            "--seed", seed,
        ]],
        fingerprint,
        force,
        outputs=(dataset4,),
    )
    verify_csv(dataset4, DATASET_SHAPES["dataset4"])
    return dataset4


def package_b_submission(dataset3: Path, dataset4: Path) -> Path:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    targets = {"dataset3": dataset3, "dataset4": dataset4}
    for name, source in targets.items():
        destination = FINAL_DIR / f"{name}.csv"
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        verify_csv(destination, DATASET_SHAPES[name])
        targets[name] = destination
    with zipfile.ZipFile(FINAL_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in targets.items():
            archive.write(path, f"{name}.csv")
    manifest = {
        "method": (
            "craft_tree_temporal_hawkes_repeat_transition_graphmf_"
            "lightgcn_sparse_collaborative_ease_learned_context_stack_"
            "candidate_ranker"
        ),
        **{f"{name}_sha256": file_digest(path) for name, path in targets.items()},
        "zip_sha256": file_digest(FINAL_ZIP),
    }
    (FINAL_DIR / "manifest-b.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return FINAL_ZIP


def package_submission(dataset1: Path, dataset2: Path) -> Path:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    final_dataset1 = FINAL_DIR / "dataset1.csv"
    if dataset1.resolve() != final_dataset1.resolve():
        shutil.copyfile(dataset1, final_dataset1)
    final_dataset2 = FINAL_DIR / "dataset2.csv"
    if dataset2.resolve() != final_dataset2.resolve():
        shutil.copyfile(dataset2, final_dataset2)
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
        "dataset3_ranker",
        "dataset4_prepare",
        "dataset4_ranker",
    ]
    print("\n".join(stages))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=(
            "dataset1", "dataset2", "dataset3", "dataset4",
            "a", "b", "all", "package", "package-b",
        ),
        default="b",
    )
    parser.add_argument(
        "--force-stage",
        action="append",
        default=[],
        help="忽略指定阶段的完成状态；可以重复传入。",
    )
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1),
        help="NumPy/SciPy/LightGBM 使用的 CPU 线程数。",
    )
    parser.add_argument(
        "--chunk-rows", type=int, default=2_500,
        help="Dataset4 每个推理块的查询行数；18GB 内存默认 2500。",
    )
    parser.add_argument(
        "--cf-dim", type=int, default=128,
        help="Dataset4 CountSketch 协同维度；18GB 内存推荐 128，紧张时用 64。",
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

    for variable in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, str(max(1, args.workers)))
    if args.target in {"package", "package-b"}:
        required = []
    elif args.target in {"dataset1"}:
        required = ["dataset1"]
    elif args.target in {"dataset2", "a"}:
        required = ["dataset1", "dataset2"]
    elif args.target == "dataset3":
        required = ["dataset3"]
    elif args.target == "dataset4":
        required = ["dataset4"]
    else:
        required = ["dataset3", "dataset4"] if args.target in {"b", "package-b"} else list(DATASET_SHAPES)
    require_raw_data(required)
    fingerprint = source_fingerprint()
    dataset3_fingerprint = source_fingerprint(("dataset3",))
    dataset4_fingerprint = source_fingerprint(("dataset4",))
    force = set(args.force_stage)
    dataset1 = DATASET1_INFERENCE_DIR / "dataset1.csv"
    dataset2 = FINAL_DIR / "dataset2.csv"
    dataset3 = FINAL_DIR / "dataset3.csv"
    dataset4 = FINAL_DIR / "dataset4.csv"
    if args.target in {"dataset1", "dataset2", "a", "all"}:
        dataset1 = build_dataset1(fingerprint, force, args.seed)
    if args.target in {"dataset2", "a", "all"}:
        dataset2 = build_dataset2(fingerprint, force, args.seed)
    if args.target in {"dataset3", "b", "all"}:
        dataset3 = build_dataset3(
            dataset3_fingerprint, force, args.seed, args.workers
        )
    if args.target in {"dataset4", "b", "all"}:
        dataset4 = build_dataset4(
            dataset4_fingerprint, force, args.seed, args.workers,
            args.chunk_rows, args.cf_dim,
        )
    if args.target == "dataset1":
        print(dataset1)
        return
    if args.target == "dataset2":
        print(package_submission(dataset1, dataset2))
        return
    if args.target == "dataset3":
        print(dataset3)
        return
    if args.target == "dataset4":
        print(dataset4)
        return
    if args.target == "package":
        if not dataset1.is_file() or not dataset2.is_file():
            raise FileNotFoundError("请先完成 dataset1 和 dataset2 生成")
        print(package_submission(dataset1, dataset2))
        return
    if args.target == "a":
        print(package_submission(dataset1, dataset2))
        return
    if args.target == "package-b" and (not dataset3.is_file() or not dataset4.is_file()):
        raise FileNotFoundError("请先完成 dataset3 和 dataset4 生成")
    print(package_b_submission(dataset3, dataset4))


if __name__ == "__main__":
    main()
