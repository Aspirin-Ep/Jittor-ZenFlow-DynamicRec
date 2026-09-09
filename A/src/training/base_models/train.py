"""Train the Dataset2 base ensemble from the competition CSV files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "code"
CHECKPOINT_NAMES = (
    "craft.pkl",
    "tree_ranker.txt",
    "temporal_ranker.txt",
    "graph_mf.pkl",
    "graph_ranker.txt",
    "lightgcn_ranker.txt",
)


def run_module(module: str, arguments: list[object]) -> None:
    environment = os.environ.copy()
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(SOURCE_ROOT)
        if not previous
        else str(SOURCE_ROOT) + os.pathsep + previous
    )
    command = [sys.executable, "-m", module, *map(str, arguments)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def export_checkpoint(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint was not produced: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def train(
    data_root: Path,
    output_root: Path,
    checkpoint_root: Path,
    seed: int,
) -> None:
    craft = output_root / "craft"
    tree = output_root / "tree"
    temporal = output_root / "temporal"
    graph = output_root / "graph"
    lightgcn = output_root / "lightgcn"

    run_module(
        "training.base_models.dataset2_craft",
        [
            "--dataset", "dataset2",
            "--data_dir", data_root,
            "--save_dir", craft / "models",
            "--output_dir", craft / "predictions",
            "--epochs", "100",
            "--batch_size", "200",
            "--early_stop", "10",
            "--seed", seed,
        ],
    )
    export_checkpoint(
        craft / "models" / "dataset2_CRAFT_best.pkl",
        checkpoint_root / "craft.pkl",
    )

    run_module(
        "training.base_models.dataset2_tree",
        [
            "--dataset", "dataset2",
            "--data_dir", data_root,
            "--save_dir", tree / "models",
            "--output_dir", tree / "predictions",
            "--n_neg", "5",
            "--n_train", "100000",
            "--n_workers", "2",
            "--seed", seed,
        ],
    )
    export_checkpoint(
        tree / "models" / "dataset2_tree_ranker.txt",
        checkpoint_root / "tree_ranker.txt",
    )

    run_module(
        "training.base_models.dataset2_temporal",
        [
            "--dataset", "dataset2",
            "--data_dir", data_root,
            "--output_dir", temporal,
            "--cutoff", "0.9",
            "--train_queries", "20000",
            "--valid_queries", "5000",
            "--num_candidates", "50",
            "--seed", seed,
            "--predict_test",
        ],
    )
    export_checkpoint(
        temporal / "dataset2" / "cutoff_0p90" / "model.txt",
        checkpoint_root / "temporal_ranker.txt",
    )

    run_module(
        "training.base_models.dataset2_graph_mf",
        [
            "--data_dir", data_root,
            "--output_dir", graph,
            "--train_queries", "5000",
            "--valid_queries", "2000",
            "--num_candidates", "100",
            "--internal_cutoff", "0.8",
            "--dim", "64",
            "--epochs", "5",
            "--batch_size", "4096",
            "--lr", "0.002",
            "--negatives", "10",
            "--seed", seed,
            "--final_queries", "20000",
            "--predict_test",
        ],
    )
    graph_model = graph / "dataset2" / "full_graph_mf.pkl"
    export_checkpoint(graph_model, checkpoint_root / "graph_mf.pkl")
    export_checkpoint(
        graph / "dataset2" / "final_ranker.txt",
        checkpoint_root / "graph_ranker.txt",
    )

    run_module(
        "training.base_models.dataset2_lightgcn",
        [
            "--data_dir", data_root,
            "--output_dir", lightgcn,
            "--graph_model", graph_model,
            "--final_queries", "20000",
            "--epochs", "5",
            "--batch_size", "4096",
            "--lr", "0.002",
            "--negatives", "10",
            "--seed", seed,
            "--prediction_batch", "250",
        ],
    )
    export_checkpoint(
        lightgcn / "final_ranker.txt",
        checkpoint_root / "lightgcn_ranker.txt",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/dataset2/training/base_models",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "outputs/dataset2/checkpoints/base_models",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    required_data = (
        data_root / "dataset2/train.csv",
        data_root / "dataset2/test.csv",
    )
    missing = [path for path in required_data if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing Dataset2 CSV files:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    checkpoints = [checkpoint_root / name for name in CHECKPOINT_NAMES]
    if args.force or not all(path.is_file() for path in checkpoints):
        train(data_root, output_root, checkpoint_root, args.seed)
    print(
        json.dumps(
            {
                "data_root": str(data_root),
                "training_output": str(output_root),
                "checkpoints": [str(path) for path in checkpoints],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
