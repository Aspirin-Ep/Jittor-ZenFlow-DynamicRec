#!/usr/bin/env python3
"""Recreate B-board predictions by loading the supplied production checkpoints.

No supervised model is fitted by this entry point.  It rebuilds only the
compact CSV representation and deterministic inference indices required to
evaluate the saved Dataset3/Dataset4 models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoints() -> None:
    """Fail before inference if any archived production weight was modified."""
    root = ROOT / "checkpoints"
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(f"Checkpoint checksum list is missing: {manifest}")
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {relative}")
        checked += 1
    print(f"verified {checked} archived checkpoint files", flush=True)


def link_or_copy(source: Path, destination: Path) -> None:
    """Expose an immutable archived model at the runtime path expected by code."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == source.stat().st_size:
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def copy_tree_files(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_file():
            link_or_copy(path, destination / path.relative_to(source))


def write_normalized_csv(values: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        for start in range(0, len(values), 10_000):
            block = np.asarray(values[start:start + 10_000], dtype=np.float32)
            low = block.min(axis=1, keepdims=True)
            high = block.max(axis=1, keepdims=True)
            block = (block - low) / np.maximum(high - low, 1e-12)
            np.savetxt(handle, block, delimiter=",", fmt="%.8f")


def package_submission(output_dir: Path) -> Path:
    """Validate the two official CSVs and create the required result.zip."""
    from solution.common.scoring import validate_score_csv

    targets = {
        "dataset3": output_dir / "dataset3.csv",
        "dataset4": output_dir / "dataset4.csv",
    }
    expected_rows = {"dataset3": 157_670, "dataset4": 2_322_538}
    for name, path in targets.items():
        if not path.is_file():
            raise FileNotFoundError(f"Cannot package: missing {path}")
        validate_score_csv(path, expected_rows[name], 100)
    result_zip = output_dir / "result.zip"
    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in targets.items():
            archive.write(path, f"{name}.csv")
    manifest = {
        "method": "checkpoint-inference",
        **{f"{name}_sha256": sha256(path) for name, path in targets.items()},
        "zip_sha256": sha256(result_zip),
    }
    (output_dir / "manifest-b.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"checkpoint inference package complete: {result_zip}", flush=True)
    return result_zip


def dataset3_gcn_tag(data: dict[str, object], config: dict[str, object]) -> str:
    """Return the exact Dataset3 GCN cache name used by ``training.score``."""
    digest = hashlib.sha256()
    for key in ("hist_src", "hist_dst", "hist_time"):
        values = np.asarray(data[key])
        digest.update(np.ascontiguousarray(values).view(np.uint8))
    digest.update((CODE_ROOT / "solution/dataset1/graph_embeddings.py").read_bytes())
    return (
        f"dataset3_full_{len(np.asarray(data['hist_src']))}_"
        f"d{config['gcn_dim']}L{config['gcn_layers']}s{config['gcn_steps']}"
        f"_seed{config['gcn_seed']}_{digest.hexdigest()[:12]}"
    )


def infer_dataset3(args: argparse.Namespace) -> Path:
    import lightgbm as lgb

    from solution.common.scoring import validate_score_csv
    from solution.dataset1.training import score
    from solution.dataset3.data import d3_real

    config: dict[str, object] = {
        "dataset": "dataset3", "struct": True, "gcn": True,
        "neg": 24, "rounds": 900, "early_stop_fraction": 0.15,
        "early_stopping_rounds": 50, "threads": args.workers,
        "feature_workers": min(4, args.workers), "gcn_dim": 64,
        "gcn_layers": 3, "gcn_steps": 1500, "gcn_seed": args.seed,
    }
    data = d3_real()
    tag = dataset3_gcn_tag(data, config)
    archived_gcn = next((ROOT / "checkpoints/dataset3/gcn").glob("*.npz"), None)
    if archived_gcn is None:
        raise FileNotFoundError("Dataset3 archived GCN checkpoint is missing")
    link_or_copy(archived_gcn, ROOT / "outputs/gcn" / f"{tag}.npz")
    model_path = ROOT / "checkpoints/dataset3/ranker.txt"
    booster = lgb.Booster(model_file=str(model_path))
    score_path = args.work_dir / "dataset3/raw_scores.npy"
    values, _ = score(
        config, booster, data, block=4096, output_path=score_path,
        resume_signature={"mode": "checkpoint-inference-v1", "model": sha256(model_path)},
    )
    output = args.output_dir / "dataset3.csv"
    write_normalized_csv(values, output)
    validate_score_csv(output, len(values), 100)
    print(f"Dataset3 checkpoint inference complete: {output}", flush=True)
    return output


def _checkpoint_runtime_layout(runtime: Path) -> None:
    checkpoint = ROOT / "checkpoints/dataset4"
    link_or_copy(checkpoint / "final_ranker.txt", runtime / "final_ranker.txt")
    link_or_copy(checkpoint / "craft/craft_best.pkl", runtime / "craft/full/craft_best.pkl")
    copy_tree_files(checkpoint / "components", runtime / "components")
    copy_tree_files(checkpoint / "context_models", runtime / "context_models")


def infer_dataset4(args: argparse.Namespace) -> Path:
    import lightgbm as lgb

    from solution.common.prepared_data import prepare
    from solution.dataset4.channels import SketchCollaborative
    from solution.dataset4.craft import CraftChannel, TemporalIndex
    from solution.dataset4.model import load_graph_mf_checkpoint, propagate_lightgcn
    from solution.dataset4.pipeline import candidate_frequency, infer, load_prepared

    runtime = args.work_dir / "dataset4"
    _checkpoint_runtime_layout(runtime)
    prepared = runtime / "prepared"
    prepare("dataset4", prepared, chunksize=500_000)
    data = load_prepared(prepared)
    metadata = data["metadata"]
    if int(np.max(data["train_time"])) > int(np.min(data["test_time"])):
        raise ValueError("Dataset4 test query precedes the end of training history")

    expected_data_shape = {
        "num_sources": 680_640,
        "num_destinations": 862_246,
        "train_rows": 16_408_399,
    }
    for name, expected in expected_data_shape.items():
        actual = metadata[name]
        if int(actual) != expected:
            raise ValueError(
                f"Dataset4 checkpoint/data mismatch for {name}: {actual} != {expected}"
            )
    mf_source, mf_destination, mf_bias = load_graph_mf_checkpoint(
        ROOT / "checkpoints/dataset4/graph_mf/graph_mf.pkl",
        int(metadata["num_sources"]), int(metadata["num_destinations"]), 64,
    )
    cache_signature = {
        "mode": "checkpoint-inference-v1", "model": sha256(
            ROOT / "checkpoints/dataset4/graph_mf/graph_mf.pkl"
        ), "layers": 2,
    }
    gcn_source, gcn_destination = propagate_lightgcn(
        data["train_src"], data["train_dst"], mf_source, mf_destination, 2,
        cache_dir=runtime / "representations/full", cache_signature=cache_signature,
    )
    representations = (mf_source, mf_destination, mf_bias, gcn_source, gcn_destination)

    temporal_index = TemporalIndex.build(
        data["train_src"], data["train_dst"], data["train_time"],
        int(metadata["num_sources"]), int(metadata["num_destinations"]), 1,
        runtime / "craft/indices/full",
        {"mode": "checkpoint-inference-v1", "data": metadata["source_signature"]},
    )
    craft = CraftChannel(
        runtime / "craft/full/craft_best.pkl", temporal_index,
        int(metadata["num_sources"]), int(metadata["num_destinations"]),
        neighbors=30, batch_size=args.craft_inference_batch,
    )
    collaborative = SketchCollaborative.build(
        data["train_src"], data["train_dst"], data["train_time"],
        int(metadata["num_sources"]), int(metadata["num_destinations"]),
        runtime / "collaborative/full",
        {"mode": "checkpoint-inference-v1", "data": metadata["source_signature"]},
        dim=args.cf_dim, recent_fraction=0.20, seed=args.seed,
    )
    frequency = candidate_frequency(
        data["test_candidates_raw"], runtime / "candidate_frequency.npy",
        args.chunk_rows,
        {"mode": "checkpoint-inference-v1", "data": metadata["source_signature"]},
        args.workers,
    )
    output = args.output_dir / "dataset4.csv"
    inference_args = argparse.Namespace(
        output=output, output_dir=runtime, workers=args.workers,
        chunk_rows=args.chunk_rows, max_test_rows=args.max_test_rows,
        dim=64, epochs=5, layers=2, selected_graph_epochs=5,
        batch_size=32768, negatives=5, learning_rate=0.002,
        graph_patience=2, graph_min_delta=1e-4, selected_craft_epochs=20,
        craft_epoch_edges=2_000_000, craft_validation_edges=100_000,
        craft_batch_size=256, craft_patience=4, craft_neighbors=30,
        cf_dim=args.cf_dim, cf_recent_fraction=0.20, component_rounds=400,
        context_rounds=300, algorithm_signature="checkpoint-inference-v1",
    )
    ranker = lgb.Booster(model_file=str(runtime / "final_ranker.txt"))
    infer(data, frequency, representations, craft, collaborative, ranker, inference_args)
    print(f"Dataset4 checkpoint inference complete: {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate B-board predictions from archived checkpoints without training."
    )
    parser.add_argument("--target", choices=("dataset3", "dataset4", "b"), default="b")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-rows", type=int, default=2500)
    parser.add_argument("--cf-dim", type=int, default=128)
    parser.add_argument("--craft-inference-batch", type=int, default=256)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument(
        "--package-only", action="store_true",
        help="Validate existing dataset3.csv/dataset4.csv and create result.zip without inference.",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--work-dir", type=Path, default=ROOT / "outputs/checkpoint_inference",
        help="Restartable derived arrays and inference caches (not model training output).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "submissions/checkpoint_inference",
    )
    args = parser.parse_args()
    if args.workers <= 0 or args.chunk_rows <= 0 or args.cf_dim <= 0:
        raise ValueError("workers, chunk-rows and cf-dim must be positive")
    if args.package_only:
        package_submission(args.output_dir)
        return
    from solution.common.datasets import validate_headers
    from solution.common.runtime import configure_cpu_threads

    datasets = ["dataset3", "dataset4"] if args.target == "b" else [args.target]
    missing = [
        ROOT / dataset / name for dataset in datasets for name in ("train.csv", "test.csv")
        if not (ROOT / dataset / name).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing official input files:\n" + "\n".join(map(str, missing)))
    validate_headers(ROOT, datasets)
    configure_cpu_threads(args.workers)
    verify_checkpoints()
    if args.target in {"dataset3", "b"}:
        infer_dataset3(args)
    if args.target in {"dataset4", "b"}:
        infer_dataset4(args)
    if args.target == "b" and args.max_test_rows == 0:
        package_submission(args.output_dir)


if __name__ == "__main__":
    main()
