"""Dataset2 base ensemble inference with fixed blending weights."""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import jittor as jt
import lightgbm as lgb
import numpy as np
import pandas as pd

from .graph_mf import BipartiteMF, mf_scores
from .lightgcn import candidate_scores, normalized_bipartite, propagate
from .temporal_features import FEATURE_NAMES, FrozenTemporalFeatures


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[2]


def normalize_rows(values):
    low = values.min(axis=1, keepdims=True)
    high = values.max(axis=1, keepdims=True)
    return (values - low) / np.maximum(high - low, 1e-12)


def write_rows(handle, values):
    for row in values:
        handle.write(",".join(f"{value:.8f}" for value in row) + "\n")


def validate_csv(path, expected_rows=153420):
    rows = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            values = np.fromstring(line, sep=",", dtype=np.float64)
            if len(values) != 100 or not np.isfinite(values).all():
                raise ValueError(f"{path}: invalid row {rows + 1}")
            if values.min() < 0 or values.max() > 1:
                raise ValueError(f"{path}: score outside [0, 1]")
            rows += 1
    if rows != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, got {rows}")
    return True


def run_command(arguments):
    print("+", " ".join(map(str, arguments)))
    subprocess.run(list(map(str, arguments)), check=True)


def ensure_classical_components(
    data_root, component_dir, model_dir, seed, force=False
):
    component_dir.mkdir(parents=True, exist_ok=True)
    craft_root = component_dir / "craft"
    paths = {
        "craft": craft_root / "dataset2" / "dataset2_result.csv",
        "stat": component_dir / "stat.csv",
        "tree": component_dir / "tree.csv",
    }
    commands = {
        "craft": [
            sys.executable, "-m", "solution.base.dataset2_craft",
            "--dataset", "dataset2",
            "--data_dir", data_root,
            "--model", model_dir / "craft.pkl",
            "--output_dir", craft_root,
            "--batch_size", "256",
            "--seed", seed,
        ],
        "stat": [
            sys.executable, "-m", "solution.base.dataset2_stat",
            "--data_dir", data_root,
            "--output_path", paths["stat"],
        ],
        "tree": [
            sys.executable, "-m", "solution.base.dataset2_tree",
            "--data_dir", data_root,
            "--model", model_dir / "tree_ranker.txt",
            "--output_path", paths["tree"],
            "--verify_rows", "0",
        ],
    }
    for name in ("craft", "stat", "tree"):
        path = paths[name]
        if path.exists() and not force:
            validate_csv(path)
            print(f"Reusing {name}: {path}")
            continue
        run_command(commands[name])
        validate_csv(path)
    return paths


def neural_components(
    data_root, component_dir, model_dir, limit=None
):
    frame = pd.read_csv(Path(data_root) / "dataset2" / "train.csv").sort_values(
        "time", kind="stable"
    ).reset_index(drop=True)
    full_test = pd.read_csv(Path(data_root) / "dataset2" / "test.csv")
    test = full_test if limit is None else full_test.iloc[:limit].copy()
    full_candidates = full_test.iloc[:, 2:].to_numpy(dtype=np.int64)
    candidates = test.iloc[:, 2:].to_numpy(dtype=np.int32)
    sources = test["src"].to_numpy(dtype=np.int32)
    times = test["time"].to_numpy(dtype=np.float64)
    frequency = Counter(map(int, full_candidates.reshape(-1)))
    num_src = int(frame["src"].max()) + 1
    num_dst = int(max(frame["dst"].max(), full_candidates.max())) + 1

    extractor = FrozenTemporalFeatures(frame)
    graph_model = BipartiteMF(num_src, num_dst, 64)
    graph_model.load_state_dict(jt.load(str(model_dir / "graph_mf.pkl")))
    temporal_ranker = lgb.Booster(
        model_file=str(model_dir / "temporal_ranker.txt")
    )
    graph_ranker = lgb.Booster(
        model_file=str(model_dir / "graph_ranker.txt")
    )
    lightgcn_ranker = lgb.Booster(
        model_file=str(model_dir / "lightgcn_ranker.txt")
    )
    if temporal_ranker.feature_name() != FEATURE_NAMES:
        raise ValueError("temporal ranker feature schema mismatch")
    if graph_ranker.feature_name() != FEATURE_NAMES + ["graph_mf_score"]:
        raise ValueError("GraphMF ranker feature schema mismatch")
    if lightgcn_ranker.feature_name() != FEATURE_NAMES + ["lightgcn_score"]:
        raise ValueError("LightGCN ranker feature schema mismatch")

    adjacency = normalized_bipartite(frame, num_src, num_dst)
    src_layers, dst_layers = propagate(
        adjacency,
        graph_model.src.weight.numpy(),
        graph_model.dst.weight.numpy(),
        2,
    )
    light_src = np.mean(src_layers, axis=0).astype(np.float32)
    light_dst = np.mean(dst_layers, axis=0).astype(np.float32)
    bias = graph_model.dst_bias.weight.numpy().reshape(-1)

    component_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "temporal": component_dir / "temporal.csv",
        "graph": component_dir / "graph.csv",
        "lightgcn": component_dir / "lightgcn.csv",
    }
    handles = {
        name: open(path, "w", encoding="utf-8")
        for name, path in paths.items()
    }
    try:
        for start in range(0, len(test), 250):
            stop = min(start + 250, len(test))
            rows = []
            for row in range(start, stop):
                rows.extend(
                    extractor.row(
                        int(sources[row]), int(candidate), float(times[row]),
                        frequency,
                    )
                    for candidate in candidates[row]
                )
            base = np.asarray(rows, dtype=np.float32).reshape(
                stop - start, 100, -1
            )
            flat = base.reshape(-1, base.shape[-1])
            temporal = temporal_ranker.predict(
                flat, num_iteration=temporal_ranker.best_iteration
            ).reshape(stop - start, 100)
            graph_latent = mf_scores(
                graph_model,
                sources[start:stop],
                candidates[start:stop],
                batch_size=250,
            )
            graph = graph_ranker.predict(
                np.concatenate(
                    [base, graph_latent[..., None]], axis=-1
                ).reshape(-1, base.shape[-1] + 1)
            ).reshape(stop - start, 100)
            light_latent = candidate_scores(
                light_src, light_dst, bias,
                sources[start:stop], candidates[start:stop],
            )
            lightgcn = lightgcn_ranker.predict(
                np.concatenate(
                    [base, light_latent[..., None]], axis=-1
                ).reshape(-1, base.shape[-1] + 1)
            ).reshape(stop - start, 100)
            for name, values in (
                ("temporal", temporal),
                ("graph", graph),
                ("lightgcn", lightgcn),
            ):
                write_rows(handles[name], normalize_rows(values))
            if start == 0 or stop == len(test) or stop % 10000 == 0:
                print(f"Dataset2 neural inference {stop}/{len(test)}")
    finally:
        for handle in handles.values():
            handle.close()
    return paths


def blend_files(inputs, weights, output_path, expected_rows=153420):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handles = [
        open(Path(path), "r", encoding="utf-8") for path in inputs
    ]
    rows = 0
    try:
        with open(output_path, "w", encoding="utf-8") as output:
            while True:
                lines = [handle.readline() for handle in handles]
                if not lines[0]:
                    if any(lines[1:]):
                        raise ValueError("component row counts differ")
                    break
                if any(not line for line in lines):
                    raise ValueError("component row counts differ")
                arrays = [
                    np.fromstring(line, sep=",", dtype=np.float64)
                    for line in lines
                ]
                if any(len(values) != 100 for values in arrays):
                    raise ValueError(f"invalid component row {rows + 1}")
                blended = sum(
                    weight * values
                    for weight, values in zip(weights, arrays)
                )
                low, high = float(blended.min()), float(blended.max())
                blended = (blended - low) / max(high - low, 1e-12)
                output.write(
                    ",".join(f"{value:.8f}" for value in blended) + "\n"
                )
                rows += 1
    finally:
        for handle in handles:
            handle.close()
    if rows != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {rows}")
    print(f"Saved blend: {output_path}")
    return output_path


def assemble(classical, neural, work_dir, output_path, expected_rows=153420):
    blend_dir = Path(work_dir) / "blends"
    classical_blend = blend_files(
        [classical["tree"], classical["craft"], classical["stat"]],
        [0.50, 0.30, 0.20],
        blend_dir / "classical.csv",
        expected_rows,
    )
    temporal_blend = blend_files(
        [neural["temporal"], classical_blend],
        [0.65, 0.35],
        blend_dir / "temporal.csv",
        expected_rows,
    )
    graph_blend = blend_files(
        [temporal_blend, neural["graph"]],
        [0.50, 0.50],
        blend_dir / "graph.csv",
        expected_rows,
    )
    return blend_files(
        [graph_blend, neural["lightgcn"]],
        [0.65, 0.35],
        output_path,
        expected_rows,
    )


def run(data_root, output_path, work_dir, model_dir, seed, force=False):
    started = time.time()
    work_dir = Path(work_dir)
    model_dir = Path(model_dir)
    component_dir = work_dir / "components"
    classical = ensure_classical_components(
        data_root, component_dir, model_dir, seed, force=force
    )
    neural_paths = {
        name: component_dir / f"{name}.csv"
        for name in ("temporal", "graph", "lightgcn")
    }
    reuse_neural = (
        not force
        and all(path.exists() for path in neural_paths.values())
    )
    if reuse_neural:
        for path in neural_paths.values():
            validate_csv(path)
        print("Reusing Dataset2 neural component predictions")
    else:
        neural_paths = neural_components(
            data_root, component_dir, model_dir
        )
        for path in neural_paths.values():
            validate_csv(path)
    final_path = assemble(
        classical, neural_paths, work_dir, output_path
    )
    validate_csv(final_path)
    report = {
        "dataset": "dataset2",
        "architecture": "temporal, graph and LightGCN base ensemble",
        "formula": (
            "classical(0.50 tree + 0.30 CRAFT + 0.20 stat) -> "
            "temporal(0.65 temporal + 0.35 classical) -> "
            "graph(0.50 temporal + 0.50 GraphMF) -> "
            "final(0.65 graph + 0.35 LightGCN)"
        ),
        "rows": 153420,
        "candidates": 100,
        "output": str(final_path),
        "runtime_minutes": (time.time() - started) / 60.0,
        "validated": True,
    }
    report_path = Path(output_path).with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return final_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", default=str(PROJECT_ROOT)
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs/dataset2/production/base/dataset2.csv"),
    )
    parser.add_argument(
        "--work-dir",
        default=str(PROJECT_ROOT / "outputs/dataset2/production/base/components"),
    )
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "outputs/dataset2/checkpoints/base_models"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--neural-smoke", type=int, default=0,
        help="Only infer the first N neural rows for implementation checks.",
    )
    args = parser.parse_args()
    if args.neural_smoke:
        neural_components(
            args.data_root,
            Path(args.work_dir) / "smoke",
            Path(args.model_dir),
            limit=args.neural_smoke,
        )
        return
    run(
        args.data_root,
        args.output,
        args.work_dir,
        args.model_dir,
        args.seed,
        args.force,
    )


if __name__ == "__main__":
    main()
