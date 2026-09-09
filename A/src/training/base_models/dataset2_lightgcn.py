"""Train the promoted two-layer LightGCN feature and predict Dataset2 test."""

import argparse
import json
import os
import time

import jittor as jt
import numpy as np
import pandas as pd

from solution.base.temporal_features import FEATURE_NAMES, FrozenTemporalFeatures
from solution.base.candidate_sampling import (
    CompetitionCandidateSampler,
    eligible_new_link_indices,
    sample_indices,
)
from solution.base.graph_mf import (
    BipartiteMF,
    build_groups,
    train_lgbm_fixed,
    train_mf,
)
from solution.base.lightgcn import (
    candidate_scores,
    normalized_bipartite,
    propagate,
)
from solution.base.ranking import append_experiment


def load_or_build_official_groups(
    frame, split0_end, test_candidates, count, seed, cache_path
):
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        print(f"Loaded official groups: {cache_path}")
        return tuple(cached[key] for key in (
            "x", "y", "sources", "candidates"
        ))
    rng = np.random.default_rng(seed)
    extractor = FrozenTemporalFeatures(frame.iloc[:split0_end])
    pool = eligible_new_link_indices(
        frame, split0_end, len(frame), extractor
    )
    indices = sample_indices(pool, count, rng)
    sampler = CompetitionCandidateSampler(test_candidates, extractor)
    print(f"Building {len(indices)} official ranking groups...")
    values = build_groups(
        extractor, frame, indices, sampler, 100, rng
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        x=values[0], y=values[1],
        sources=values[2], candidates=values[3],
    )
    print(f"Saved official groups: {cache_path}")
    return values


def predict_test(
    extractor, ranker, test, frequency,
    src_embedding, dst_embedding, bias, output_path, batch_rows,
):
    candidates = test.iloc[:, 2:].to_numpy(dtype=np.int32)
    sources = test["src"].to_numpy(dtype=np.int32)
    times = test["time"].to_numpy(dtype=np.float64)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for start in range(0, len(test), batch_rows):
            stop = min(start + batch_rows, len(test))
            rows = []
            for row in range(start, stop):
                rows.extend(
                    extractor.row(
                        int(sources[row]), int(candidate), float(times[row]),
                        frequency,
                    )
                    for candidate in candidates[row]
                )
            features = np.asarray(rows, dtype=np.float32).reshape(
                stop - start, 100, -1
            )
            latent = candidate_scores(
                src_embedding, dst_embedding, bias,
                sources[start:stop], candidates[start:stop],
            )
            augmented = np.concatenate(
                [features, latent[..., None]], axis=-1
            )
            scores = ranker.predict(
                augmented.reshape(-1, augmented.shape[-1])
            ).reshape(stop - start, 100)
            low = scores.min(axis=1, keepdims=True)
            high = scores.max(axis=1, keepdims=True)
            scores = (scores - low) / np.maximum(high - low, 1e-12)
            for values in scores:
                handle.write(
                    ",".join(f"{value:.8f}" for value in values) + "\n"
                )
            if start == 0 or stop == len(test) or stop % 5000 == 0:
                print(f"Predicted {stop}/{len(test)}")


def propagated_average(model, adjacency):
    src_layers, dst_layers = propagate(
        adjacency,
        model.src.weight.numpy(),
        model.dst.weight.numpy(),
        2,
    )
    return (
        np.mean(src_layers, axis=0).astype(np.float32),
        np.mean(dst_layers, axis=0).astype(np.float32),
        model.dst_bias.weight.numpy().reshape(-1),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".")
    parser.add_argument(
        "--output_dir", default="outputs/training/dataset2/lightgcn"
    )
    parser.add_argument(
        "--graph_model",
        default="outputs/training/dataset2/graph/dataset2/full_graph_mf.pkl",
    )
    parser.add_argument("--cache_path", default=None)
    parser.add_argument("--final_queries", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--negatives", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--prediction_batch", type=int, default=250)
    args = parser.parse_args()
    started = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    jt.set_global_seed(args.seed)

    frame = pd.read_csv(
        os.path.join(args.data_dir, "dataset2", "train.csv")
    ).sort_values("time", kind="stable").reset_index(drop=True)
    test = pd.read_csv(os.path.join(args.data_dir, "dataset2", "test.csv"))
    test_candidates = test.iloc[:, 2:].to_numpy(dtype=np.int32)
    split0_end = int(np.flatnonzero(frame["split"].to_numpy() == 1)[0])
    num_src = int(frame["src"].max()) + 1
    num_dst = int(max(frame["dst"].max(), test_candidates.max())) + 1

    cache_path = args.cache_path or os.path.join(
        args.output_dir, "cache",
        f"dataset2_official_q{args.final_queries}_s{args.seed}.npz",
    )
    x_final, y_final, src_final, candidates_final = (
        load_or_build_official_groups(
            frame, split0_end, test_candidates,
            args.final_queries, args.seed, cache_path,
        )
    )

    split0_path = os.path.join(args.output_dir, "split0_graph_mf.pkl")
    split0_model = BipartiteMF(num_src, num_dst, 64)
    if os.path.exists(split0_path):
        split0_model.load_state_dict(jt.load(split0_path))
        print(f"Loaded split0 GraphMF: {split0_path}")
    else:
        train_mf(split0_model, frame, split0_end, args)
        jt.save(split0_model.state_dict(), split0_path)
    split0_adjacency = normalized_bipartite(
        frame.iloc[:split0_end], num_src, num_dst
    )
    split0_src, split0_dst, split0_bias = propagated_average(
        split0_model, split0_adjacency
    )
    lightgcn_final = candidate_scores(
        split0_src, split0_dst, split0_bias,
        src_final, candidates_final,
    )
    augmented_final = np.concatenate(
        [x_final, lightgcn_final[..., None]], axis=-1
    )
    ranker = train_lgbm_fixed(
        augmented_final, y_final,
        FEATURE_NAMES + ["lightgcn_score"],
        rounds=80, seed=args.seed,
    )
    ranker_path = os.path.join(args.output_dir, "final_ranker.txt")
    ranker.save_model(ranker_path)
    del split0_adjacency, split0_src, split0_dst, lightgcn_final

    full_model = BipartiteMF(num_src, num_dst, 64)
    full_model.load_state_dict(
        jt.load(args.graph_model)
    )
    full_adjacency = normalized_bipartite(
        frame, num_src, num_dst
    )
    full_src, full_dst, full_bias = propagated_average(
        full_model, full_adjacency
    )
    extractor = FrozenTemporalFeatures(frame)
    sampler = CompetitionCandidateSampler(test_candidates, extractor)
    prediction_path = os.path.join(
        args.output_dir, "dataset2_lightgcn_scores.csv"
    )
    predict_test(
        extractor, ranker, test, sampler.frequency,
        full_src, full_dst, full_bias, prediction_path,
        args.prediction_batch,
    )
    report = {
        "dataset": "dataset2",
        "architecture": "two_layer_lightgcn",
        "final_queries": len(x_final),
        "ranker_rounds": 80,
        "prediction_path": prediction_path,
        "rows": len(test),
        "runtime_minutes": (time.time() - started) / 60.0,
    }
    with open(
        os.path.join(args.output_dir, "report.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    append_experiment(
        os.path.join(args.output_dir, "training_log.jsonl"),
        {
            "experiment_id": f"lightgcn_full_seed{args.seed}",
            "dataset": "dataset2",
            "architecture": "two_layer_lightgcn",
            "config": {"layers": 2, "ranker_rounds": 80},
            "train_queries": len(x_final),
            "valid_queries": 0,
            "runtime_minutes": report["runtime_minutes"],
            "status": "completed",
            "decision": "candidate",
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
