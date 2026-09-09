"""GraphMF: Jittor bipartite graph matrix factorization as a ranking feature."""

import argparse
import json
import math
import os

import jittor as jt
from jittor import nn
import lightgbm as lgb
import numpy as np
import pandas as pd

from solution.base.temporal_features import FEATURE_NAMES, FrozenTemporalFeatures
from solution.base.candidate_sampling import (
    CompetitionCandidateSampler,
    eligible_new_link_indices,
    sample_indices,
)


def mrr_from_group_arrays(labels, scores):
    positive = labels.argmax(axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.argmax(order == positive[:, None], axis=1) + 1
    return float(np.mean(1.0 / ranks))


jt.flags.use_cuda = 1


class BipartiteMF(nn.Module):
    def __init__(self, num_src, num_dst, dim):
        super().__init__()
        self.src = nn.Embedding(num_src, dim)
        self.dst = nn.Embedding(num_dst, dim)
        self.dst_bias = nn.Embedding(num_dst, 1)
        nn.init.xavier_uniform_(self.src.weight)
        nn.init.xavier_uniform_(self.dst.weight)
        self.dst_bias.weight.assign(jt.zeros_like(self.dst_bias.weight))

    def execute(self, src, dst):
        return (
            self.src(src) * self.dst(dst)
        ).sum(dim=-1) + self.dst_bias(dst).squeeze(-1)


def build_groups(extractor, frame, indices, sampler, num_candidates, rng):
    features, labels, sources, candidates_all = [], [], [], []
    for idx in indices:
        row = frame.iloc[int(idx)]
        src, positive, query_time = int(row.src), int(row.dst), float(row.time)
        candidates = sampler.sample(
            src, positive, extractor, num_candidates, rng
        )
        features.append(np.asarray([
            extractor.row(src, candidate, query_time, sampler.frequency)
            for candidate in candidates
        ], dtype=np.float32))
        labels.append(np.asarray(
            [int(candidate == positive) for candidate in candidates],
            dtype=np.int8,
        ))
        sources.append(src)
        candidates_all.append(np.asarray(candidates, dtype=np.int32))
    return (
        np.asarray(features), np.asarray(labels),
        np.asarray(sources, dtype=np.int32),
        np.asarray(candidates_all, dtype=np.int32),
    )


def build_or_load_groups(args):
    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(
        cache_dir,
        f"groups_t{args.train_queries}_v{args.valid_queries}_"
        f"c{args.num_candidates}_s{args.seed}.npz",
    )
    frame = pd.read_csv(
        os.path.join(args.data_dir, "dataset2", "train.csv")
    ).sort_values("time", kind="stable").reset_index(drop=True)
    split0_end = int(np.flatnonzero(frame["split"].to_numpy() == 1)[0])
    internal_end = int(split0_end * args.internal_cutoff)
    if os.path.exists(path) and not args.rebuild_cache:
        data = np.load(path)
        values = tuple(data[key] for key in (
            "x_train", "y_train", "s_train", "c_train",
            "x_valid", "y_valid", "s_valid", "c_valid",
        ))
        return frame, internal_end, values

    rng = np.random.default_rng(args.seed)
    test = pd.read_csv(os.path.join(args.data_dir, "dataset2", "test.csv"))
    test_candidates = test.iloc[:, 2:].to_numpy(dtype=np.int64)
    train_extractor = FrozenTemporalFeatures(frame.iloc[:internal_end])
    train_pool = eligible_new_link_indices(
        frame, internal_end, split0_end, train_extractor
    )
    train_indices = sample_indices(train_pool, args.train_queries, rng)
    train_sampler = CompetitionCandidateSampler(test_candidates, train_extractor)
    train_values = build_groups(
        train_extractor, frame, train_indices, train_sampler,
        args.num_candidates, rng,
    )
    valid_extractor = FrozenTemporalFeatures(frame.iloc[:split0_end])
    valid_pool = eligible_new_link_indices(
        frame, split0_end, len(frame), valid_extractor
    )
    valid_indices = sample_indices(valid_pool, args.valid_queries, rng)
    valid_sampler = CompetitionCandidateSampler(test_candidates, valid_extractor)
    valid_values = build_groups(
        valid_extractor, frame, valid_indices, valid_sampler,
        args.num_candidates, rng,
    )
    np.savez(
        path,
        x_train=train_values[0], y_train=train_values[1],
        s_train=train_values[2], c_train=train_values[3],
        x_valid=valid_values[0], y_valid=valid_values[1],
        s_valid=valid_values[2], c_valid=valid_values[3],
    )
    return frame, internal_end, train_values + valid_values


def train_mf(model, frame, internal_end, args):
    src = frame.iloc[:internal_end]["src"].to_numpy(dtype=np.int32)
    dst = frame.iloc[:internal_end]["dst"].to_numpy(dtype=np.int32)
    seen_dst = np.unique(dst)
    rng = np.random.default_rng(args.seed)
    optimizer = nn.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(src))
        losses = []
        model.train()
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            # Draw from edge destinations (popularity-weighted) and optimize
            # against the hardest sampled negative.
            negative = rng.choice(
                dst,
                size=(len(idx), args.negatives),
                replace=True,
            ).astype(np.int32)
            src_jt = jt.array(src[idx])
            positive_score = model(src_jt, jt.array(dst[idx]))
            expanded_src = np.repeat(
                src[idx, None], args.negatives, axis=1
            )
            negative_score = model(
                jt.array(expanded_src), jt.array(negative)
            ).max(dim=1)
            difference = jt.clamp(
                positive_score - negative_score, -30.0, 30.0
            )
            loss = -jt.log(jt.sigmoid(difference) + 1e-8).mean()
            optimizer.step(loss)
            losses.append(float(loss.item()))
        print(f"MF epoch {epoch:02d}: loss={np.mean(losses):.5f}")


def mf_scores(model, sources, candidates, batch_size=256):
    model.eval()
    output = []
    with jt.no_grad():
        for start in range(0, len(sources), batch_size):
            stop = start + batch_size
            src = np.repeat(
                sources[start:stop, None], candidates.shape[1], axis=1
            )
            output.append(model(
                jt.array(src), jt.array(candidates[start:stop])
            ).numpy())
    return np.concatenate(output)


def lgbm_score(x_train, y_train, x_valid, y_valid, names, seed):
    train_groups = np.full(len(x_train), x_train.shape[1], dtype=np.int32)
    valid_groups = np.full(len(x_valid), x_valid.shape[1], dtype=np.int32)
    train = lgb.Dataset(
        x_train.reshape(-1, x_train.shape[-1]),
        label=y_train.reshape(-1), group=train_groups, feature_name=names,
    )
    valid = lgb.Dataset(
        x_valid.reshape(-1, x_valid.shape[-1]),
        label=y_valid.reshape(-1), group=valid_groups,
        feature_name=names, reference=train,
    )
    model = lgb.train(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 100,
            "verbosity": -1,
            "seed": seed,
        },
        train, num_boost_round=300, valid_sets=[valid],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    score = model.predict(
        x_valid.reshape(-1, x_valid.shape[-1])
    ).reshape(y_valid.shape)
    return mrr_from_group_arrays(y_valid, score), model


def train_lgbm_fixed(x, y, names, rounds, seed):
    groups = np.full(len(x), x.shape[1], dtype=np.int32)
    dataset = lgb.Dataset(
        x.reshape(-1, x.shape[-1]),
        label=y.reshape(-1), group=groups, feature_name=names,
    )
    return lgb.train(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 100,
            "verbosity": -1,
            "seed": seed,
        },
        dataset, num_boost_round=max(int(rounds), 10),
    )


def predict_test(
    extractor, mf_model, ranker, test, frequency, output_path,
    batch_rows=250,
):
    candidates = test.iloc[:, 2:].to_numpy(dtype=np.int32)
    sources = test["src"].to_numpy(dtype=np.int32)
    times = test["time"].to_numpy(dtype=np.float64)
    feature_names = FEATURE_NAMES + ["graph_mf_score"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for start in range(0, len(test), batch_rows):
            stop = min(start + batch_rows, len(test))
            feature_rows = []
            for row in range(start, stop):
                feature_rows.extend(
                    extractor.row(
                        int(sources[row]), int(candidate), float(times[row]),
                        frequency,
                    )
                    for candidate in candidates[row]
                )
            features = np.asarray(feature_rows, dtype=np.float32).reshape(
                stop - start, 100, -1
            )
            latent = mf_scores(
                mf_model, sources[start:stop], candidates[start:stop],
                batch_size=batch_rows,
            )
            augmented = np.concatenate([features, latent[..., None]], axis=-1)
            scores = ranker.predict(
                augmented.reshape(-1, augmented.shape[-1])
            ).reshape(stop - start, 100)
            low = scores.min(axis=1, keepdims=True)
            high = scores.max(axis=1, keepdims=True)
            scores = (scores - low) / np.maximum(high - low, 1e-12)
            for values in scores:
                handle.write(",".join(f"{value:.8f}" for value in values) + "\n")
            if start == 0 or stop == len(test) or stop % 10000 == 0:
                print(f"Predicted {stop}/{len(test)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".")
    parser.add_argument(
        "--output_dir", default="outputs/training/dataset2/graph"
    )
    parser.add_argument("--train_queries", type=int, default=5000)
    parser.add_argument("--valid_queries", type=int, default=2000)
    parser.add_argument("--num_candidates", type=int, default=100)
    parser.add_argument("--internal_cutoff", type=float, default=0.8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--negatives", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--predict_test", action="store_true")
    parser.add_argument("--final_queries", type=int, default=20000)
    args = parser.parse_args()

    frame, internal_end, values = build_or_load_groups(args)
    (
        x_train, y_train, src_train, candidates_train,
        x_valid, y_valid, src_valid, candidates_valid,
    ) = values
    num_src = int(frame["src"].max()) + 1
    test_max = pd.read_csv(
        os.path.join(args.data_dir, "dataset2", "test.csv")
    ).iloc[:, 2:].to_numpy().max()
    num_dst = int(max(frame["dst"].max(), test_max)) + 1
    model = BipartiteMF(num_src, num_dst, args.dim)
    train_mf(model, frame, internal_end, args)
    train_prior = mf_scores(model, src_train, candidates_train)
    valid_prior = mf_scores(model, src_valid, candidates_valid)

    baseline, _ = lgbm_score(
        x_train, y_train, x_valid, y_valid, FEATURE_NAMES, args.seed
    )
    x_train_mf = np.concatenate(
        [x_train, train_prior[..., None]], axis=-1
    )
    x_valid_mf = np.concatenate(
        [x_valid, valid_prior[..., None]], axis=-1
    )
    augmented, ranker = lgbm_score(
        x_train_mf, y_train, x_valid_mf, y_valid,
        FEATURE_NAMES + ["graph_mf_score"], args.seed,
    )
    out = os.path.join(args.output_dir, "dataset2")
    os.makedirs(out, exist_ok=True)
    jt.save(model.state_dict(), os.path.join(out, "graph_mf.pkl"))
    ranker.save_model(os.path.join(out, "ranker.txt"))
    report = {
        "dataset": "dataset2",
        "train_queries": len(x_train),
        "valid_queries": len(x_valid),
        "lightgbm_mrr": baseline,
        "lightgbm_graphmf_mrr": augmented,
        "delta": augmented - baseline,
        "mf_dim": args.dim,
        "mf_epochs": args.epochs,
    }
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not args.predict_test:
        return

    # Stage 1: train graph embeddings on split=0 and use known split=1 links
    # to train the final feature ranker.
    split0_end = int(np.flatnonzero(frame["split"].to_numpy() == 1)[0])
    test = pd.read_csv(os.path.join(args.data_dir, "dataset2", "test.csv"))
    test_candidates = test.iloc[:, 2:].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    official_extractor = FrozenTemporalFeatures(frame.iloc[:split0_end])
    official_pool = eligible_new_link_indices(
        frame, split0_end, len(frame), official_extractor
    )
    official_indices = sample_indices(
        official_pool, args.final_queries, rng
    )
    official_sampler = CompetitionCandidateSampler(
        test_candidates, official_extractor
    )
    final_values = build_groups(
        official_extractor, frame, official_indices, official_sampler,
        args.num_candidates, rng,
    )
    x_final, y_final, src_final, candidates_final = final_values

    split0_mf = BipartiteMF(num_src, num_dst, args.dim)
    train_mf(split0_mf, frame, split0_end, args)
    latent_final = mf_scores(
        split0_mf, src_final, candidates_final
    )
    x_final_augmented = np.concatenate(
        [x_final, latent_final[..., None]], axis=-1
    )
    final_ranker = train_lgbm_fixed(
        x_final_augmented, y_final,
        FEATURE_NAMES + ["graph_mf_score"],
        ranker.best_iteration, args.seed,
    )

    # Stage 2: update graph embeddings with every official training edge, then
    # apply the learned ranker to competition candidates.
    full_mf = BipartiteMF(num_src, num_dst, args.dim)
    train_mf(full_mf, frame, len(frame), args)
    final_ranker.save_model(os.path.join(out, "final_ranker.txt"))
    jt.save(full_mf.state_dict(), os.path.join(out, "full_graph_mf.pkl"))
    full_extractor = FrozenTemporalFeatures(frame)
    full_sampler = CompetitionCandidateSampler(test_candidates, full_extractor)
    prediction_path = os.path.join(out, "dataset2_graph_mf_scores.csv")
    predict_test(
        full_extractor, full_mf, final_ranker, test, full_sampler.frequency,
        prediction_path,
    )
    print(f"Saved predictions: {prediction_path}")


if __name__ == "__main__":
    main()
