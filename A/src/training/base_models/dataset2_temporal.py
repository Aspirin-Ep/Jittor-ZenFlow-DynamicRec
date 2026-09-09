"""temporal ranker: candidate-aware temporal learning-to-rank baseline.

This script is deliberately validation-first:

* graph statistics are built only from interactions before a time cutoff;
* each query is a group containing one positive and sampled competition-like
  candidates;
* LightGBM optimizes LambdaRank instead of pointwise binary classification;
* Dataset2 uses typed bipartite item-to-item transitions rather than invalid
  intersections between source IDs and destination IDs.

Example quick experiment:
    python3 -m training.base_models.dataset2_temporal --dataset dataset2 \
        --train_queries 20000 --valid_queries 5000 --num_candidates 50
"""

import argparse
import json
import math
import os
from bisect import bisect_left
from collections import Counter, OrderedDict, defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "pair_count",
    "pair_count_log",
    "pair_recency_log",
    "pair_hawkes_fast",
    "pair_hawkes_medium",
    "pair_hawkes_slow",
    "pair_due_ratio_log",
    "pair_due_error_log",
    "src_degree_log",
    "src_unique_log",
    "dst_degree_log",
    "dst_unique_src_log",
    "dst_recency_log",
    "dst_recent_short",
    "dst_recent_long",
    "dst_recent_ratio",
    "dst_hawkes_fast",
    "dst_hawkes_medium",
    "dst_hawkes_slow",
    "dst_intensity_acceleration",
    "is_seen_dst",
    "is_src_history",
    "src_history_share",
    "last_item_transition",
    "recent_item_transition",
    "recent_item_transition_norm",
    "bipartite_cf_support",
    "bipartite_cf_support_log",
    "bipartite_cf_norm",
    "candidate_frequency_log",
]


class FrozenTemporalFeatures:
    """Features frozen at a cutoff, matching competition-time observability."""

    def __init__(self, prefix):
        self.src_history = defaultdict(list)
        self.dst_times = defaultdict(list)
        self.dst_srcs = defaultdict(set)
        self.dst_recent_srcs = defaultdict(list)
        self.pair_times = defaultdict(list)

        src = prefix["src"].to_numpy(dtype=np.int64)
        dst = prefix["dst"].to_numpy(dtype=np.int64)
        times = prefix["time"].to_numpy(dtype=np.float64)
        for s, d, t in zip(src, dst, times):
            s, d, t = int(s), int(d), float(t)
            self.src_history[s].append((d, t))
            self.dst_times[d].append(t)
            self.dst_srcs[d].add(s)
            self.dst_recent_srcs[d].append(s)
            self.pair_times[(s, d)].append(t)

        # Directed item transition counts from every source sequence.
        self.transition = defaultdict(Counter)
        for hist in self.src_history.values():
            for (left, _), (right, _) in zip(hist, hist[1:]):
                self.transition[left][right] += 1

        self.seen_dst = set(self.dst_times)
        self.dst_degree = {d: len(ts) for d, ts in self.dst_times.items()}
        time_min = float(prefix["time"].min())
        time_max = float(prefix["time"].max())
        self.time_span = max(time_max - time_min, 1.0)
        self.short_window = max(self.time_span * 0.005, 1.0)
        self.long_window = max(self.time_span * 0.05, self.short_window)
        self._context_cache = OrderedDict()
        self._context_cache_size = 512

    @staticmethod
    def _before(times, query_time):
        return bisect_left(times, query_time)

    @staticmethod
    def _hawkes(times, end, query_time, tau, limit=100):
        start = max(0, end - limit)
        return sum(math.exp(-(query_time - t) / tau) for t in times[start:end])

    def row(self, src, dst, query_time, candidate_frequency):
        if src not in self._context_cache:
            hist = self.src_history.get(src, ())
            # Prefix is frozen before every suffix query, but retaining this
            # filter makes the class safe for arbitrary cutoffs.
            hist_end = bisect_left([t for _, t in hist], query_time)
            past_hist = hist[:hist_end]
            src_counts = Counter(d for d, _ in past_hist)
            recent_items = [d for d, _ in past_hist[-20:]]
            transition_mass = sum(
                sum(self.transition.get(item, {}).values())
                for item in recent_items
            )
            # Typed bipartite solution.dataset2.collaborative_features filtering:
            # src -> recent dst -> other src -> their recent dst.
            # Every expansion is capped to remain viable on the B leaderboard.
            cf_scores = Counter()
            for item in recent_items[-10:]:
                users = self.dst_recent_srcs.get(item, ())
                # Reverse traversal favors users who touched the item recently.
                unique_users = list(dict.fromkeys(reversed(users)))[:50]
                item_user_norm = math.sqrt(max(len(self.dst_srcs.get(item, ())), 1))
                for other_src in unique_users:
                    if other_src == src:
                        continue
                    other_hist = self.src_history.get(other_src, ())
                    user_norm = math.sqrt(max(len(other_hist), 1))
                    weight = 1.0 / (item_user_norm * user_norm)
                    for other_dst, _ in other_hist[-20:]:
                        cf_scores[other_dst] += weight
            cf_mass = sum(cf_scores.values())
            self._context_cache[src] = (
                len(past_hist), src_counts, recent_items, transition_mass,
                cf_scores, cf_mass,
            )
            if len(self._context_cache) > self._context_cache_size:
                self._context_cache.popitem(last=False)
        else:
            self._context_cache.move_to_end(src)
        (
            src_degree, src_counts, recent_items, transition_mass,
            cf_scores, cf_mass,
        ) = self._context_cache[src]
        pair_ts = self.pair_times.get((src, dst), ())
        pair_count = self._before(pair_ts, query_time)

        if pair_count:
            pair_age = max(0.0, query_time - pair_ts[pair_count - 1])
            pair_recency = math.log1p(pair_age)
        else:
            pair_age = self.time_span
            pair_recency = math.log1p(self.time_span)
        tau_fast = max(self.time_span * 0.002, 1.0)
        tau_medium = max(self.time_span * 0.02, tau_fast)
        tau_slow = max(self.time_span * 0.2, tau_medium)
        pair_hawkes_fast = self._hawkes(
            pair_ts, pair_count, query_time, tau_fast
        )
        pair_hawkes_medium = self._hawkes(
            pair_ts, pair_count, query_time, tau_medium
        )
        pair_hawkes_slow = self._hawkes(
            pair_ts, pair_count, query_time, tau_slow
        )
        if pair_count >= 2:
            gaps = np.diff(pair_ts[max(0, pair_count - 8):pair_count])
            expected_gap = float(np.median(gaps))
            pair_due_ratio_log = math.log1p(pair_age / max(expected_gap, 1.0))
            pair_due_error_log = math.log1p(abs(pair_age - expected_gap))
        else:
            pair_due_ratio_log = 0.0
            pair_due_error_log = math.log1p(self.time_span)

        dst_ts = self.dst_times.get(dst, ())
        dst_count = self._before(dst_ts, query_time)
        if dst_count:
            dst_recency = math.log1p(max(0.0, query_time - dst_ts[dst_count - 1]))
        else:
            dst_recency = math.log1p(self.time_span)

        last_transition = (
            self.transition.get(recent_items[-1], {}).get(dst, 0)
            if recent_items else 0
        )
        recent_transition = sum(
            self.transition.get(item, {}).get(dst, 0) for item in recent_items
        )
        cf_support = cf_scores.get(dst, 0.0)
        short_start = bisect_left(dst_ts, query_time - self.short_window, 0, dst_count)
        long_start = bisect_left(dst_ts, query_time - self.long_window, 0, dst_count)
        dst_recent_short = dst_count - short_start
        dst_recent_long = dst_count - long_start
        dst_hawkes_fast = self._hawkes(
            dst_ts, dst_count, query_time, tau_fast
        )
        dst_hawkes_medium = self._hawkes(
            dst_ts, dst_count, query_time, tau_medium
        )
        dst_hawkes_slow = self._hawkes(
            dst_ts, dst_count, query_time, tau_slow
        )
        dst_intensity_acceleration = (
            dst_hawkes_fast / max(dst_hawkes_medium, 1e-12)
        )

        return [
            pair_count,
            math.log1p(pair_count),
            pair_recency,
            pair_hawkes_fast,
            pair_hawkes_medium,
            pair_hawkes_slow,
            pair_due_ratio_log,
            pair_due_error_log,
            math.log1p(src_degree),
            math.log1p(len(src_counts)),
            math.log1p(dst_count),
            math.log1p(len(self.dst_srcs.get(dst, ()))),
            dst_recency,
            dst_recent_short,
            dst_recent_long,
            dst_recent_short / max(dst_recent_long, 1),
            dst_hawkes_fast,
            dst_hawkes_medium,
            dst_hawkes_slow,
            dst_intensity_acceleration,
            int(dst in self.seen_dst),
            int(pair_count > 0),
            pair_count / max(src_degree, 1),
            math.log1p(last_transition),
            math.log1p(recent_transition),
            recent_transition / max(transition_mass, 1),
            cf_support,
            math.log1p(cf_support),
            cf_support / max(cf_mass, 1e-12),
            math.log1p(candidate_frequency.get(dst, 0)),
        ]


def reciprocal_rank(labels, scores, groups):
    offset = 0
    values = []
    for size in groups:
        y = labels[offset:offset + size]
        p = scores[offset:offset + size]
        positive = int(np.flatnonzero(y == 1)[0])
        # Stable tie handling gives deterministic validation.
        order = np.argsort(-p, kind="mergesort")
        rank = int(np.flatnonzero(order == positive)[0]) + 1
        values.append(1.0 / rank)
        offset += size
    return float(np.mean(values))


def baseline_mrrs(labels, features, groups):
    """Report transparent baselines on exactly the same candidate groups."""
    columns = {name: i for i, name in enumerate(FEATURE_NAMES)}
    return {
        "pair_count": reciprocal_rank(
            labels, features[:, columns["pair_count"]], groups
        ),
        "dst_degree": reciprocal_rank(
            labels, features[:, columns["dst_degree_log"]], groups
        ),
        "transition": reciprocal_rank(
            labels, features[:, columns["recent_item_transition_norm"]], groups
        ),
        "bipartite_cf": reciprocal_rank(
            labels, features[:, columns["bipartite_cf_support"]], groups
        ),
        "candidate_frequency": reciprocal_rank(
            labels, features[:, columns["candidate_frequency_log"]], groups
        ),
    }


def sample_query_indices(frame, start, stop, count, rng):
    available = np.arange(start, stop)
    if count >= len(available):
        return available
    return np.sort(rng.choice(available, size=count, replace=False))


def make_groups(
    extractor,
    frame,
    indices,
    candidate_pool,
    candidate_frequency,
    num_candidates,
    rng,
):
    features, labels, groups = [], [], []
    for idx in indices:
        row = frame.iloc[int(idx)]
        src, positive, query_time = int(row.src), int(row.dst), float(row.time)

        candidates = [positive]
        used = {positive}
        while len(candidates) < num_candidates:
            candidate = int(candidate_pool[rng.integers(0, len(candidate_pool))])
            if candidate not in used:
                used.add(candidate)
                candidates.append(candidate)

        # Randomize the positive position so ties cannot exploit position.
        rng.shuffle(candidates)
        for candidate in candidates:
            features.append(
                extractor.row(
                    src, candidate, query_time, candidate_frequency
                )
            )
            labels.append(int(candidate == positive))
        groups.append(len(candidates))

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(groups, dtype=np.int32),
    )


def predict_test(
    extractor,
    model,
    test,
    candidate_frequency,
    output_path,
    row_batch_size=250,
):
    candidates = test.iloc[:, 2:].to_numpy(dtype=np.int64)
    srcs = test["src"].to_numpy(dtype=np.int64)
    times = test["time"].to_numpy(dtype=np.float64)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        for start in range(0, len(test), row_batch_size):
            stop = min(start + row_batch_size, len(test))
            rows = []
            for i in range(start, stop):
                src, query_time = int(srcs[i]), float(times[i])
                for candidate in candidates[i]:
                    rows.append(
                        extractor.row(
                            src, int(candidate), query_time, candidate_frequency
                        )
                    )
            matrix = np.asarray(rows, dtype=np.float32)
            scores = model.predict(
                matrix, num_iteration=model.best_iteration
            ).reshape(stop - start, candidates.shape[1])
            # Only ordering matters, but competition requires [0, 1].
            row_min = scores.min(axis=1, keepdims=True)
            row_max = scores.max(axis=1, keepdims=True)
            scores = (scores - row_min) / np.maximum(row_max - row_min, 1e-12)
            for score_row in scores:
                handle.write(",".join(f"{value:.8f}" for value in score_row) + "\n")
            if start == 0 or stop == len(test) or stop % 10000 == 0:
                print(f"Predicted {stop}/{len(test)} test rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dataset1", "dataset2"], required=True)
    parser.add_argument("--data_dir", default=".")
    parser.add_argument(
        "--output_dir", default="outputs/training/dataset2/temporal"
    )
    parser.add_argument("--cutoff", type=float, default=0.8)
    parser.add_argument("--train_queries", type=int, default=20000)
    parser.add_argument("--valid_queries", type=int, default=5000)
    parser.add_argument("--num_candidates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--predict_test", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    train_path = os.path.join(args.data_dir, args.dataset, "train.csv")
    test_path = os.path.join(args.data_dir, args.dataset, "test.csv")
    frame = pd.read_csv(train_path).sort_values("time", kind="stable").reset_index(drop=True)
    test = pd.read_csv(test_path)

    cutoff_idx = int(len(frame) * args.cutoff)
    prefix = frame.iloc[:cutoff_idx]
    suffix_start = cutoff_idx
    suffix_mid = cutoff_idx + (len(frame) - cutoff_idx) // 2
    if suffix_mid <= suffix_start or suffix_mid >= len(frame):
        raise ValueError("cutoff leaves insufficient train/validation suffix data")

    # Sampling directly from flattened competition candidates reproduces their
    # marginal ID/seen distribution without using any labels.
    candidate_pool = test.iloc[:, 2:].to_numpy(dtype=np.int64).reshape(-1)
    candidate_frequency = Counter(map(int, candidate_pool))
    extractor = FrozenTemporalFeatures(prefix)

    train_indices = sample_query_indices(
        frame, suffix_start, suffix_mid, args.train_queries, rng
    )
    valid_indices = sample_query_indices(
        frame, suffix_mid, len(frame), args.valid_queries, rng
    )

    x_train, y_train, g_train = make_groups(
        extractor, frame, train_indices, candidate_pool, candidate_frequency,
        args.num_candidates, rng
    )
    x_valid, y_valid, g_valid = make_groups(
        extractor, frame, valid_indices, candidate_pool, candidate_frequency,
        args.num_candidates, rng
    )

    train_set = lgb.Dataset(
        x_train, label=y_train, group=g_train, feature_name=FEATURE_NAMES
    )
    valid_set = lgb.Dataset(
        x_valid, label=y_valid, group=g_valid, feature_name=FEATURE_NAMES,
        reference=train_set
    )
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "lambdarank_truncation_level": min(20, args.num_candidates),
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": args.seed,
        "num_threads": -1,
    }
    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(25)],
    )
    valid_scores = model.predict(x_valid, num_iteration=model.best_iteration)
    mrr = reciprocal_rank(y_valid, valid_scores, g_valid)

    cutoff_tag = f"cutoff_{args.cutoff:.2f}".replace(".", "p")
    out_dir = os.path.join(args.output_dir, args.dataset, cutoff_tag)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.txt")
    report_path = os.path.join(out_dir, "report.json")
    model.save_model(model_path)
    report = {
        "dataset": args.dataset,
        "cutoff": args.cutoff,
        "prefix_rows": len(prefix),
        "train_queries": len(g_train),
        "valid_queries": len(g_valid),
        "num_candidates": args.num_candidates,
        "best_iteration": model.best_iteration,
        "validation_mrr": mrr,
        "baseline_mrr": baseline_mrrs(y_valid, x_valid, g_valid),
        "feature_importance_gain": dict(
            sorted(
                zip(FEATURE_NAMES, map(float, model.feature_importance("gain"))),
                key=lambda item: -item[1],
            )
        ),
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")

    if args.predict_test:
        print("Rebuilding frozen features from all official training rows...")
        full_extractor = FrozenTemporalFeatures(frame)
        prediction_path = os.path.join(
            out_dir, f"{args.dataset}_temporal_scores.csv"
        )
        predict_test(
            full_extractor, model, test, candidate_frequency, prediction_path
        )
        print(f"Saved predictions: {prediction_path}")


if __name__ == "__main__":
    main()
