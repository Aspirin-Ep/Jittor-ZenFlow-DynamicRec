"""candidate-aware: Dataset2 official-split new-link ranker.

The provided ``split`` column defines the trustworthy validation protocol:

* split=0 is historical graph/training data;
* split=1 contains only links between seen endpoints whose pair is new.

candidate-aware reproduces the competition candidate mix and explicitly trains against
unseen-node distractors and historical-pair hard negatives.
"""

import argparse
import json
import os
import zipfile
from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd

from .temporal_features import (
    FEATURE_NAMES,
    FrozenTemporalFeatures,
    baseline_mrrs,
    reciprocal_rank,
)


def eligible_new_link_indices(frame, start, stop, extractor):
    eligible = []
    seen_src = extractor.src_history
    seen_dst = extractor.seen_dst
    seen_pair = extractor.pair_times
    src = frame["src"].to_numpy(dtype=np.int64)
    dst = frame["dst"].to_numpy(dtype=np.int64)
    for idx in range(start, stop):
        s, d = int(src[idx]), int(dst[idx])
        if s in seen_src and d in seen_dst and (s, d) not in seen_pair:
            eligible.append(idx)
    return np.asarray(eligible, dtype=np.int64)


def sample_indices(indices, count, rng):
    if count <= 0 or count >= len(indices):
        return np.sort(indices)
    return np.sort(rng.choice(indices, size=count, replace=False))


class CompetitionCandidateSampler:
    def __init__(self, test_candidates, extractor):
        flat = test_candidates.reshape(-1).astype(np.int64)
        self.frequency = Counter(map(int, flat))
        seen_mask = np.isin(flat, np.fromiter(extractor.seen_dst, dtype=np.int64))
        self.seen_pool = flat[seen_mask]
        self.unseen_pool = flat[~seen_mask]
        if not len(self.seen_pool) or not len(self.unseen_pool):
            raise ValueError("candidate pools must contain both seen and unseen dst")

    @staticmethod
    def _add_from_pool(candidates, used, pool, target_count, rng, reject=None):
        attempts = 0
        while target_count > 0 and attempts < target_count * 1000 + 1000:
            value = int(pool[rng.integers(0, len(pool))])
            attempts += 1
            if value in used or (reject is not None and reject(value)):
                continue
            used.add(value)
            candidates.append(value)
            target_count -= 1
        return target_count

    def sample(self, src, positive, extractor, num_candidates, rng):
        # Test has about 53.6 unseen candidates and 1.5 historical candidates
        # per row. Scale these counts if a smaller smoke-test group is used.
        unseen_count = min(
            int(round(num_candidates * 0.536)), num_candidates - 1
        )
        history_count = min(
            max(1, int(round(num_candidates * 0.02))),
            num_candidates - 1 - unseen_count,
        )
        candidates, used = [positive], {positive}

        remaining = self._add_from_pool(
            candidates, used, self.unseen_pool, unseen_count, rng
        )
        if remaining:
            raise RuntimeError("unable to sample enough unique unseen candidates")

        history = list(dict.fromkeys(
            d for d, _ in extractor.src_history.get(src, ())
            if d != positive
        ))
        rng.shuffle(history)
        for value in history[:history_count]:
            if value not in used:
                used.add(value)
                candidates.append(int(value))

        def reject_seen(value):
            return (src, value) in extractor.pair_times

        remaining_seen = num_candidates - len(candidates)
        left = self._add_from_pool(
            candidates, used, self.seen_pool, remaining_seen, rng,
            reject=reject_seen,
        )
        if left:
            # Very conservative fallback: permit extra historical negatives.
            self._add_from_pool(
                candidates, used, self.seen_pool, left, rng
            )
        if len(candidates) != num_candidates:
            raise RuntimeError(
                f"candidate construction failed: {len(candidates)}"
            )
        rng.shuffle(candidates)
        return candidates


def make_official_groups(
    extractor,
    frame,
    indices,
    sampler,
    num_candidates,
    rng,
):
    features, labels, groups = [], [], []
    for idx in indices:
        row = frame.iloc[int(idx)]
        src, positive, query_time = int(row.src), int(row.dst), float(row.time)
        candidates = sampler.sample(
            src, positive, extractor, num_candidates, rng
        )
        for candidate in candidates:
            features.append(
                extractor.row(
                    src, candidate, query_time, sampler.frequency
                )
            )
            labels.append(int(candidate == positive))
        groups.append(num_candidates)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        np.asarray(groups, dtype=np.int32),
    )


def train_ranker(x, y, groups, rounds, seed, valid=None):
    train_set = lgb.Dataset(
        x, label=y, group=groups, feature_name=FEATURE_NAMES
    )
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3, 5, 10],
        "lambdarank_truncation_level": 20,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": seed,
        "num_threads": -1,
    }
    if valid is None:
        return lgb.train(params, train_set, num_boost_round=rounds)

    x_valid, y_valid, g_valid = valid
    valid_set = lgb.Dataset(
        x_valid, label=y_valid, group=g_valid,
        feature_name=FEATURE_NAMES, reference=train_set,
    )
    return lgb.train(
        params,
        train_set,
        num_boost_round=rounds,
        valid_sets=[valid_set],
        valid_names=["official"],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(25)],
    )


def apply_rules(scores, features, unseen=True, history=True):
    adjusted = scores.copy()
    if unseen:
        seen_col = FEATURE_NAMES.index("is_seen_dst")
        adjusted[features[:, seen_col] < 0.5] = -1e9
    if history:
        history_col = FEATURE_NAMES.index("is_src_history")
        adjusted[features[:, history_col] > 0.5] = -1e9
    return adjusted


def predict_test(extractor, model, test, frequency, output_path):
    candidates = test.iloc[:, 2:].to_numpy(dtype=np.int64)
    srcs = test["src"].to_numpy(dtype=np.int64)
    times = test["time"].to_numpy(dtype=np.float64)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for start in range(0, len(test), 250):
            stop = min(start + 250, len(test))
            rows = []
            for i in range(start, stop):
                for candidate in candidates[i]:
                    rows.append(
                        extractor.row(
                            int(srcs[i]), int(candidate), float(times[i]),
                            frequency,
                        )
                    )
            matrix = np.asarray(rows, dtype=np.float32)
            scores = model.predict(matrix).reshape(stop - start, 100)
            seen_col = FEATURE_NAMES.index("is_seen_dst")
            history_col = FEATURE_NAMES.index("is_src_history")
            seen = matrix[:, seen_col].reshape(stop - start, 100) > 0.5
            history = matrix[:, history_col].reshape(stop - start, 100) > 0.5
            valid = seen & ~history
            low = np.where(valid, scores, np.inf).min(axis=1, keepdims=True)
            high = np.where(valid, scores, -np.inf).max(axis=1, keepdims=True)
            normalized = (scores - low) / np.maximum(high - low, 1e-12)
            # Invalid candidates are strictly below every valid candidate while
            # valid candidates retain their full internal score resolution.
            scores = np.where(valid, 0.001 + 0.999 * normalized, 0.0)
            for values in scores:
                handle.write(",".join(f"{value:.8f}" for value in values) + "\n")
            if start == 0 or stop == len(test) or stop % 10000 == 0:
                print(f"Predicted {stop}/{len(test)}")


