"""Exact cached inference for the retained statistical ranker Dataset2 statistical scorer.

The original bytecode recomputes a source's solution.dataset2.collaborative_features neighborhood for
every test row.  Dataset2 test timestamps are all after training and there are
only 2,180 unique test sources, so that neighborhood is invariant and can be
cached without changing the formula.
"""

import argparse
import os
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def build_indices(frame):
    src_history = defaultdict(list)
    dst_history = defaultdict(list)
    pair_times = defaultdict(list)
    dst_all_srcs = defaultdict(set)
    src_all_dsts = defaultdict(set)
    for src, dst, timestamp in frame[["src", "dst", "time"]].itertuples(
        index=False, name=None
    ):
        src, dst, timestamp = int(src), int(dst), float(timestamp)
        src_history[src].append((dst, timestamp))
        dst_history[dst].append((src, timestamp))
        pair_times[(src, dst)].append(timestamp)
        dst_all_srcs[dst].add(src)
        src_all_dsts[src].add(dst)
    for values in src_history.values():
        values.sort(key=lambda item: item[1])
    for values in dst_history.values():
        values.sort(key=lambda item: item[1])
    for values in pair_times.values():
        values.sort()
    return (
        src_history, dst_history, pair_times,
        dst_all_srcs, src_all_dsts,
    )


def compute_cached(
    src_history, dst_history, pair_times, dst_all_srcs,
    test_src, test_time, test_candidates, max_time,
):
    destination_sources = {
        destination: set(source for source, _ in history)
        for destination, history in dst_history.items()
    }
    context = {}
    for src_value in np.unique(test_src):
        src = int(src_value)
        recent_destinations = [
            destination
            for destination, _ in src_history.get(src, ())[-50:]
        ]
        recent_set = set(recent_destinations)
        co_occurrence = Counter()
        for recent_destination in recent_destinations[-20:]:
            for other_source in dst_all_srcs.get(
                recent_destination, ()
            ):
                if other_source == src:
                    continue
                for other_destination, _ in src_history.get(
                    other_source, ()
                )[-20:]:
                    if other_destination not in recent_set:
                        co_occurrence[other_destination] += 1
        context[src] = (recent_set, co_occurrence)

    output = np.zeros(test_candidates.shape, dtype=np.float64)
    recency_threshold = max_time * 0.05
    destination_popularity = {
        destination: len(sources)
        for destination, sources in dst_all_srcs.items()
    }
    for row in range(len(test_src)):
        src = int(test_src[row])
        query_time = float(test_time[row])
        recent_set, co_occurrence = context[src]
        for column, candidate_value in enumerate(test_candidates[row]):
            candidate = int(candidate_value)
            score = 0.5 * co_occurrence.get(candidate, 0)
            times = pair_times.get((src, candidate), ())
            if times:
                score += len(times)
                if query_time - times[-1] < recency_threshold:
                    score += 1.5
            score += 0.3 * np.log1p(
                destination_popularity.get(candidate, 0)
            )
            common = len(
                recent_set & destination_sources.get(candidate, set())
            )
            score += 0.3 * common
            output[row, column] = score
        if row == 0 or (row + 1) % 10000 == 0 or row + 1 == len(test_src):
            print(f"  cached statistical ranker {row + 1}/{len(test_src)}")
    low = output.min(axis=1, keepdims=True)
    high = output.max(axis=1, keepdims=True)
    span = high - low
    span[span == 0] = 1.0
    return (output - low) / span


def save(path, values):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, values, delimiter=",", fmt="%.8f")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".")
    parser.add_argument(
        "--output_path",
        default="solution.base/work/dataset2/components/stat.csv",
    )
    args = parser.parse_args()
    started = time.time()
    frame = pd.read_csv(os.path.join(
        args.data_dir, "dataset2", "train.csv"
    ))
    test = pd.read_csv(os.path.join(
        args.data_dir, "dataset2", "test.csv"
    ))
    test_src = test["src"].to_numpy(dtype=np.int64)
    test_time = test["time"].to_numpy(dtype=np.float64)
    candidates = test.iloc[:, 2:].to_numpy(dtype=np.int64)
    max_time = float(frame["time"].max())
    if float(test_time.min()) <= max_time:
        raise ValueError("cached equivalence requires test after training")
    indices = build_indices(frame)

    values = compute_cached(
        indices[0], indices[1], indices[2], indices[3],
        test_src, test_time, candidates, max_time,
    )
    save(args.output_path, values)
    print(
        f"Saved {args.output_path}; rows={len(values)}; "
        f"minutes={(time.time() - started) / 60.0:.2f}"
    )


if __name__ == "__main__":
    main()
