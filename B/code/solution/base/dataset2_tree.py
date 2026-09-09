"""Exact test-time specialization of the retained tree ranker Dataset2 model.

All Dataset2 test timestamps are later than the final training timestamp.
tree ranker's history slices and graph neighborhoods are therefore fixed at inference
time.  This implementation caches those fixed quantities, evaluates the
time-dependent terms in arrays, and uses the unchanged saved LightGBM model.
"""

import argparse
import math
import os
import time
from collections import Counter, defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd

from .dataset2_tree_features import TreeFeatureExtractor


FEATURES = [
    "pair_count", "pair_count_log", "pair_recency", "pair_recency_log",
    "pair_decay_short", "pair_decay_long", "pair_avg_interval",
    "pair_std_interval", "pair_freq_trend", "src_degree",
    "src_degree_log", "src_unique_dst", "src_unique_dst_log",
    "src_activity_rate", "src_last_active", "src_entropy", "dst_degree",
    "dst_degree_log", "dst_unique_src", "dst_unique_src_log",
    "dst_popularity", "dst_activity_rate", "dst_last_active",
    "common_neighbors", "common_neighbors_log", "jaccard", "pref_attach",
    "adamic_adar", "resource_allocation", "src_in_dst_neighbors",
    "dst_in_recent_src", "dst_freq_in_recent_src", "pair_count_pctile",
    "pair_recency_vs_src", "dst_recent_unique_src",
]
COL = {name: index for index, name in enumerate(FEATURES)}


class SpecializedTreeFeatures:
    def __init__(self, extractor, maximum_id):
        self.extractor = extractor
        self.max_time = float(extractor.max_time)
        self.base = int(maximum_id) + 1
        size = self.base

        self.src_degree = np.zeros(size, dtype=np.float64)
        self.src_unique = np.zeros(size, dtype=np.float64)
        self.src_first = np.zeros(size, dtype=np.float64)
        self.src_last = np.zeros(size, dtype=np.float64)
        self.src_entropy = np.zeros(size, dtype=np.float64)
        self.src_recent50 = {}
        self.src_recent10 = {}
        for src, history in extractor.src_history.items():
            destinations = [destination for destination, _ in history]
            count = len(history)
            self.src_degree[src] = count
            self.src_unique[src] = len(set(destinations))
            self.src_first[src] = history[0][1]
            self.src_last[src] = history[-1][1]
            counter = Counter(destinations)
            entropy = 0.0
            for frequency in counter.values():
                probability = frequency / count
                entropy -= probability * math.log2(probability + 1e-12)
            self.src_entropy[src] = entropy
            self.src_recent50[src] = set(destinations[-50:])
            self.src_recent10[src] = Counter(destinations[-10:])

        self.dst_degree = np.zeros(size, dtype=np.float64)
        self.dst_unique = np.zeros(size, dtype=np.float64)
        self.dst_first = np.zeros(size, dtype=np.float64)
        self.dst_last = np.zeros(size, dtype=np.float64)
        self.dst_recent50_len = np.zeros(size, dtype=np.float64)
        self.dst_recent20_unique = np.zeros(size, dtype=np.float64)
        self.inverted50 = defaultdict(list)
        self.inverted20 = defaultdict(list)
        for dst, history in extractor.dst_history.items():
            sources = [source for source, _ in history]
            self.dst_degree[dst] = len(history)
            self.dst_unique[dst] = len(set(sources))
            self.dst_first[dst] = history[0][1]
            self.dst_last[dst] = history[-1][1]
            recent50 = set(sources[-50:])
            recent20 = set(sources[-20:])
            self.dst_recent50_len[dst] = len(recent50)
            self.dst_recent20_unique[dst] = len(recent20)
            for source in recent50:
                self.inverted50[source].append(dst)
            for source in recent20:
                self.inverted20[source].append(dst)

        self.node_degree = np.zeros(size, dtype=np.float64)
        for node, degree in extractor.src_degree.items():
            self.node_degree[node] += degree
        for node, degree in extractor.dst_degree.items():
            self.node_degree[node] += degree

        self._build_pair_table()

    def _build_pair_table(self):
        number = len(self.extractor.pair_times)
        keys = np.empty(number, dtype=np.int64)
        values = np.empty((number, 10), dtype=np.float64)
        short_scale = self.max_time * 0.02 + 1.0
        long_scale = self.max_time * 0.2 + 1.0
        percentile = {}
        for src, destinations in self.extractor.src_dsts.items():
            counts = Counter(
                len(self.extractor.pair_times[(src, destination)])
                for destination in destinations
            )
            total = len(destinations)
            running = 0
            lookup = {}
            for count in sorted(counts):
                running += counts[count]
                lookup[count] = running / total
            percentile[src] = lookup

        for row, ((src, dst), times_list) in enumerate(
            self.extractor.pair_times.items()
        ):
            times = np.asarray(times_list, dtype=np.float64)
            count = len(times)
            keys[row] = int(src) * self.base + int(dst)
            values[row, 0] = count
            values[row, 1] = times[-1]
            values[row, 2] = sum(
                math.exp(float(value) / short_scale) for value in times
            )
            values[row, 3] = sum(
                math.exp(float(value) / long_scale) for value in times
            )
            if count > 1:
                intervals = np.diff(times)
                values[row, 4] = float(np.mean(intervals))
                values[row, 5] = (
                    float(np.std(intervals)) if len(intervals) > 1 else 0.0
                )
            else:
                values[row, 4:6] = 0.0
            if count >= 4:
                half = count // 2
                values[row, 6] = half
                values[row, 7] = times[half]
                values[row, 8] = half / (
                    times[half] - times[0] + 1.0
                )
            else:
                values[row, 6:9] = 0.0
            values[row, 9] = percentile[src][count]
        order = np.argsort(keys)
        self.pair_keys = keys[order]
        self.pair_values = values[order]

    def graph_features(self, src):
        common = {}
        source_recent = self.src_recent50.get(src, set())
        for node in source_recent:
            degree = self.node_degree[node]
            aa = 1.0 / (math.log(degree) + 1e-9) if degree > 1 else 0.0
            ra = 1.0 / (degree + 1e-9) if degree > 1 else 0.0
            for dst in self.inverted50.get(node, ()):
                current = common.get(dst)
                if current is None:
                    common[dst] = [1.0, aa, ra]
                else:
                    current[0] += 1.0
                    current[1] += aa
                    current[2] += ra
        common20 = Counter()
        for node in source_recent:
            common20.update(self.inverted20.get(node, ()))
        return common, common20

    def features_for_source(self, src, query_times, candidates):
        rows, width = candidates.shape
        dst = candidates.reshape(-1).astype(np.int64, copy=False)
        query = np.repeat(
            query_times.astype(np.float64, copy=False), width
        )
        count = len(dst)
        x = np.zeros((count, len(FEATURES)), dtype=np.float64)

        src_degree = self.src_degree[src]
        src_unique = self.src_unique[src]
        source_recent = self.src_recent50.get(src, set())
        recent10 = self.src_recent10.get(src, Counter())
        x[:, COL["src_degree"]] = src_degree
        x[:, COL["src_degree_log"]] = math.log1p(src_degree)
        x[:, COL["src_unique_dst"]] = src_unique
        x[:, COL["src_unique_dst_log"]] = math.log1p(src_unique)
        if src_degree:
            x[:, COL["src_activity_rate"]] = src_degree / (
                query - self.src_first[src] + 1.0
            )
            x[:, COL["src_last_active"]] = query - self.src_last[src]
        else:
            x[:, COL["src_last_active"]] = self.max_time
        x[:, COL["src_entropy"]] = self.src_entropy[src]

        dst_degree = self.dst_degree[dst]
        dst_unique = self.dst_unique[dst]
        x[:, COL["dst_degree"]] = dst_degree
        x[:, COL["dst_degree_log"]] = np.log1p(dst_degree)
        x[:, COL["dst_unique_src"]] = dst_unique
        x[:, COL["dst_unique_src_log"]] = np.log1p(dst_unique)
        x[:, COL["dst_popularity"]] = np.log1p(dst_unique)
        present_dst = dst_degree > 0
        x[present_dst, COL["dst_activity_rate"]] = (
            dst_degree[present_dst]
            / (query[present_dst] - self.dst_first[dst[present_dst]] + 1.0)
        )
        x[present_dst, COL["dst_last_active"]] = (
            query[present_dst] - self.dst_last[dst[present_dst]]
        )
        x[~present_dst, COL["dst_last_active"]] = self.max_time
        x[:, COL["dst_recent_unique_src"]] = self.dst_recent20_unique[dst]

        graph, graph20 = self.graph_features(src)
        common = np.fromiter(
            (graph.get(int(value), (0.0, 0.0, 0.0))[0] for value in dst),
            dtype=np.float64, count=count,
        )
        aa = np.fromiter(
            (graph.get(int(value), (0.0, 0.0, 0.0))[1] for value in dst),
            dtype=np.float64, count=count,
        )
        ra = np.fromiter(
            (graph.get(int(value), (0.0, 0.0, 0.0))[2] for value in dst),
            dtype=np.float64, count=count,
        )
        common20 = np.fromiter(
            (graph20.get(int(value), 0.0) for value in dst),
            dtype=np.float64, count=count,
        )
        x[:, COL["common_neighbors"]] = common
        x[:, COL["common_neighbors_log"]] = np.log1p(common)
        union = len(source_recent) + self.dst_recent50_len[dst] - common
        x[:, COL["jaccard"]] = common / np.maximum(union, 1.0)
        x[:, COL["pref_attach"]] = (
            math.log1p(src_degree) * np.log1p(dst_degree)
        )
        x[:, COL["adamic_adar"]] = aa
        x[:, COL["resource_allocation"]] = ra
        x[:, COL["src_in_dst_neighbors"]] = common20
        recent_frequency = np.fromiter(
            (recent10.get(int(value), 0) for value in dst),
            dtype=np.float64, count=count,
        )
        x[:, COL["dst_in_recent_src"]] = recent_frequency > 0
        x[:, COL["dst_freq_in_recent_src"]] = recent_frequency

        keys = int(src) * self.base + dst
        positions = np.searchsorted(self.pair_keys, keys)
        matched = positions < len(self.pair_keys)
        safe_positions = np.minimum(positions, len(self.pair_keys) - 1)
        matched &= self.pair_keys[safe_positions] == keys
        pair = np.zeros((count, 10), dtype=np.float64)
        pair[matched] = self.pair_values[safe_positions[matched]]
        pair_count = pair[:, 0]
        x[:, COL["pair_count"]] = pair_count
        x[:, COL["pair_count_log"]] = np.log1p(pair_count)
        x[:, COL["pair_recency"]] = self.max_time
        x[:, COL["pair_recency_log"]] = math.log1p(self.max_time)
        x[:, COL["pair_freq_trend"]] = 1.0
        x[:, COL["pair_recency_vs_src"]] = 1.0
        if np.any(matched):
            x[matched, COL["pair_recency"]] = (
                query[matched] - pair[matched, 1]
            )
            x[matched, COL["pair_recency_log"]] = np.log1p(
                query[matched] - pair[matched, 1]
            )
            short_scale = self.max_time * 0.02 + 1.0
            long_scale = self.max_time * 0.2 + 1.0
            x[matched, COL["pair_decay_short"]] = (
                np.exp(-query[matched] / short_scale) * pair[matched, 2]
            )
            x[matched, COL["pair_decay_long"]] = (
                np.exp(-query[matched] / long_scale) * pair[matched, 3]
            )
            x[matched, COL["pair_avg_interval"]] = pair[matched, 4]
            x[matched, COL["pair_std_interval"]] = pair[matched, 5]
            frequent = matched & (pair_count >= 4)
            x[frequent, COL["pair_freq_trend"]] = (
                (pair[frequent, 0] - pair[frequent, 6])
                / (query[frequent] - pair[frequent, 7] + 1.0)
                / (pair[frequent, 8] + 1e-9)
            )
            x[matched, COL["pair_count_pctile"]] = pair[matched, 9]
            x[matched, COL["pair_recency_vs_src"]] = (
                (query[matched] - pair[matched, 1])
                / (query[matched] - self.src_last[src] + 1.0)
            )
        return x


def normalize(values):
    low = values.min(axis=1, keepdims=True)
    high = values.max(axis=1, keepdims=True)
    span = high - low
    span[span == 0] = 1.0
    return (values - low) / span


def write(path, values):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, values, delimiter=",", fmt="%.8f")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".")
    parser.add_argument(
        "--model", default="solution.base/models/dataset2/tree_ranker.txt"
    )
    parser.add_argument(
        "--output_path",
        default="solution.base/work/dataset2/components/tree.csv",
    )
    parser.add_argument("--verify_rows", type=int, default=3)
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()
    started = time.time()
    train = pd.read_csv(os.path.join(args.data_dir, "dataset2", "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "dataset2", "test.csv"))
    if float(test["time"].min()) <= float(train["time"].max()):
        raise ValueError("specialization requires every test time after training")
    candidates = test.iloc[:, 2:].to_numpy(dtype=np.int64)
    maximum_id = max(
        int(train["src"].max()), int(train["dst"].max()),
        int(candidates.max()),
    )
    extractor = TreeFeatureExtractor(train)
    specialized = SpecializedTreeFeatures(extractor, maximum_id)
    model = lgb.Booster(model_file=args.model)
    if model.feature_name() != FEATURES:
        raise ValueError("saved model feature order differs from tree ranker source")

    if args.verify_rows:
        expected = []
        actual = []
        for row in range(min(args.verify_rows, len(test))):
            source = int(test.iloc[row]["src"])
            query_time = float(test.iloc[row]["time"])
            expected.extend(
                extractor.extract_features(source, int(dst), query_time)
                for dst in candidates[row]
            )
            actual.append(specialized.features_for_source(
                source,
                np.asarray([query_time]),
                candidates[row:row + 1],
            ))
        expected_matrix = np.asarray([
            [row[name] for name in FEATURES] for row in expected
        ], dtype=np.float64)
        actual_matrix = np.concatenate(actual)
        difference = np.abs(expected_matrix - actual_matrix)
        print(
            "Feature verification: "
            f"max_abs_error={difference.max():.12g}, "
            f"mismatches_gt_1e-8={int(np.sum(difference > 1e-8))}"
        )
        old_prediction = model.predict(expected_matrix)
        new_prediction = model.predict(actual_matrix)
        prediction_error = float(np.max(np.abs(
            old_prediction - new_prediction
        )))
        print(f"Prediction verification max_abs_error={prediction_error:.12g}")
        if prediction_error != 0.0:
            raise RuntimeError("specialized features change tree ranker predictions")
        if args.verify_only:
            return

    source_values = test["src"].to_numpy(dtype=np.int64)
    time_values = test["time"].to_numpy(dtype=np.float64)
    output = np.empty(candidates.shape, dtype=np.float64)
    unique_sources = np.unique(source_values)
    for number, source in enumerate(unique_sources, 1):
        indices = np.flatnonzero(source_values == source)
        features = specialized.features_for_source(
            int(source), time_values[indices], candidates[indices]
        )
        output[indices] = model.predict(
            features, num_threads=os.cpu_count() or 1
        ).reshape(len(indices), candidates.shape[1])
        if number == 1 or number % 100 == 0 or number == len(unique_sources):
            print(f"  fast tree ranker sources {number}/{len(unique_sources)}")
    output = normalize(output)
    write(args.output_path, output)
    print(
        f"Saved {args.output_path}; shape={output.shape}; "
        f"minutes={(time.time() - started) / 60.0:.2f}"
    )


if __name__ == "__main__":
    main()
