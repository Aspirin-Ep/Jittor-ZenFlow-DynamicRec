"""Shared evaluation and experiment-registry helpers for base ensemble."""

import json
import math
import os
import time

import numpy as np


def ranking_metrics(labels, scores, ks=(1, 5, 10, 20, 30)):
    positive = labels.argmax(axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.argmax(order == positive[:, None], axis=1) + 1
    result = {
        "mrr": float(np.mean(1.0 / ranks)),
        "ndcg10": float(np.mean(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0))),
        "mean_rank": float(np.mean(ranks)),
    }
    for k in ks:
        result["hits1" if k == 1 else f"recall{k}"] = float(np.mean(ranks <= k))
    return result, ranks


def metric_by_bins(labels, scores, values, boundaries, names):
    output = {}
    for index, name in enumerate(names):
        low, high = boundaries[index], boundaries[index + 1]
        mask = (values >= low) & (values < high)
        if not mask.any():
            continue
        metrics, _ = ranking_metrics(labels[mask], scores[mask])
        output[name] = {"count": int(mask.sum()), **metrics}
    return output


def model_comparison(labels, left, right):
    left_order = np.argsort(-left, axis=1)
    right_order = np.argsort(-right, axis=1)
    left_metrics, _ = ranking_metrics(labels, left)
    right_metrics, _ = ranking_metrics(labels, right)
    correlations = []
    for a, b in zip(left[:10000], right[:10000]):
        value = np.corrcoef(a, b)[0, 1]
        if np.isfinite(value):
            correlations.append(value)
    return {
        "left": left_metrics,
        "right": right_metrics,
        "top1_agreement": float(np.mean(left_order[:, 0] == right_order[:, 0])),
        "top20_overlap": float(np.mean([
            len(set(a[:20]) & set(b[:20])) / 20.0
            for a, b in zip(left_order, right_order)
        ])),
        "mean_score_correlation": float(np.mean(correlations)),
    }


def append_experiment(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = dict(record)
    record.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

