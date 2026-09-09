"""LightGCN: post-training LightGCN propagation over hard-negative GraphMF embeddings."""

import argparse
import json
import os
import time

import jittor as jt
import numpy as np
import pandas as pd
from scipy import sparse

from .temporal_features import FEATURE_NAMES
from .graph_mf import BipartiteMF, lgbm_score
from .ranking import append_experiment, ranking_metrics


def normalized_bipartite(frame, num_src, num_dst):
    src = frame["src"].to_numpy(dtype=np.int64)
    dst = frame["dst"].to_numpy(dtype=np.int64)
    src_degree = np.bincount(src, minlength=num_src).astype(np.float32)
    dst_degree = np.bincount(dst, minlength=num_dst).astype(np.float32)
    weights = (
        1.0
        / np.sqrt(np.maximum(src_degree[src], 1.0))
        / np.sqrt(np.maximum(dst_degree[dst], 1.0))
    )
    matrix = sparse.coo_matrix(
        (weights, (src, dst)), shape=(num_src, num_dst), dtype=np.float32
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


def propagate(adjacency, src0, dst0, layers):
    src_layers = [src0.astype(np.float32)]
    dst_layers = [dst0.astype(np.float32)]
    for _ in range(layers):
        src_layers.append(adjacency @ dst_layers[-1])
        dst_layers.append(adjacency.T @ src_layers[-2])
    return src_layers, dst_layers


def candidate_scores(src_embedding, dst_embedding, bias, sources, candidates):
    output = np.empty(candidates.shape, dtype=np.float32)
    for start in range(0, len(sources), 512):
        stop = min(start + 512, len(sources))
        src = src_embedding[sources[start:stop]]
        dst = dst_embedding[candidates[start:stop]]
        output[start:stop] = (
            src[:, None, :] * dst
        ).sum(axis=-1) + bias[candidates[start:stop]]
    return output


