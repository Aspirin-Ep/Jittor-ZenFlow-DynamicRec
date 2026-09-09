"""Small shared training helpers."""

import lightgbm as lgb
import numpy as np


def train_fixed_ranker(features, labels, names, rounds, seed):
    groups = np.full(
        len(features), features.shape[1], dtype=np.int32
    )
    dataset = lgb.Dataset(
        features.reshape(-1, features.shape[-1]),
        label=labels.reshape(-1),
        group=groups,
        feature_name=names,
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
        dataset,
        num_boost_round=max(int(rounds), 20),
    )

