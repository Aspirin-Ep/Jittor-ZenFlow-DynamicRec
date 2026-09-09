"""
tree: Enhanced LightGBM with advanced graph features.
Improvements over earlier tree:
1. Adamic-Adar / Resource Allocation indices
2. Src interaction entropy & percentile rank features
3. Time trend features (frequency acceleration)
4. Pre-computed time indices for O(log n) lookups
5. Larger training set (200K) + stronger LGBM params
"""
import os
import os.path as osp
import time
import math
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from bisect import bisect_left, bisect_right
from tqdm import tqdm
import lightgbm as lgb
import argparse
import multiprocessing as mp
import tempfile


class TreeFeatureExtractor:
    """Enhanced feature extractor with pre-computed time indices."""

    def __init__(self, train_df, max_time=None):
        self.max_time = max_time or train_df['time'].max()
        src_np = train_df['src'].values
        dst_np = train_df['dst'].values
        t_np = train_df['time'].values.astype(np.float64)

        print('Building indices (tree)...')
        t0 = time.time()

        # Sorted histories with time for binary search
        self.src_history = defaultdict(list)   # src -> [(dst, time)]
        self.dst_history = defaultdict(list)   # dst -> [(src, time)]
        self.pair_times = defaultdict(list)    # (src, dst) -> [times] sorted

        # Global sets (full training data, no time filter)
        self.dst_srcs = defaultdict(set)       # dst -> set of all srcs
        self.src_dsts = defaultdict(set)       # src -> set of all dsts
        self.src_degree = defaultdict(int)
        self.dst_degree = defaultdict(int)

        for i in range(len(train_df)):
            s, d, t = int(src_np[i]), int(dst_np[i]), float(t_np[i])
            self.src_history[s].append((d, t))
            self.dst_history[d].append((s, t))
            self.pair_times[(s, d)].append(t)
            self.dst_srcs[d].add(s)
            self.src_dsts[s].add(d)
            self.src_degree[s] += 1
            self.dst_degree[d] += 1

        # Sort all by time
        for s in self.src_history:
            self.src_history[s].sort(key=lambda x: x[1])
        for d in self.dst_history:
            self.dst_history[d].sort(key=lambda x: x[1])
        for k in self.pair_times:
            self.pair_times[k].sort()

        # Pre-compute time arrays for binary search
        self._src_time_arrays = {}
        for s, hist in self.src_history.items():
            self._src_time_arrays[s] = np.array([t for _, t in hist])
        self._dst_time_arrays = {}
        for d, hist in self.dst_history.items():
            self._dst_time_arrays[d] = np.array([t for _, t in hist])
        self._pair_time_arrays = {}
        for k, times in self.pair_times.items():
            self._pair_time_arrays[k] = np.array(times)

        # Pre-compute log degree for Adamic-Adar
        self._log_dst_degree = {}
        for d, deg in self.dst_degree.items():
            self._log_dst_degree[d] = math.log1p(deg)
        self._log_src_degree = {}
        for s, deg in self.src_degree.items():
            self._log_src_degree[s] = math.log1p(deg)

        self._max_node = max(max(self.src_degree.keys(), default=0),
                             max(self.dst_degree.keys(), default=0))

        print(f'  Built in {time.time()-t0:.1f}s')
        print(f'  Unique src: {len(self.src_history)}, Unique dst: {len(self.dst_history)}')

    def _get_past_count(self, time_arr, query_time):
        """Binary search for count of elements < query_time."""
        if time_arr is None or len(time_arr) == 0:
            return 0
        return bisect_left(time_arr, query_time)

    def _get_past_items_before(self, history, time_arr, query_time):
        """Get items before query_time using binary search."""
        if time_arr is None or len(time_arr) == 0:
            return []
        idx = bisect_left(time_arr, query_time)
        return history[:idx]

    def extract_features(self, src, dst, query_time):
        """Extract enhanced features for a single (src, dst, time) triple."""
        features = {}
        max_t = self.max_time

        # === 1. Historical pair features (same as earlier tree but faster) ===
        pt_arr = self._pair_time_arrays.get((src, dst))
        past_count = self._get_past_count(pt_arr, query_time)

        features['pair_count'] = past_count
        features['pair_count_log'] = math.log1p(past_count)

        if past_count > 0:
            past_pair_times = pt_arr[:past_count]
            last_t = past_pair_times[-1]
            features['pair_recency'] = query_time - last_t
            features['pair_recency_log'] = math.log1p(query_time - last_t)

            # Time decay weighted count (short + long term)
            decay_short = 0.0
            decay_long = 0.0
            for t in past_pair_times:
                dt = query_time - t
                decay_short += math.exp(-dt / (max_t * 0.02 + 1))
                decay_long += math.exp(-dt / (max_t * 0.2 + 1))
            features['pair_decay_short'] = decay_short
            features['pair_decay_long'] = decay_long

            # Interval statistics
            if past_count > 1:
                intervals = np.diff(past_pair_times)
                features['pair_avg_interval'] = float(np.mean(intervals))
                features['pair_std_interval'] = float(np.std(intervals)) if len(intervals) > 1 else 0
                # Time trend: is interaction frequency increasing?
                if past_count >= 4:
                    half = past_count // 2
                    first_half_span = past_pair_times[half] - past_pair_times[0] + 1
                    second_half_span = query_time - past_pair_times[half] + 1
                    first_rate = half / first_half_span
                    second_rate = (past_count - half) / second_half_span
                    features['pair_freq_trend'] = second_rate / (first_rate + 1e-9)
                else:
                    features['pair_freq_trend'] = 1.0
            else:
                features['pair_avg_interval'] = 0
                features['pair_std_interval'] = 0
                features['pair_freq_trend'] = 1.0
        else:
            features['pair_recency'] = max_t
            features['pair_recency_log'] = math.log1p(max_t)
            features['pair_decay_short'] = 0
            features['pair_decay_long'] = 0
            features['pair_avg_interval'] = 0
            features['pair_std_interval'] = 0
            features['pair_freq_trend'] = 1.0

        # === 2. Source node features ===
        src_t_arr = self._src_time_arrays.get(src)
        src_past_count = self._get_past_count(src_t_arr, query_time)
        features['src_degree'] = src_past_count
        features['src_degree_log'] = math.log1p(src_past_count)

        src_past_items = self._get_past_items_before(
            self.src_history.get(src, []), src_t_arr, query_time)
        src_past_dsts = [d for d, t in src_past_items]
        src_unique = len(set(src_past_dsts))
        features['src_unique_dst'] = src_unique
        features['src_unique_dst_log'] = math.log1p(src_unique)

        if src_past_count > 0:
            src_span = query_time - src_past_items[0][1] + 1
            features['src_activity_rate'] = src_past_count / src_span
            features['src_last_active'] = query_time - src_past_items[-1][1]
        else:
            features['src_activity_rate'] = 0
            features['src_last_active'] = max_t

        # NEW: Src interaction entropy (how diverse are src's interactions?)
        if src_past_count > 0:
            dst_counter = Counter(src_past_dsts)
            total = sum(dst_counter.values())
            entropy = 0.0
            for cnt in dst_counter.values():
                p = cnt / total
                if p > 0:
                    entropy -= p * math.log2(p + 1e-12)
            features['src_entropy'] = entropy
        else:
            features['src_entropy'] = 0

        # === 3. Destination node features ===
        dst_t_arr = self._dst_time_arrays.get(dst)
        dst_past_count = self._get_past_count(dst_t_arr, query_time)
        features['dst_degree'] = dst_past_count
        features['dst_degree_log'] = math.log1p(dst_past_count)

        dst_past_items = self._get_past_items_before(
            self.dst_history.get(dst, []), dst_t_arr, query_time)
        dst_past_srcs = [s for s, t in dst_past_items]
        dst_unique = len(set(dst_past_srcs))
        features['dst_unique_src'] = dst_unique
        features['dst_unique_src_log'] = math.log1p(dst_unique)

        features['dst_popularity'] = math.log1p(len(self.dst_srcs.get(dst, set())))

        if dst_past_count > 0:
            dst_span = query_time - dst_past_items[0][1] + 1
            features['dst_activity_rate'] = dst_past_count / dst_span
            features['dst_last_active'] = query_time - dst_past_items[-1][1]
        else:
            features['dst_activity_rate'] = 0
            features['dst_last_active'] = max_t

        # === 4. Common neighbor features ===
        src_dst_set = set(src_past_dsts[-50:])  # src's recent dsts
        dst_src_set = set(dst_past_srcs[-50:])  # dst's recent srcs
        common = src_dst_set & dst_src_set
        features['common_neighbors'] = len(common)
        features['common_neighbors_log'] = math.log1p(len(common))

        union = src_dst_set | dst_src_set
        features['jaccard'] = len(common) / max(len(union), 1)
        features['pref_attach'] = features['src_degree_log'] * features['dst_degree_log']

        # NEW: Adamic-Adar index
        # AA = sum(1/log(degree(common_neighbor))) for each common neighbor
        aa_score = 0.0
        ra_score = 0.0
        for cn in common:
            # For non-bipartite: cn could be in both src and dst roles
            cn_deg = self.src_degree.get(cn, 0) + self.dst_degree.get(cn, 0)
            if cn_deg > 1:
                log_deg = math.log(cn_deg)
                aa_score += 1.0 / (log_deg + 1e-9)
                ra_score += 1.0 / (cn_deg + 1e-9)
        features['adamic_adar'] = aa_score
        features['resource_allocation'] = ra_score

        # === 5. Temporal & contextual features ===
        dst_neighbors_recent = set(dst_past_srcs[-20:])
        features['src_in_dst_neighbors'] = len(src_dst_set & dst_neighbors_recent)

        recent_src_dsts = src_past_dsts[-10:]
        features['dst_in_recent_src'] = 1 if dst in recent_src_dsts else 0
        features['dst_freq_in_recent_src'] = recent_src_dsts.count(dst)

        # NEW: Percentile rank of pair_count among src's all pair counts
        if src_past_count > 0:
            all_pair_counts = []
            for d in set(src_past_dsts):
                pc = self._get_past_count(
                    self._pair_time_arrays.get((src, d)), query_time)
                all_pair_counts.append(pc)
            if all_pair_counts:
                rank = sum(1 for pc in all_pair_counts if pc <= past_count)
                features['pair_count_pctile'] = rank / len(all_pair_counts)
            else:
                features['pair_count_pctile'] = 0
        else:
            features['pair_count_pctile'] = 0

        # NEW: Dst recency among src's interactions
        # How recently did src interact with dst relative to src's last interaction?
        if past_count > 0 and src_past_count > 0:
            pair_last = pt_arr[past_count - 1]
            src_last = src_t_arr[src_past_count - 1]
            features['pair_recency_vs_src'] = (query_time - pair_last) / (query_time - src_last + 1)
        else:
            features['pair_recency_vs_src'] = 1.0

        # NEW: Dst recent popularity (how many unique srcs interacted with dst recently?)
        dst_recent_srcs = set(dst_past_srcs[-20:])
        features['dst_recent_unique_src'] = len(dst_recent_srcs)

        return features


def build_training_data(extractor, train_df, n_neg=5, sample_size=100000):
    """Build training data with RANDOM negative sampling (same as earlier tree)."""
    srcs = train_df['src'].values
    dsts = train_df['dst'].values
    times = train_df['time'].values

    n = len(train_df)
    sample_size = min(sample_size, n)
    indices = np.random.choice(n, sample_size, replace=False)
    indices.sort()

    all_features = []
    all_labels = []

    print(f'Building training data from {sample_size} interactions...')
    for idx in tqdm(indices, desc='Training data', ncols=100):
        s, d, t = int(srcs[idx]), int(dsts[idx]), float(times[idx])

        # Positive example
        feat_pos = extractor.extract_features(s, d, t)
        all_features.append(feat_pos)
        all_labels.append(1)

        # Random negative examples (NOT from src history - that causes regression)
        for _ in range(n_neg):
            neg_d = np.random.randint(0, extractor._max_node + 1)
            while neg_d == d:
                neg_d = np.random.randint(0, extractor._max_node + 1)
            feat_neg = extractor.extract_features(s, neg_d, t)
            all_features.append(feat_neg)
            all_labels.append(0)

    return pd.DataFrame(all_features), np.array(all_labels)


def train_lgbm(
    df_features,
    labels,
    val_df_features=None,
    val_labels=None,
    seed=20260724,
):
    """Train LightGBM with stronger params."""
    feature_names = list(df_features.columns)
    train_data = lgb.Dataset(df_features[feature_names], label=labels,
                              feature_name=feature_names)

    params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': seed,
        'feature_fraction_seed': seed,
        'bagging_seed': seed,
        'data_random_seed': seed,
    }

    valid_sets = [train_data]
    valid_names = ['train']
    if val_df_features is not None:
        val_data = lgb.Dataset(val_df_features[feature_names], label=val_labels,
                                feature_name=feature_names)
        valid_sets.append(val_data)
        valid_names.append('val')

    model = lgb.train(
        params, train_data,
        num_boost_round=500,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    return model


def _mp_init_worker(ext_path):
    """Worker init: each process loads extractor from disk (avoids IPC of huge objects)."""
    global _worker_ext
    import pickle as _pkl
    with open(ext_path, 'rb') as f:
        _worker_ext = _pkl.load(f)


def _mp_extract_batch(args):
    """Extract features for a batch of (src, dst, time) triples in worker process."""
    srcs, dsts, times = args
    features = []
    for i in range(len(srcs)):
        feat = _worker_ext.extract_features(int(srcs[i]), int(dsts[i]), float(times[i]))
        features.append(feat)
    return features


def predict_scores(extractor, model, test_src, test_time, test_candidates,
                   batch_size=1000, n_workers=0):
    """Predict scores. n_workers=0 for single-process (safe), 2-4 for multiprocessing."""
    feature_names = model.feature_name()
    n_test = len(test_src)
    n_cands = test_candidates.shape[1]
    all_scores = np.zeros((n_test, n_cands))

    if n_workers > 0:
        # === Multiprocessing mode (use sparingly in WSL2, max 4 workers) ===
        n_workers = min(n_workers, 4)
        import pickle as _pkl
        print(f'Saving extractor for {n_workers} workers...')
        with tempfile.NamedTemporaryFile(
            prefix="dataset2_tree_",
            suffix=".pkl",
            delete=False,
        ) as temporary:
            ext_save_path = temporary.name
            _pkl.dump(extractor, temporary)
        ext_mb = os.path.getsize(ext_save_path) / 1e6
        print(f'Extractor: {ext_mb:.0f} MB x {n_workers} workers = ~{ext_mb * n_workers:.0f} MB')
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_workers, initializer=_mp_init_worker,
                      initargs=(ext_save_path,)) as pool:
            for start in tqdm(range(0, n_test, batch_size), desc='Predicting', ncols=100):
                end = min(start + batch_size, n_test)
                batch_scores = np.zeros((end - start, n_cands))
                args_list = [(test_src[start:end], test_candidates[start:end, j],
                              test_time[start:end]) for j in range(n_cands)]
                results = pool.map(_mp_extract_batch, args_list)
                for j, features in enumerate(results):
                    df_feat = pd.DataFrame(features)[feature_names]
                    batch_scores[:, j] = model.predict(df_feat)
                all_scores[start:end] = batch_scores
        if os.path.exists(ext_save_path):
            os.remove(ext_save_path)
    else:
        # === Single-process mode (safe, no extra memory, recommended for WSL2) ===
        print('Single-process prediction (safe mode). Use --n_workers 2~4 for parallel.')
        for start in tqdm(range(0, n_test, batch_size), desc='Predicting', ncols=100):
            end = min(start + batch_size, n_test)
            batch_scores = np.zeros((end - start, n_cands))
            for j in range(n_cands):
                features = []
                for i in range(end - start):
                    feat = extractor.extract_features(
                        int(test_src[start + i]),
                        int(test_candidates[start + i, j]),
                        float(test_time[start + i]))
                    features.append(feat)
                df_feat = pd.DataFrame(features)[feature_names]
                batch_scores[:, j] = model.predict(df_feat)
            all_scores[start:end] = batch_scores

    row_min = all_scores.min(axis=1, keepdims=True)
    row_max = all_scores.max(axis=1, keepdims=True)
    row_range = row_max - row_min
    row_range[row_range == 0] = 1.0
    return (all_scores - row_min) / row_range


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument(
        '--save_dir', type=str,
        default='outputs/training/dataset2/tree/models',
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='outputs/training/dataset2/tree/predictions',
    )
    parser.add_argument('--n_neg', type=int, default=5)
    parser.add_argument('--n_train', type=int, default=100000)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260724)
    parser.add_argument('--inference_only', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)

    ds = args.dataset
    print('=' * 70)
    print(f'Enhanced LightGBM tree - {ds}')
    print('=' * 70)

    df = pd.read_csv(f'{args.data_dir}/{ds}/train.csv')
    test_df = pd.read_csv(f'{args.data_dir}/{ds}/test.csv')

    test_src = test_df['src'].values
    test_time = test_df['time'].values.astype(np.float64)
    test_cands = test_df.iloc[:, 2:].values
    print(f'Train: {len(df)}, Test: {len(test_df)}')

    save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)
    model_path = f'{save_path}/{ds}_tree_ranker.txt'

    if not args.inference_only:
        # Use 80/20 split
        n_train = int(len(df) * 0.8)
        train_part = df.iloc[:n_train]
        val_part = df.iloc[n_train:]

        extractor = TreeFeatureExtractor(train_part)
        train_features, train_labels = build_training_data(
            extractor, train_part, n_neg=args.n_neg, sample_size=args.n_train)
        print(f'Train: {len(train_labels)} samples (pos={train_labels.sum()}, neg={(1-train_labels).sum()})')

        val_extractor = TreeFeatureExtractor(df.iloc[:n_train + len(val_part)])
        val_features, val_labels = build_training_data(
            val_extractor, val_part, n_neg=3, sample_size=50000)
        print(f'Val: {len(val_labels)} samples')

        print('\nTraining LightGBM tree...')
        model = train_lgbm(
            train_features,
            train_labels,
            val_features,
            val_labels,
            seed=args.seed,
        )

        # Feature importance
        importance = model.feature_importance(importance_type='gain')
        feat_names = model.feature_name()
        feat_imp = sorted(zip(feat_names, importance), key=lambda x: -x[1])
        print('\nTop 15 features:')
        for name, imp in feat_imp[:15]:
            print(f'  {name}: {imp:.1f}')

        model.save_model(model_path)
        print(f'Model saved: {model_path}')

    # Load model
    if not os.path.exists(model_path):
        print(f'ERROR: Model not found: {model_path}')
        return
    model = lgb.Booster(model_file=model_path)
    print(f'Loaded model: {model_path}')

    # Rebuild extractor with full data
    print('\nRebuilding extractor with full training data...')
    full_extractor = TreeFeatureExtractor(df)

    print('\nGenerating predictions...')
    t0 = time.time()
    scores = predict_scores(full_extractor, model, test_src, test_time, test_cands,
                            n_workers=args.n_workers)
    print(f'Done in {(time.time()-t0)/60:.1f} min')
    print(f'Scores: {scores.shape}, [{scores.min():.6f}, {scores.max():.6f}]')

    output_file = f'{args.output_dir}/{ds}/{ds}_tree_scores.csv'
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for row in scores:
            f.write(','.join([f'{p:.8f}' for p in row]) + '\n')
    print(f'Saved: {output_file}')


if __name__ == '__main__':
    main()
