"""
Inference-only script: load trained best model and generate competition results.
No training, just prediction.
"""
import os
import os.path as osp
import sys
import time

# Use installed jittor_geometric package
os.environ['JT_SYNC'] = '1'

import jittor as jt
import numpy as np
import pandas as pd
from tqdm import tqdm
from jittor_geometric.data import TemporalData
from jittor_geometric.nn.models.craft import CRAFT
from jittor_geometric.dataloader.temporal_dataloader import get_neighbor_sampler
import argparse

jt.flags.use_cuda = 1


def test_competition(model, test_src, test_time, test_candidates,
                     full_neighbor_sampler, num_neighbors, batch_size=256):
    model.eval()
    all_scores = []
    num_samples = len(test_src)
    num_batches = (num_samples + batch_size - 1) // batch_size

    pbar = tqdm(range(num_batches), ncols=120, desc='Testing')
    for batch_idx in pbar:
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_samples)

        batch_src = test_src[start:end]
        batch_time = test_time[start:end]
        batch_cand = test_candidates[start:end]

        src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
            node_ids=batch_src, node_interact_times=batch_time, num_neighbors=num_neighbors)
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)

        test_dst = jt.Var(batch_cand)

        dst_last_neighbor, _, dst_last_update_time = full_neighbor_sampler.get_historical_neighbors_left(
            node_ids=test_dst.flatten().numpy(),
            node_interact_times=np.broadcast_to(batch_time[:, np.newaxis],
                                                 (len(batch_time), test_dst.shape[1])).flatten(),
            num_neighbors=1)
        dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
        dst_last_update_time[dst_last_neighbor.reshape(len(test_dst), -1) == 0] = -100000
        dst_last_update_time = jt.Var(dst_last_update_time)

        src_neighb_seq_adj = jt.Var(src_neighb_seq) - model.dst_min_idx + 1
        test_dst_adj = test_dst - model.dst_min_idx + 1
        src_neighb_seq_adj = jt.where(src_neighb_seq_adj < 0, jt.zeros_like(src_neighb_seq_adj),
                                       src_neighb_seq_adj)

        logits = model.forward(src_neighb_seq_adj, jt.Var(neighbor_num),
                               jt.Var(src_neighb_interact_times),
                               jt.Var(batch_time), test_dst=test_dst_adj,
                               dst_last_update_times=dst_last_update_time)
        probs = jt.sigmoid(logits.squeeze(-1)).numpy()
        all_scores.append(probs)

    return np.vstack(all_scores)


# Dataset configs (must match training config)
DATASET_CONFIGS = {
    'dataset1': {
        'num_neighbors': 50,
        'hidden_size': 128,
        'n_layers': 3,
        'n_heads': 4,
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'input_cat_time_intervals': False,
        'output_cat_time_intervals': True,
        'output_cat_repeat_times': True,
        'loss_type': 'BPR',
    },
    'dataset2': {
        # Anti-overfitting configuration: hidden=64, two layers/heads
        # and a 30-item history.
        'num_neighbors': 30,
        'hidden_size': 64,
        'n_layers': 2,
        'n_heads': 2,
        'dropout': 0.3,
        'emb_dropout': 0.3,
        'input_cat_time_intervals': False,
        'output_cat_time_intervals': True,
        'output_cat_repeat_times': False,
        'loss_type': 'BPR',
    },
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--data_dir', type=str, default='.')
parser.add_argument(
    '--save_dir', type=str, default='solution.base/models/dataset2'
)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--seed', type=int, default=20260724)
args = parser.parse_args()

if args.output_dir is None:
    args.output_dir = args.data_dir

cfg = DATASET_CONFIGS[args.dataset]
num_neighbors = cfg['num_neighbors']

print('=' * 70)
print(f'Inference: {args.dataset}')
print('=' * 70)

# Load data
df = pd.read_csv(f'{args.data_dir}/{args.dataset}/train.csv')
test_df = pd.read_csv(f'{args.data_dir}/{args.dataset}/test.csv')

src_np = df['src'].values.astype(np.int32)
dst_np = df['dst'].values.astype(np.int32)
t_np = df['time'].values.astype(np.int32)
edge_ids_np = np.arange(len(df), dtype=np.int32) + 1

test_src = test_df['src'].values.astype(np.int32)
test_time = test_df['time'].values.astype(np.int32)
test_candidates = test_df.iloc[:, 2:].values.astype(np.int32)

print(f'Train: {len(df)}, Test: {len(test_df)}')

# Build full data for neighbor sampling
full_data = TemporalData(
    src=jt.Var(src_np), dst=jt.Var(dst_np),
    t=jt.Var(t_np), edge_ids=jt.Var(edge_ids_np)
)
full_neighbor_sampler = get_neighbor_sampler(full_data, 'recent', seed=args.seed)

# Build model
max_node = max(int(src_np.max()), int(dst_np.max()), int(test_candidates.max()))
node_size = max_node + 1
dst_min = min(int(dst_np.min()), int(test_candidates.min()))
src_min = int(src_np.min())

model = CRAFT(
    n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
    hidden_size=cfg['hidden_size'],
    hidden_dropout_prob=cfg['dropout'], attn_dropout_prob=cfg['dropout'],
    hidden_act='gelu', layer_norm_eps=1e-12, initializer_range=0.02,
    n_nodes=node_size, max_seq_length=num_neighbors,
    loss_type=cfg['loss_type'], use_pos=True,
    input_cat_time_intervals=cfg['input_cat_time_intervals'],
    output_cat_time_intervals=cfg['output_cat_time_intervals'],
    output_cat_repeat_times=cfg['output_cat_repeat_times'],
    num_output_layer=1, emb_dropout_prob=cfg['emb_dropout'], skip_connection=True
)
model.set_min_idx(src_min, dst_min)

# Load best model
best_path = f'{args.save_dir}/{args.dataset}_CRAFT_best.pkl'
latest_path = f'{args.save_dir}/{args.dataset}_CRAFT.pkl'
if args.model is not None and os.path.exists(args.model):
    model.load_state_dict(jt.load(args.model))
    print(f'Loaded explicit model: {args.model}')
elif os.path.exists(best_path):
    model.load_state_dict(jt.load(best_path))
    print(f'Loaded best model: {best_path}')
elif os.path.exists(latest_path):
    model.load_state_dict(jt.load(latest_path))
    print(f'Loaded latest model: {latest_path}')
else:
    print(f'ERROR: No model found at {best_path} or {latest_path}')
    sys.exit(1)

# Generate predictions
print('\nGenerating predictions...')
t0 = time.time()
scores = test_competition(model, test_src, test_time, test_candidates,
                          full_neighbor_sampler, num_neighbors, args.batch_size)
print(f'Done in {(time.time()-t0)/60:.1f} min')
print(f'Scores shape: {scores.shape}, range: [{scores.min():.6f}, {scores.max():.6f}]')

# Save
output_file = f'{args.output_dir}/{args.dataset}/{args.dataset}_result.csv'
os.makedirs(osp.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    for row in scores:
        f.write(','.join([f'{p:.8f}' for p in row]) + '\n')

print(f'Results saved to: {output_file}')
print('=' * 70)
