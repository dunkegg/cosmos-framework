from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from wzj_tools.reward_model.conditioned import ConditionedRankingDataset, InstructionTemporalRewardModel

NAMES = ('gt', 'mild', 'medium', 'bad')

@torch.no_grad()
def run(model, loader, device, mode, shuffled=None, mean_emb=None):
    model.eval()
    count = 0
    correct = {k: 0 for k in NAMES[1:]}
    margins = {k: 0.0 for k in NAMES[1:]}
    top1 = 0
    winners = {k: 0 for k in NAMES}
    swap_offset = 0
    amp = device.type == 'cuda'
    for batch in loader:
        emb = batch['instruction_embedding']
        bs = emb.shape[0]
        if mode == 'shuffle':
            emb = shuffled[swap_offset:swap_offset+bs]
        elif mode == 'zero':
            emb = torch.zeros_like(emb)
        elif mode == 'mean':
            emb = mean_emb.expand(bs, -1)
        swap_offset += bs
        emb = emb.to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp):
            scores = {k: model(batch[k].to(device, non_blocking=True), emb).float() for k in NAMES}
        stacked = torch.stack([scores[k] for k in NAMES], dim=1)
        chosen = stacked.argmax(dim=1)
        count += bs
        top1 += (chosen == 0).sum().item()
        for i, k in enumerate(NAMES):
            winners[k] += (chosen == i).sum().item()
        for k in NAMES[1:]:
            correct[k] += (scores['gt'] > scores[k]).sum().item()
            margins[k] += (scores['gt'] - scores[k]).sum().item()
    return {
        'num_groups': count,
        'acc_overall': sum(correct.values()) / (3 * count),
        'gt_top1': top1 / count,
        **{f'acc_{k}': correct[k] / count for k in NAMES[1:]},
        **{f'margin_{k}': margins[k] / count for k in NAMES[1:]},
        'winner_counts': winners,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--embeddings', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', default='')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--shuffle-seed', type=int, default=1234)
    p.add_argument('--shuffle-repeats', type=int, default=5)
    a = p.parse_args()
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['args']
    dataset = ConditionedRankingDataset(
        embedding_path=a.embeddings, manifest_path=a.manifest,
        split='val', val_ratio=cfg['val_ratio'], split_seed=cfg['split_seed'],
    )
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, pin_memory=torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InstructionTemporalRewardModel(
        instruction_dim=ckpt['instruction_dim'], hidden_dim=cfg['hidden_dim']
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    # Build all validation embeddings in exactly the same order as DataLoader (shuffle=False).
    embs = torch.stack([
        dataset.embeddings[row['_instruction_key']].float() for row in dataset.rows
    ])
    assert embs.shape == (len(dataset), ckpt['instruction_dim']), embs.shape
    mean_emb = embs.mean(dim=0, keepdim=True)
    results = {}
    for mode in ('normal', 'zero', 'mean'):
        results[mode] = run(model, loader, device, mode, mean_emb=mean_emb)
        print(f'[{mode}] {json.dumps(results[mode], ensure_ascii=False)}', flush=True)
    # Shuffle distinct instruction keys (not merely row positions) to avoid matching duplicates.
    keys = [row['_instruction_key'] for row in dataset.rows]
    unique_keys = list(dict.fromkeys(keys))
    if len(unique_keys) < 2:
        raise ValueError('Need at least two distinct instruction texts for shuffle ablation')
    for rep in range(a.shuffle_repeats):
        rng = random.Random(a.shuffle_seed + rep)
        perm = unique_keys.copy()
        rng.shuffle(perm)
        # Ensure no unique text maps to itself.
        if any(x == y for x, y in zip(unique_keys, perm)):
            shift = rng.randrange(1, len(unique_keys))
            perm = unique_keys[shift:] + unique_keys[:shift]
        keymap = dict(zip(unique_keys, perm))
        assert all(keymap[k] != k for k in unique_keys)
        swapped = torch.stack([dataset.embeddings[keymap[k]].float() for k in keys])
        name = f'shuffle_{rep}'
        results[name] = run(model, loader, device, 'shuffle', shuffled=swapped)
        print(f'[{name}] {json.dumps(results[name], ensure_ascii=False)}', flush=True)
    summary = {}
    for metric in ('acc_overall', 'gt_top1', 'acc_mild', 'acc_medium', 'acc_bad'):
        vals = [results[f'shuffle_{i}'][metric] for i in range(a.shuffle_repeats)]
        summary[metric] = {'normal': results['normal'][metric],
                           'shuffle_mean': sum(vals) / len(vals),
                           'shuffle_min': min(vals), 'shuffle_max': max(vals),
                           'zero': results['zero'][metric], 'mean': results['mean'][metric]}
    print('\n[comparison]')
    for metric, x in summary.items():
        print(f'{metric:12s} normal={x["normal"]:.4f} shuffle={x["shuffle_mean"]:.4f} '
              f'zero={x["zero"]:.4f} mean={x["mean"]:.4f}')
    output = a.output or str(Path(a.checkpoint).with_name('instruction_ablation.json'))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump({'checkpoint': a.checkpoint, 'epoch': ckpt['epoch'],
                   'shuffle_seed': a.shuffle_seed, 'shuffle_repeats': a.shuffle_repeats,
                   'results': results, 'comparison': summary}, f, indent=2)
    print('[saved]', output)

if __name__ == '__main__':
    main()