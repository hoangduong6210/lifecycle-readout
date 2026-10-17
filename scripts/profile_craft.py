"""Bounded author-model feasibility profile, never a predictive efficacy result."""
from pathlib import Path
import argparse
import json
import sys
import time
import tomllib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.craft_adapter import StrictPastCraftHistory, make_author_model
from lifecycle_readout.benchmark import save_json, sha256


def run(settings, output):
    if not torch.cuda.is_available():
        raise RuntimeError('Full-input profiling requires Colab CUDA')
    for item in settings['input_files']:
        if sha256(ROOT / item['path']) != item['sha256']:
            raise ValueError('Input identity mismatch: ' + item['path'])
    with np.load(ROOT / settings['prepared_path']) as stored:
        data = {k: stored[k] for k in stored.files}
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    history = StrictPastCraftHistory(data)
    build_seconds = time.monotonic() - started
    train = np.flatnonzero(data['split'] == 0)
    catalog = np.unique(data['dst'][train])
    with np.load(ROOT / settings['candidates_path']) as stored:
        candidate_data = {k: stored[k] for k in stored.files}
    validation = np.flatnonzero(data['split'] == 1)
    candidate_indices = candidate_data.get('validation_rows', np.arange(len(validation)))
    selected_validation = validation[candidate_indices]
    np.testing.assert_array_equal(data['event_ids'][selected_validation], candidate_data['event_ids'])
    np.testing.assert_array_equal(data['dst'][selected_validation], candidate_data['positive_destinations'])
    np.testing.assert_array_equal(data['src'][selected_validation], candidate_data['sources'])
    np.testing.assert_array_equal(data['time'][selected_validation], candidate_data['timestamps'])
    seed = settings['seeds'][0]
    profile = []
    for batch_size in settings['train_batch_sizes']:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        rng = np.random.RandomState(seed)
        model = make_author_model(ROOT / settings['author_path'], settings,
                                  item_count=int(data['dst'].max()), device='cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
        torch.cuda.reset_peak_memory_stats()
        elapsed = []
        for step in range(settings['warmup_steps'] + settings['measured_steps']):
            torch.cuda.synchronize(); begin = time.monotonic()
            rows = rng.choice(train, batch_size, replace=True)
            negative = rng.choice(catalog, batch_size, replace=True)
            lo = np.searchsorted(data['time'], data['time'][rows], side='left')
            hi = np.searchsorted(data['time'], data['time'][rows], side='right')
            for i, (row, start, stop) in enumerate(zip(rows, lo, hi)):
                forbidden = set(data['dst'][start:stop][data['src'][start:stop] == data['src'][row]])
                if len(forbidden) >= len(catalog) and all(int(v) in forbidden for v in catalog):
                    raise ValueError('No eligible training negative')
                while negative[i] in forbidden:
                    negative[i] = rng.choice(catalog)
            inputs = history.batch(data['src'][rows], data['time'][rows],
                                   np.c_[data['dst'][rows], negative], settings['history_length'], 'cuda')
            optimizer.zero_grad()
            loss, _, _ = model.calculate_loss(*inputs)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite author BPR loss')
            loss.backward(); optimizer.step()
            torch.cuda.synchronize()
            if step >= settings['warmup_steps']:
                elapsed.append(time.monotonic() - begin)
        profile.append(dict(train_batch_size=batch_size, measured_steps=len(elapsed),
            seconds_per_step_mean=float(np.mean(elapsed)), seconds_per_step_samples=elapsed,
            projected_train_epoch_seconds=float(np.mean(elapsed) * np.ceil(len(train) / batch_size)),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            parameter_count=sum(p.numel() for p in model.parameters()),
            projection_boundary='Timing projection only; excludes validation, early stopping, checkpointing and warmup.'))
        print('Profile completed:', batch_size, profile[-1]['seconds_per_step_mean'], flush=True)
    model.eval()
    anchors = np.linspace(0, len(candidate_indices) - 1, settings['validation_query_count'], dtype=int)
    candidates = np.c_[candidate_data['positive_destinations'][anchors],
                       candidate_data['negative_destinations'][anchors]]
    inputs = history.batch(candidate_data['sources'][anchors], candidate_data['timestamps'][anchors],
                           candidates, settings['history_length'], 'cuda')
    torch.cuda.synchronize(); begin = time.monotonic()
    with torch.no_grad():
        positive, negative = model.predict(*inputs)
    torch.cuda.synchronize(); eval_seconds = time.monotonic() - begin
    assert torch.isfinite(positive).all() and torch.isfinite(negative).all()
    np.savez_compressed(output / 'roundtrip.npz',
        **{f'input_{i}': x.cpu().numpy() for i, x in enumerate(inputs)},
        positive=positive.cpu().numpy(), negative=negative.cpu().numpy(), validation_rows=candidate_indices[anchors])
    torch.save(model.state_dict(), output / 'roundtrip_state.pt')
    protocol = tomllib.loads((ROOT / settings['protocol']).read_text())
    result = dict(protocol_id=protocol['protocol_id'], dataset='ML-20M', smoke=True,
        purpose='Runtime and interface feasibility only; partial random-batch fits are not efficacy estimates.',
        settings=settings, publication_eligible=False, test_scored=False, mrr_computed=False,
        history_build_seconds=build_seconds, training_profiles=profile,
        validation_profile=dict(query_count=len(anchors), candidates_per_query=candidates.shape[1],
                                gpu_forward_seconds=eval_seconds),
        per_seed_runs=[dict(seed=seed, models={'CRAFT': {'feasibility_profile_only': True}})])
    save_json(output / 'results.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int)
    args = parser.parse_args()
    settings = tomllib.loads(args.config.read_text())
    if (args.dataset != 'ML-20M' or args.device != 'cuda'
            or (args.seeds is not None and args.seeds != settings['seeds'])):
        parser.error('Undeclared dataset/device/seed')
    run(settings, args.output_dir)
