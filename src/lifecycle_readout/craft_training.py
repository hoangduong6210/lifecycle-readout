"""Full-support validation development fit of the pinned author architecture."""
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .benchmark import save_json, sha256
from .craft_adapter import StrictPastCraftHistory, make_author_model
from .ranking import reciprocal_ranks


def training_negatives(data, rows, rng, catalog):
    """One uniform catalogue draw, rejecting all same-source/time positives."""
    negative = rng.choice(catalog, len(rows), replace=True)
    lo = np.searchsorted(data['time'], data['time'][rows], side='left')
    hi = np.searchsorted(data['time'], data['time'][rows], side='right')
    for i, (row, start, stop) in enumerate(zip(rows, lo, hi)):
        forbidden = set(data['dst'][start:stop][data['src'][start:stop] == data['src'][row]])
        if len(forbidden) >= len(catalog) and all(int(v) in forbidden for v in catalog):
            raise ValueError('No eligible training negative')
        while negative[i] in forbidden:
            negative[i] = rng.choice(catalog)
    return negative


def evaluate(model, history, candidates, length, batch_size, device):
    model.eval()
    n = len(candidates['event_ids'])
    positive = np.empty(n, dtype=np.float32)
    negative = np.empty(candidates['negative_destinations'].shape, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            items = np.c_[candidates['positive_destinations'][start:stop],
                          candidates['negative_destinations'][start:stop]]
            inputs = history.batch(candidates['sources'][start:stop], candidates['timestamps'][start:stop],
                                   items, length, device)
            p, q = model.predict(*inputs)
            positive[start:stop], negative[start:stop] = p.cpu().numpy(), q.reshape(stop - start, -1).cpu().numpy()
    rr = reciprocal_ranks(positive, negative)
    return positive, negative, rr


def fit(data, candidates, settings, author_folder, output, device='cuda', technical_smoke=False):
    if device != 'cuda' and (not technical_smoke or len(data['src']) > 1000):
        raise ValueError('Full training requires bounded Colab CUDA; CPU is tiny technical smoke only')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    history = StrictPastCraftHistory(data)
    train = np.flatnonzero(data['split'] == 0)
    val = np.flatnonzero(data['split'] == 1)
    if not len(train) or not len(val) or data['time'][train].max() >= data['time'][val].min():
        raise ValueError('Strict train/validation boundary required')
    for saved, raw in [('event_ids', 'event_ids'), ('sources', 'src'),
                       ('timestamps', 'time'), ('positive_destinations', 'dst')]:
        np.testing.assert_array_equal(candidates[saved], data[raw][val])
    assert candidates['negative_destinations'].shape == (len(val), settings['negative_count'])
    catalog = np.unique(data['dst'][train])
    result = dict(protocol_id=settings['protocol_id'], dataset='ML-20M', smoke=technical_smoke,
        evaluation_split='validation', test_scored=False, publication_eligible=False,
        settings=settings, train_count=len(train), validation_count=len(val), per_seed_runs=[])
    for seed in settings['seeds']:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)
        order_rng, negative_rng = np.random.RandomState(seed), np.random.RandomState(seed + 100000)
        model = make_author_model(author_folder, settings, int(data['dst'].max()), device)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'], weight_decay=0.)
        best, stale, selected, epochs = -np.inf, 0, None, []
        stop_reason = 'max_epochs'
        completed_epochs = 0
        for epoch in range(1, settings['max_epochs'] + 1):
            if time.monotonic() - started >= settings['training_budget_seconds']:
                stop_reason = 'runtime_cap_between_epochs'; break
            model.train(); epoch_started = time.monotonic()
            order = order_rng.permutation(train)
            trace = hashlib.sha256(); loss_sum, observed = 0., 0
            incomplete = False
            for start in range(0, len(order), settings['train_batch_size']):
                if time.monotonic() - started >= settings['training_budget_seconds']:
                    incomplete = True; break
                rows = order[start:start + settings['train_batch_size']]
                negative = training_negatives(data, rows, negative_rng, catalog)
                trace.update(np.c_[data['event_ids'][rows], negative].astype('<i8').tobytes())
                inputs = history.batch(data['src'][rows], data['time'][rows], np.c_[data['dst'][rows], negative],
                                       settings['history_length'], device)
                optimizer.zero_grad()
                loss, _, _ = model.calculate_loss(*inputs)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward(); optimizer.step()
                loss_sum += float(loss.detach()) * len(rows); observed += len(rows)
                if observed // 1000000 > (observed - len(rows)) // 1000000:
                    save_json(output / 'progress.json', dict(seed=seed, epoch=epoch, epoch_rows=observed,
                        total_epoch_rows=len(train), elapsed_seconds=time.monotonic() - started))
                    print(f'CRAFT seed={seed} epoch={epoch} trained={observed}/{len(train)}', flush=True)
            if incomplete:
                torch.save(model.state_dict(), output / f'partial_seed{seed}_epoch{epoch}.pt')
                save_json(output / f'partial_seed{seed}_epoch{epoch}.json', dict(rows=observed,
                    total_epoch_rows=len(train), candidate_trace_sha256=trace.hexdigest(), epoch_completed=False))
                stop_reason = 'runtime_cap_partial_epoch'; break
            train_seconds = time.monotonic() - epoch_started
            positive, negative, rr = evaluate(model, history, candidates, settings['history_length'],
                                              settings['validation_batch_size'], device)
            mrr = float(rr.mean())
            np.savez_compressed(output / f'seed{seed}_epoch{epoch}_validation_rr.npz', reciprocal_rank=rr)
            improved = mrr > best
            if improved:
                best, selected, stale = mrr, epoch, 0
                torch.save(model.state_dict(), output / f'best_seed{seed}.pt')
            else:
                stale += 1
            torch.save(model.state_dict(), output / f'last_complete_seed{seed}.pt')
            epochs.append(dict(epoch=epoch, train_count=observed, train_loss=loss_sum / observed,
                training_seconds=train_seconds, elapsed_seconds=time.monotonic() - started,
                validation_mrr=mrr, candidate_trace_sha256=trace.hexdigest(), selected_at_this_epoch=improved))
            completed_epochs += 1
            save_json(output / f'seed{seed}_history.json', epochs)
            print(f'CRAFT seed={seed} epoch={epoch} validation_mrr={mrr:.8f} best_epoch={selected}', flush=True)
            if stale >= settings['patience']:
                stop_reason = 'validation_patience'; break
        if selected is None:
            raise RuntimeError('No complete epoch/validation checkpoint; partial artifacts retained')
        model.load_state_dict(torch.load(output / f'best_seed{seed}.pt', map_location=device, weights_only=True))
        positive, negative, rr = evaluate(model, history, candidates, settings['history_length'],
                                          settings['validation_batch_size'], device)
        assert float(rr.mean()) == best
        prediction = output / f'CRAFT_seed{seed}_validation_predictions.npz'
        np.savez_compressed(prediction, event_ids=candidates['event_ids'], positive=positive,
                            negative=negative, reciprocal_rank=rr)
        result['per_seed_runs'].append(dict(seed=seed, selected_epoch=selected, completed_epochs=completed_epochs,
            stop_reason=stop_reason, validation_history=epochs,
            models={'CRAFT': dict(validation_mrr=best, query_count=len(rr),
                prediction_sha256=sha256(prediction), checkpoint_sha256=sha256(output / f'best_seed{seed}.pt'),
                parameter_count=sum(p.numel() for p in model.parameters()))}))
        save_json(output / 'results.json', result)
    result.update(complete_declared_seeds=len(result['per_seed_runs']) == len(settings['seeds']),
                  elapsed_seconds=time.monotonic() - started,
                  convergence_demonstrated=False, publication_eligible=False)
    save_json(output / 'results.json', result)
    return result
