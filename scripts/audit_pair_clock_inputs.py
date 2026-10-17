"""Prepare and independently check train/validation cadence inputs; never fit."""
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import tomllib
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.benchmark import load_dataset, save_json, sha256
from lifecycle_readout.causal_stream import build_candidate_stream
from lifecycle_readout.pair_clock import augment_pair_clock


def independently_check(data, stream, augmented, metadata, raw):
    """Use separate per-pair timestamp arrays and binary searches, not a tracker."""
    histories = defaultdict(list)
    excluded = set(stream.excluded_training_ids.tolist())
    for event, t in enumerate(data['timestamps']):
        if t > stream.validation_end:
            break
        if event not in excluded:
            histories[(int(data['sources'][event]), int(data['destinations'][event]))].append(float(t))
    histories = {key: np.asarray(value) for key, value in histories.items()}
    wanted = stream.split <= 1
    mean = np.full(raw['mean_interval_seconds'].shape, np.nan)
    elapsed = np.zeros(mean.shape)
    count = np.zeros(mean.shape, dtype=np.int64)
    for row in np.flatnonzero(wanted):
        event = stream.event_ids[row]
        u, t = int(data['sources'][event]), data['timestamps'][event]
        for slot, dst in enumerate([data['destinations'][event], stream.negative_destinations[row]]):
            history = histories.get((u, int(dst)), np.array([]))
            n = np.searchsorted(history, t, side='left')
            count[row, slot] = n
            if n:
                elapsed[row, slot] = t - history[n-1]
            if n >= 2 and history[n-1] > history[0]:
                mean[row, slot] = (history[n-1] - history[0]) / (n-1)
    np.testing.assert_array_equal(raw['queried'], wanted)
    np.testing.assert_array_equal(raw['prior_count'], count)
    np.testing.assert_allclose(raw['mean_interval_seconds'], mean, rtol=0, atol=0, equal_nan=True)
    valid = np.isfinite(mean) & (mean > 0)
    np.testing.assert_array_equal(raw['cadence_available'], valid)
    assert metadata['train_candidate_slots']==int(2*(stream.split==0).sum())
    assert metadata['train_available_slots']==int(valid[stream.split==0].sum())
    assert metadata['validation_candidate_slots']==int(2*(stream.split==1).sum())
    assert metadata['validation_available_slots']==int(valid[stream.split==1].sum())
    fallback = float(np.median(mean[(stream.split == 0)[:, None] & valid]))
    assert fallback == metadata['fallback_seconds']
    effective = np.where(valid, mean, fallback)
    expected = np.stack([np.log1p(effective/fallback), valid.astype(float),
                         np.log1p(elapsed/(effective + metadata['tau0_seconds']))], axis=-1)
    expected[~wanted] = 0
    np.testing.assert_array_equal(augmented.pair_features[:, :, 6:], expected.astype(np.float32))
    np.testing.assert_array_equal(augmented.pair_features[:, :, :6], stream.pair_features)
    for key in ['event_ids', 'negative_destinations', 'elapsed_times', 'split', 'excluded_training_ids',
                'skipped_no_negative_ids', 'neighbor_edges']:
        np.testing.assert_array_equal(getattr(augmented, key), getattr(stream, key))
    return dict(status='PASS', checked_candidate_slots=int(2*wanted.sum()),
                test_clocks_constructed=False, fallback_train_only=True,
                all_added_feature_values_checked=True, sampling_and_support_unchanged=True)


def verify_saved_clock(data, candidates, folder, settings, protocol):
    """Audit saved clocks from a completed arm, including its masked control."""
    metadata=json.loads((folder/'pair_clock.json').read_text())
    with np.load(folder/'pair_clock.npz',allow_pickle=False) as z:
        raw={k:z[k] for k in z.files}
    assert metadata['test_clocks_constructed'] is False
    assert metadata['sampling_or_support_changed'] is False
    assert metadata['tau0_seconds']==settings['pair_clock_tau0_seconds']==protocol['tau0_seconds']
    masked=settings.get('mask_pair_clock_features',False)
    assert metadata['features_masked_for_capacity_control']==masked
    assert candidates['pair_features'].shape==(len(candidates['event_ids']),2,9)
    effective=np.where(raw['cadence_available'],raw['mean_interval_seconds'],metadata['fallback_seconds'])
    extra=np.stack([np.log1p(effective/metadata['fallback_seconds']),
                    raw['cadence_available'].astype(float),raw['clock']],axis=-1).astype(np.float32)
    extra[~raw['queried']]=0
    actual=candidates['pair_features'][:,:,6:]
    if masked:
        np.testing.assert_array_equal(actual,np.zeros_like(actual))
    else:
        np.testing.assert_array_equal(actual,extra)
    # Neighbor snapshots are not present in candidate_identity.npz. This helper
    # audits pair clocks only; full neighbor reconstruction is a separate check.
    fields=dict(event_ids=candidates['event_ids'],negative_destinations=candidates['negative_destinations'],
        elapsed_times=candidates['elapsed_times'],split=candidates['split'],
        excluded_training_ids=candidates['excluded_training_ids'],
        skipped_no_negative_ids=candidates['skipped_no_hard_support_ids'],
        neighbor_edges=np.array([],dtype=np.int64),
        validation_end=float(np.quantile(data['timestamps'],sum(protocol['split_chronological'][:2]))))
    stream=SimpleNamespace(**fields,pair_features=candidates['pair_features'][:,:,:6])
    augmented=SimpleNamespace(**fields,pair_features=np.concatenate([stream.pair_features,extra],axis=-1))
    return independently_check(data,stream,augmented,metadata,raw)


def prepare(dataset, config, output):
    started = time.monotonic()
    settings = tomllib.loads(config.read_text())
    protocol = tomllib.loads((ROOT / settings['protocol']).read_text())
    assert protocol['protocol_id'] == 'LCR-P-CADENCE-DEV-001'
    assert settings['evaluation_split'] == 'validation' and protocol['test_scoring_permitted'] is False
    assert settings['seeds'] == protocol['seeds'] and dataset in protocol['datasets']
    output.mkdir(parents=True, exist_ok=False)
    data, identity = load_dataset(dataset, ROOT / 'resources/corpora')
    reports = []
    for seed in settings['seeds']:
        stream = build_candidate_stream(**data, seed=seed, split=protocol['split_chronological'],
            split_seed=settings['split_seed'], unseen_ratio=protocol['inductive_unseen_node_ratio'],
            max_neighbors=settings['max_neighbors'], history_decay=settings['history_decay'],
            negative_sampling='matched_historical', hard_time_unit=settings['hard_time_unit'])
        augmented, metadata, raw = augment_pair_clock(data, stream, tau0_seconds=settings['pair_clock_tau0_seconds'])
        check = independently_check(data, stream, augmented, metadata, raw)
        selected = stream.split <= 1
        target = output / f'seed{seed}_train_validation_clock.npz'
        np.savez_compressed(target, event_ids=stream.event_ids[selected],
            negative_destinations=stream.negative_destinations[selected], split=stream.split[selected],
            pair_features=augmented.pair_features[selected], elapsed_times=stream.elapsed_times[selected],
            **{k:v[selected] for k,v in raw.items()})
        prior = None
        if dataset == 'wikipedia':
            old = ROOT/'results/audit/lcr-wikipedia-bdec2912ae/artifacts'/f'seed{seed}/candidate_identity.npz'
            if old.exists():
                with np.load(old) as z:
                    for key in ['event_ids', 'negative_destinations', 'pair_features', 'elapsed_times', 'split']:
                        np.testing.assert_array_equal(z[key], getattr(stream, key))
                prior = dict(candidate_file_sha256=sha256(old), identical_prior_candidates_and_six_features=True)
        record = dict(seed=seed, raw_stream_sha256=stream.digest(),
            augmented_stream_sha256=augmented.digest(), metadata=metadata, independent_check=check,
            train_validation_archive_sha256=sha256(target), prior_wikipedia_development=prior)
        reports.append(record)
        save_json(output/'partial_report.json',dict(dataset=dataset,seeds=reports,model_fitted=False))
        print(f'Clock input audit PASS: {dataset} seed={seed}',flush=True)
    save_json(output/'report.json',dict(status='PASS',dataset=dataset,protocol_id=protocol['protocol_id'],
        config_sha256=sha256(config),protocol_sha256=sha256(ROOT/settings['protocol']),data=identity,
        seeds=reports,model_fitted=False,new_test_metric=False,test_cadences_constructed=False,
        publication_eligible=False,elapsed_seconds=time.monotonic()-started,
        auditor_sha256=sha256(Path(__file__)),
        limitation='Feature implementation and history identity only; no evidence of predictive value or novelty.'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['wikipedia','reddit'],required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/pair_cadence_features_v1.toml')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    prepare(args.dataset,args.config,args.output)
