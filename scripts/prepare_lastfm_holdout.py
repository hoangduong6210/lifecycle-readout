"""Prepare a fixed chronological window, without prediction or fitting."""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import math
import sys
import tomllib
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.causal_stream import CandidateStream
from lifecycle_readout.lastfm_holdout_data import append_frozen_window, check_stream_prefix
from lifecycle_readout.lastfm_model_stream import build_lastfm_stream
from prepare_lastfm_model import direct_history


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def anchor_audit(frame, stream, extra, settings, test_rows, primary, configuration):
    timestamps = frame.timestamp_ms.to_numpy()
    sources = frame.source.to_numpy()
    targets = frame.destination.to_numpy()
    records = []
    for label, eligible in [('primary', test_rows[primary]), ('complement', test_rows[~primary])]:
        selected = sorted(eligible.tolist(), key=lambda j: hashlib.sha256(
            (configuration['anchor_hash_prefix'] + str(int(stream.event_ids[j]))).encode()).digest()
        )[:configuration['anchors_per_primary_and_complement']]
        for j in selected:
            i = int(stream.event_ids[j]); user = int(sources[i]); time = int(timestamps[i])
            positive = int(targets[i]); negative = int(stream.negative_destinations[j]); pair = []
            assert negative not in set(targets[(sources == user) & (timestamps == time)])
            for slot, track in enumerate((positive, negative)):
                expected = direct_history(frame, user, track, time, settings['query_hash_seed'])
                np.testing.assert_array_equal(extra['raw'][j, slot], [expected[k] for k in (
                    'total', 'same', 'unknown', 'status', 'ref', 'count', 'alternate_available')])
                assert extra['alternate_exposure'][j, slot] == expected['alternate_exposure']
                np.testing.assert_allclose(stream.elapsed_times[j, slot], expected['gap_seconds'], rtol=1e-6, atol=1e-5)
                day = (time / 1000) % 86400
                gap = expected['gap_seconds']; count = expected['count']
                index = int(np.searchsorted(timestamps, time, side='left'))
                phi = [.1+.9*math.exp(-gap/3600), math.log1p(gap), math.log1p(count),
                       count/math.sqrt(max(1,index)), math.sin(2*math.pi*day/86400),
                       math.cos(2*math.pi*day/86400), math.log1p(expected['total']),
                       math.log1p(expected['same']), int(expected['status']==1),
                       math.log1p(expected['unknown']), expected['alternate_available']]
                np.testing.assert_allclose(stream.pair_features[j, slot], phi, rtol=1e-6, atol=1e-7)
                pair.append(expected)
            assert int(math.log2(pair[0]['count'])) == int(math.log2(pair[1]['count']))
            unit = settings['hard_time_unit_seconds']
            assert int(math.log2(1+pair[0]['gap_seconds']/unit)) == int(math.log2(1+pair[1]['gap_seconds']/unit))
            for slot, (kind, node) in enumerate((('u',user), ('v',positive), ('v',negative))):
                past = (timestamps < time) & ((sources == node) if kind == 'u' else ((targets == node) & (targets >= 0)))
                expected = np.flatnonzero(past)[-settings['max_neighbors']:]
                observed = stream.neighbor_edges[j,slot]
                np.testing.assert_array_equal(observed[observed >= 0], expected)
            records.append(dict(cohort=label, event_id=i, negative_destination=negative, expected=pair))
    return dict(status='PASS', anchors=records, scope='Fixed hash-ranked February primary/complement queries; direct full prepared-history features, bins, exclusions and typed graph snapshots. Not all-row raw lineage replay.')


def run(config, raw, archive, output):
    config = config.resolve(); output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        c = tomllib.loads(config.read_text())
        protocol = ROOT / c['protocol']; p = tomllib.loads(protocol.read_text())
        settings = tomllib.loads((ROOT / c['parent_config']).read_text())
        assert c['training_end_exclusive_utc'] == p['training_end_exclusive_utc']
        assert c['window_start_inclusive_utc'] == p['test_start_inclusive_utc']
        assert c['window_end_exclusive_utc'] == p['test_end_exclusive_utc']
        specification = json.loads((ROOT / c['specification']).read_text())
        assert all(sha(ROOT / name) == h for name,h in specification['source_input_hashes'].items())
        parent = ROOT / c['parent_prepared']
        parent_manifest = json.loads((parent / 'artifact_manifest.json').read_text())
        assert all(sha(parent / name) == h for name,h in parent_manifest.items())
        raw_before = raw.stat(); archive_before = archive.stat()
        assert raw_before.st_size == c['raw_bytes'] and sha(raw) == c['raw_sha256']
        assert archive_before.st_size == c['archive_bytes'] and sha(archive) == c['archive_sha256']
        source_paths = [config, protocol, ROOT/c['parent_config'], ROOT/c['specification']]
        source_paths += [ROOT/name for name in (
            'scripts/prepare_lastfm_holdout.py', 'scripts/prepare_lastfm_model.py',
            'src/lifecycle_readout/lastfm_holdout_data.py', 'src/lifecycle_readout/lastfm_model_stream.py',
            'src/lifecycle_readout/lastfm_intake.py', 'src/lifecycle_readout/causal_stream.py',
            'tests/test_lastfm_holdout_data.py')]
        pins = {str(path.relative_to(ROOT)):sha(path) for path in source_paths}
        save(output/'preanalysis.json', dict(protocol=p, configuration=c, source_pins=pins,
            parent_manifest_sha256=sha(parent/'artifact_manifest.json'),
            raw_sha256=c['raw_sha256'], archive_sha256=c['archive_sha256'],
            before_february_parse=True, model_fit=False, predictor_scored=False))
        for path in source_paths:
            target = output/'sources'/path.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(path.read_bytes())
        prior_data = load_arrays(parent/'data.npz')
        old_maps = json.loads((parent/'identity_maps.json').read_text())
        frame, data, maps, report = append_frozen_window(prior_data, old_maps, raw,
            c['window_start_inclusive_utc'], c['window_end_exclusive_utc'])
        for name in ('sampled_users','track_uuids','artist_uuids'):
            assert maps[name][:len(old_maps[name])] == old_maps[name]
        np.savez_compressed(output/'data.npz', **data)
        save(output/'identity_maps.json',maps); save(output/'window_report.json',report)
        print('FEBRUARY_DATA',json.dumps(report),flush=True)
        epoch = lambda text: int(datetime.fromisoformat(text).timestamp())*1000
        start = epoch(c['window_start_inclusive_utc'])
        train_end = epoch(c['training_end_exclusive_utc'])
        end = epoch(c['window_end_exclusive_utc'])
        per_seed = []
        for seed in settings['seeds']:
            stream, extra = build_lastfm_stream(frame,settings,seed=seed,train_end_ms=train_end,validation_end_ms=end)
            old_stream = CandidateStream(**load_arrays(parent/f'seed{seed}/candidate_stream.npz'))
            old_extra = load_arrays(parent/f'seed{seed}/artist_features.npz')
            prefix = check_stream_prefix(old_stream,stream,old_extra,extra,len(prior_data['sources']))
            rows = np.flatnonzero(data['timestamp_ms'][stream.event_ids] >= start)
            gap = stream.elapsed_times[rows,0]; dose = extra['raw'][rows,0,1]
            primary = ((gap >= c['primary_min_gap_seconds']) & (gap < c['primary_max_gap_exclusive_seconds'])
                       & (dose >= c['primary_min_dose']) & (dose < c['primary_max_dose_exclusive']))
            users = data['sources'][stream.event_ids[rows]]
            support = dict(queries=int(primary.sum()),users=int(len(np.unique(users[primary]))))
            support['passed'] = support['queries'] >= c['support_min_queries'] and support['users'] >= c['support_min_users']
            folder = output/f'seed{seed}'; folder.mkdir()
            np.savez_compressed(folder/'candidate_stream.npz', **{name:getattr(stream,name) for name in stream.__dataclass_fields__})
            np.savez_compressed(folder/'artist_features.npz', **extra)
            np.savez_compressed(folder/'test_rows.npz', rows=rows, event_ids=stream.event_ids[rows], primary=primary, users=users)
            save(folder/'prefix_audit.json',prefix)
            anchors = anchor_audit(frame,stream,extra,settings,rows,primary,c)
            save(folder/'anchor_audit.json',anchors)
            item = dict(seed=seed, test_queries=len(rows), support=support,
                        prefix_audit='PASS', direct_anchors=len(anchors['anchors']),candidate_digest=stream.digest())
            per_seed.append(item);save(output/'partial_results.json',dict(window=report,per_seed=per_seed))
            print('FEBRUARY_CANDIDATES',json.dumps(item),flush=True)
        assert all(sha(ROOT/name)==h for name,h in pins.items())
        assert all(sha(ROOT/name)==h for name,h in specification['source_input_hashes'].items())
        assert all(sha(parent/name)==h for name,h in parent_manifest.items())
        for path,before in ((raw,raw_before),(archive,archive_before)):
            after = path.stat(); assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        save(output/'results.json',dict(status='PASS',protocol_id=p['protocol_id'],window=report,
            per_seed=per_seed,all_support_gates_passed=all(s['support']['passed'] for s in per_seed),
            source_input_guards_passed=True,model_fit=False,predictor_scored=False,test_scored=False,
            gpu_allocated=False,publication_eligible=False))
        print('FEBRUARY_PREPARATION_PASS',flush=True)
    except BaseException as exc:
        save(output/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        save(output/'artifact_manifest.json',{str(path.relative_to(output)):sha(path)
            for path in output.rglob('*') if path.is_file()})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('config','raw','archive','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();run(a.config,a.raw,a.archive,a.output)
