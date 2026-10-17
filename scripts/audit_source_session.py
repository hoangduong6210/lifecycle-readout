"""Independently replay source timestamp memory and session diagnostic metrics."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def chronological_reference(sources, timestamps):
    memory = {}
    result = np.full(len(sources), np.nan)
    for row, (source, timestamp) in enumerate(zip(sources, timestamps)):
        source, timestamp = int(source), float(timestamp)
        if source not in memory:
            memory[source] = (timestamp, None)
            continue
        latest, previous = memory[source]
        if timestamp > latest:
            result[row] = timestamp-latest
            memory[source] = (timestamp, latest)
        else:
            assert timestamp == latest
            if previous is not None:
                result[row] = timestamp-previous
    return result


def audit(folder):
    report = json.loads((folder/'analysis/diagnostic.json').read_text())
    plan = json.loads((folder/'plan.json').read_text())
    for name, expected in plan['source_sha256'].items():
        assert digest(folder/Path(name).name) == expected, name
        # Additional verification tests may be added after analysis; the exact
        # pre-analysis tests remain archived. Executed code/protocol cannot drift.
        if not name.startswith('tests/'):
            assert digest(ROOT/name) == expected, name
    for name, expected in report['verified_inputs'].items():
        assert digest(ROOT/name) == expected, name
    prepared = [name for name in report['verified_inputs'] if name.startswith('resources/corpora/')]
    assert len(prepared) == 1
    with np.load(ROOT/prepared[0]) as saved:
        source, time, split, event_ids = (saved[key] for key in ['src','time','split','event_ids'])
    assert set(np.unique(split)) == {0,1} and np.all(np.diff(time) >= 0)
    reference = chronological_reference(source,time)
    spec = importlib.util.spec_from_file_location('pinned_session_analyzer', ROOT/'scripts/analyze_source_session.py')
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    np.testing.assert_equal(reference, analyzer.source_gaps(source,time))
    train = reference[(split==0)&np.isfinite(reference)]
    cuts = np.quantile(train,[.25,.5,.75],method='linear')
    np.testing.assert_array_equal(cuts, report['training_cutoffs_seconds'])
    assert len(train) == report['training_gap_count']
    val = split == 1
    gaps = reference[val]
    with np.load(folder/'analysis/query_diagnostic.npz') as saved:
        np.testing.assert_equal(gaps,saved['source_gap_seconds'])
        np.testing.assert_array_equal(event_ids[val],saved['event_ids'])
        masks = [~np.isfinite(gaps)]
        bounds = np.r_[-np.inf,cuts,np.inf]
        masks.extend(np.isfinite(gaps)&(gaps>=lo)&(gaps<hi) for lo,hi in zip(bounds[:-1],bounds[1:]))
        counts = []
        for label,(row,mask) in enumerate(zip(report['strata'],masks)):
            np.testing.assert_array_equal(saved['interval']==label-1,mask)
            count = int(mask.sum()); counts.append(count)
            assert row['count'] == count and row['fraction'] == count/len(gaps)
            if count and label:
                assert row['mean_gap_seconds'] == float(gaps[mask].mean())
                assert row['median_gap_seconds'] == float(np.median(gaps[mask]))
            for model in ['edge_memory','popularity','pooled_transition']:
                if count:
                    assert row['mrr'][model] == float(saved[model][mask].mean())
                else:
                    assert row['mrr'][model] is None
            if count:
                assert row['transition_minus_popularity'] == row['mrr']['pooled_transition']-row['mrr']['popularity']
        assert sum(counts) == report['query_count'] == int(val.sum())
        attempt=ROOT/'results/audit/lcr-ML-20M-ba89635570/artifacts'
        for model in ['edge_memory','popularity','pooled_transition']:
            with np.load(attempt/f'{model}_predictions.npz') as original:
                np.testing.assert_array_equal(saved[model],original['reciprocal_rank'])
            assert report['overall_mrr'][model] == float(saved[model].mean())
    return dict(status='PASS',source_rows_replayed=len(source),validation_queries_checked=len(gaps),
                all_source_gap_values_replayed=True,training_cutoffs_reproduced=True,
                full_training_and_validation_gap_array_equal=True,
                all_saved_rank_vectors_equal_original=True,all_stratum_metrics_recomputed=True,
                diagnostic_sha256=digest(folder/'analysis/diagnostic.json'),
                query_artifact_sha256=digest(folder/'analysis/query_diagnostic.npz'),
                auditor_sha256=digest(Path(__file__)),test_read=False,fitting=False,
                current_test_sha256=digest(ROOT/'tests/test_source_session.py'),
                scope='Independent full clock/metric replay; relies on original count-screen score audit, not new full score inference or causal identification.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be fresh')
    result=audit(args.folder)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
