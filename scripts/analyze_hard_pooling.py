"""Posthoc pooling diagnostics; original HARD-001 endpoints remain unchanged."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('base', 'base_stats', 'scalar_decay', 'mlp_gate', 'ordered_gate')
ATTEMPTS = {'wikipedia': 'lcr-wikipedia-256094e9e7', 'reddit': 'lcr-reddit-bf7053e7a2'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparison_mass(positive, negative):
    """Count all positive-negative Cartesian wins with half credit for ties."""
    ordered = np.sort(negative)
    left = np.searchsorted(ordered, positive, side='left')
    right = np.searchsorted(ordered, positive, side='right')
    return float(np.sum(left + .5 * (right - left)))


def diagnose(positive, negative, bins):
    positive, negative = np.asarray(positive), np.asarray(negative)
    bins = np.asarray(bins)
    if positive.ndim != 1 or negative.shape != positive.shape or not len(positive):
        raise ValueError('Require nonempty paired one-dimensional scores')
    if bins.shape != (len(positive), 2):
        raise ValueError('Require one two-coordinate bin per query')
    scores = np.r_[positive, negative]
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Scores must be finite probabilities')
    n = len(positive)
    labels = np.r_[np.ones(n), np.zeros(n)]
    total_mass = comparison_mass(positive, negative)
    within_mass, within_pairs, weighted_ap = 0., 0, 0.
    per_bin = []
    for key in np.unique(bins, axis=0):
        mask = np.all(bins == key, axis=1)
        p, q = positive[mask], negative[mask]
        count = len(p)
        mass = comparison_mass(p, q)
        ap = float(average_precision_score(np.r_[np.ones(count), np.zeros(count)], np.r_[p, q]))
        within_mass += mass
        within_pairs += count * count
        weighted_ap += count / n * ap
        per_bin.append(dict(bin=key.tolist(), queries=count, positive=count, negative=count,
                            ap=ap, auc=mass / (count * count)))
    cross_pairs = n*n - within_pairs
    auc = total_mass / (n*n)
    np.testing.assert_allclose(auc, roc_auc_score(labels, scores), atol=1e-14)
    ties = positive == negative
    floor_p = positive == np.asarray(1e-6, dtype=positive.dtype)
    floor_q = negative == np.asarray(1e-6, dtype=negative.dtype)
    within_auc = within_mass/within_pairs
    cross_auc = (total_mass-within_mass)/cross_pairs if cross_pairs else None
    weight = within_pairs/(n*n)
    return dict(queries=n, pooled_ap=float(average_precision_score(labels, scores)),
                pooled_auc=auc, pooled_brier=float(np.mean((scores-labels)**2)),
                strict_pair_win=float(np.mean(positive > negative)),
                paired_ties=float(np.mean(ties)),
                half_credit_pair_accuracy=float(np.mean((positive > negative) + .5*ties)),
                query_weighted_bin_ap=weighted_ap,
                same_bin_cartesian_auc=within_auc,
                cross_bin_cartesian_auc=cross_auc,
                same_bin_comparison_weight=weight,
                auc_reconstruction=weight*within_auc+(1-weight)*(cross_auc if cross_auc is not None else 0),
                positive_mean=float(positive.mean()), negative_mean=float(negative.mean()),
                positive_exact_zero=float(np.mean(positive == 0)),
                negative_exact_zero=float(np.mean(negative == 0)),
                positive_at_original_clip=float(np.mean(floor_p)),
                negative_at_original_clip=float(np.mean(floor_q)),
                paired_ties_at_original_clip=float(np.mean(floor_p & floor_q)),
                per_bin=per_bin)


def stored_bins(features, elapsed, unit):
    """Do not pretend float32 retained arrays are a new raw-history audit."""
    count = np.rint(np.expm1(np.asarray(features[:, :, 2], dtype=np.float64)))
    if np.any(count < 1) or np.any(elapsed < 0):
        raise ValueError('Invalid recurrent history')
    cb = np.floor(np.log2(count)).astype(np.int64)
    tb = np.floor(np.log2(1 + np.asarray(elapsed, dtype=np.float64)/unit)).astype(np.int64)
    pair_bins = np.stack([cb, tb], axis=-1)
    np.testing.assert_array_equal(pair_bins[:, 0], pair_bins[:, 1])
    return pair_bins[:, 0]


def analyze(attempt):
    job = json.loads((attempt/'job.json').read_text())
    archive = attempt/'artifacts.tar.gz'
    assert digest(archive) == job['archive_sha256']
    assert job['cleanup_verified'] and job['status'] == 'completed'
    artifacts = attempt/'artifacts'
    manifest = artifacts/'checksums.sha256'
    # Bind the loose manifest to the original archive before trusting its entries.
    with tarfile.open(archive) as tar:
        members = [x for x in tar.getmembers() if x.name.removeprefix('./') == 'artifacts/checksums.sha256']
        assert len(members) == 1
        assert tar.extractfile(members[0]).read() == manifest.read_bytes()
    hashes = dict((name, sha) for sha, name in
                  (line.split(maxsplit=1) for line in manifest.read_text().splitlines() if line))
    def checked(path):
        assert digest(path) == hashes[path.relative_to(artifacts).as_posix()]
        return path
    result = json.loads(checked(artifacts/'results.json').read_text())
    assert result['protocol_id'] == 'LCR-P-HARD-001'
    runs = []
    identities = {}
    for seed in (1, 7, 42):
        reference = None
        for model in MODELS:
            path = checked(artifacts/f'seed{seed}/{model}_seed{seed}_predictions.npz')
            with np.load(path, allow_pickle=False) as data:
                identity = {k: data[k].copy() for k in
                            ('event_ids', 'negative_destinations', 'pair_features', 'elapsed_times')}
                if reference is not None:
                    for key in identity:
                        np.testing.assert_array_equal(identity[key], reference[key])
                reference = identity
                bins = stored_bins(identity['pair_features'], identity['elapsed_times'],
                                   result['settings']['hard_time_unit'])
                runs.append(dict(seed=seed, model=model,
                                 **diagnose(data['positive'], data['negative'], bins)))
            identities[str(path.relative_to(attempt))] = digest(path)
    aggregate = {}
    for model in MODELS:
        chosen = [r for r in runs if r['model'] == model]
        keys = [k for k, v in chosen[0].items() if isinstance(v, (int, float)) and k != 'seed']
        aggregate[model] = {key: dict(mean=float(np.mean([r[key] for r in chosen])),
                                     sample_sd=float(np.std([r[key] for r in chosen], ddof=1)))
                            for key in keys if all(r[key] is not None for r in chosen)}
    return dict(dataset=result['dataset'], original_archive_sha256=digest(archive),
                original_manifest_sha256=digest(manifest), prediction_sha256=identities,
                runs=runs, aggregate=aggregate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = dict(protocol_id='LCR-P-POOL-DIAG-001', posthoc=True,
                  publication_eligible=False, source_sha256=digest(Path(__file__)),
                  protocol_sha256=digest(ROOT/'protocols/pooling_diagnostic_v1.toml'),
                  limits=['Previously inspected test predictions; no new fitting or confirmation.',
                          'Bins reconstructed at retained floating-point precision; original raw-history audit is separate.',
                          'Descriptive decomposition does not identify the cause of model loss.'],
                  datasets={name: analyze(ROOT/'results/audit'/attempt) for name, attempt in ATTEMPTS.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    for name, data in report['datasets'].items():
        for model, row in data['aggregate'].items():
            print(name, model, 'AP', round(row['pooled_ap']['mean'], 6),
                  'paired', round(row['half_credit_pair_accuracy']['mean'], 6),
                  'bin_AP', round(row['query_weighted_bin_ap']['mean'], 6))


if __name__ == '__main__':
    main()
