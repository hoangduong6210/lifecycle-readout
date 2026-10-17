"""Strict-past pair cadence for a separately versioned development study.

The arithmetic mean interval is not an EMA, an event hazard, or a lifecycle
label. The normalized-recency idea has direct relative-freshness prior art:
Navarro et al., EPJ Data Science (2017), doi:10.1140/epjds/s13688-017-0127-3.
Test clocks are deliberately not constructed in validation development.
"""
from dataclasses import replace
import math
import numpy as np


def augment_pair_clock(data, stream, *, tau0_seconds):
    """Append log cadence, availability and normalized silence to six features.

    Mean interval = (last-first)/(count-1), requiring count>=2 and positive span.
    Missing/zero-span clocks use the median available TRAIN-candidate cadence,
    with a separate availability flag. The positive denominator offset is fixed
    by configuration. No candidate sampling, supervision support or split changes.
    """
    if not math.isfinite(tau0_seconds) or tau0_seconds <= 0:
        raise ValueError('tau0_seconds must be finite and positive')
    if stream.pair_features.shape[2] != 6:
        raise ValueError('Expected the unaugmented six-feature candidate stream')
    times, sources, destinations = (data[k] for k in ['timestamps', 'sources', 'destinations'])
    eligible = stream.split <= 1
    rows = {int(event): row for row, event in enumerate(stream.event_ids) if eligible[row]}
    excluded = set(stream.excluded_training_ids.tolist())
    # Diagnostics retain actual missing clocks rather than replacing them silently.
    mean = np.full((len(stream.event_ids), 2), np.nan, dtype=np.float64)
    count = np.zeros(mean.shape, dtype=np.int64)
    elapsed = np.zeros(mean.shape, dtype=np.float64)
    pairs = {}
    start = 0
    while start < len(times) and times[start] <= stream.validation_end:
        t = float(times[start])
        stop = int(np.searchsorted(times, t, side='right'))
        for event in range(start, stop):
            if event not in rows:
                continue
            row = rows[event]
            u = int(sources[event])
            for slot, v in enumerate([int(destinations[event]), int(stream.negative_destinations[row])]):
                first, last, c = pairs.get((u, v), (None, None, 0))
                count[row, slot] = c
                elapsed[row, slot] = 0. if not c else t - last
                if c >= 2 and last > first:
                    mean[row, slot] = (last - first) / (c - 1)
        for event in range(start, stop):
            if event in excluded:
                continue
            key = (int(sources[event]), int(destinations[event]))
            first, _, c = pairs.get(key, (t, t, 0))
            pairs[key] = (first, t, c + 1)
        start = stop
    np.testing.assert_allclose(elapsed[eligible], stream.elapsed_times[eligible], rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(np.log1p(count[eligible]), stream.pair_features[eligible, :, 2], rtol=2e-6, atol=1e-7)
    available = np.isfinite(mean) & (mean > 0)
    training = np.broadcast_to((stream.split == 0)[:, None], mean.shape)
    usable = mean[training & available]
    if not len(usable):
        raise ValueError('No estimable training-candidate cadence; no arbitrary fallback allowed')
    fallback = float(np.median(usable))
    effective = np.where(available, mean, fallback)
    clock = np.log1p(elapsed / (effective + tau0_seconds))
    extra = np.stack([np.log1p(effective / fallback), available.astype(float), clock], axis=-1)
    extra[~eligible] = 0.  # Test feature values are not queried or fitted.
    assert np.isfinite(extra).all() and (clock >= 0).all()
    augmented = replace(stream, pair_features=np.concatenate([stream.pair_features, extra.astype(np.float32)], axis=-1))
    metadata = dict(definition='strict-past-arithmetic-mean-interval',
        fallback_rule='median-positive-available-training-candidate-cadence',
        fallback_seconds=fallback, tau0_seconds=float(tau0_seconds),
        feature_names=['log1p_effective_cadence_over_train_median', 'cadence_available', 'log1p_silence_over_smoothed_cadence'],
        train_candidate_slots=int(training.sum()), train_available_slots=int((training & available).sum()),
        validation_candidate_slots=int(2 * (stream.split == 1).sum()),
        validation_available_slots=int(available[stream.split == 1].sum()),
        test_clocks_constructed=False, sampling_or_support_changed=False)
    diagnostics = dict(mean_interval_seconds=mean, prior_count=count, cadence_available=available,
                       clock=extra[:, :, 2], queried=eligible)
    return augmented, metadata, diagnostics
