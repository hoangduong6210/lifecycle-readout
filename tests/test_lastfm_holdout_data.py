from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lifecycle_readout.lastfm_holdout_data import append_frozen_window, check_stream_prefix
from lifecycle_readout.lastfm_model_stream import build_lastfm_stream


def test_append_keeps_january_ids_and_training_unknown_semantics(tmp_path):
    old_track = 'bbbbbbbb-1234-1234-1234-123456789abc'
    new_track = 'aaaaaaaa-1234-1234-1234-123456789abc'
    trained = 'bbbbbbbb-1234-1234-1234-123456789def'
    january = 'cccccccc-1234-1234-1234-123456789def'
    february = 'aaaaaaaa-1234-1234-1234-123456789def'
    identities = dict(sampled_users=['user_000001'], track_uuids=[old_track],
                      artist_uuids=[trained, january], training_artist_vocabulary=1)
    prior = dict(sources=np.array([0, 0]), destinations=np.array([0, 0]),
                 artist_ids=np.array([0, 1]), timestamp_ms=np.array([1230768000000, 1230854400000]),
                 timestamps=np.array([1230768000., 1230854400.]),
                 features=np.array([[1, 1, 1], [0, 1, 1]], np.float32),
                 training_artist_vocabulary=np.array([1]))
    raw = tmp_path / 'events.tsv'
    raw.write_text(''.join([
        f'user_000001\t2009-02-01T00:00:00Z\t{january}\tname\t{old_track}\tname\n',
        f'user_000001\t2009-02-02T00:00:00Z\t{february}\tname\t{new_track}\tname\n',
        f'user_000001\t2009-02-02T00:00:00Z\t{trained}\tname\t\tname\n',
        'user_000001\t2009-03-01T00:00:00Z\tmalformed_future_feedback\n',
        'user_000001\t2009-01-01T00:00:00Z\tmalformed_old_feedback\n',
    ]))
    frame, data, maps, report = append_frozen_window(prior, identities, raw,
        '2009-02-01T00:00:00Z', '2009-03-01T00:00:00Z')
    assert maps['track_uuids'] == [old_track, new_track]
    assert maps['artist_uuids'] == [trained, january, february]
    assert identities['artist_uuids'] == [trained, january]  # no input mutation
    assert data['training_artist_vocabulary'].tolist() == [1]
    assert data['features'][:, 0].tolist() == [1, 0, 0, 1, 0]
    assert data['destinations'].tolist() == [0, 0, 0, -1, 1]
    assert report['appended_rows'] == 3 and report['appended_missing_track_rows'] == 1
    assert int(frame.timestamp_ms.max()) < 1235865600000
    with pytest.raises(ValueError, match='vocabulary mismatch'):
        append_frozen_window(prior, {**identities, 'training_artist_vocabulary': 2}, raw,
                             '2009-02-01T00:00:00Z', '2009-03-01T00:00:00Z')


def test_extended_candidate_rng_prefix_and_mutation_rejection():
    spec = importlib.util.spec_from_file_location('stream_fixture', ROOT / 'tests/test_lastfm_model_stream.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = module.fixture()
    old, x = module.build(frame)
    import pandas as pd
    future = pd.DataFrame([(0, 1, 32*3600000, 5), (1, -1, 33*3600000, 8)], columns=frame.columns)
    extended = pd.concat([frame, future], ignore_index=True)
    new, y = build_lastfm_stream(extended, module.settings(), seed=101,
                                train_end_ms=10*3600000, validation_end_ms=40*3600000)
    assert check_stream_prefix(old, new, x, y, len(frame))['status'] == 'PASS'
    new.pair_features[0, 0, 0] += 1
    with pytest.raises(AssertionError):
        check_stream_prefix(old, new, x, y, len(frame))
