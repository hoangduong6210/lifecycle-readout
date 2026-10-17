from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifecycle_readout.official_data import audit_stream, inspect_npy_header


def test_official_masks_and_strict_past_recurrence_survive_chunk_boundaries(tmp_path):
    path = tmp_path / 'events.csv'
    path.write_text('src,dst,time,split\n1,8,1,0\n1,8,1,0\n2,9,2,0\n1,8,3,1\n3,8,3,1\n1,8,4,2\n')
    result = audit_stream(path, chunk_size=1, prepared_output=tmp_path / 'prepared.npz')
    assert result['official_split_counts'] == {'0': 3, '1': 2, '2': 1}
    assert result['development_pair_recurrence']['0']['strictly_past_pair_events'] == 0
    assert result['development_pair_recurrence']['1']['strictly_past_pair_events'] == 1
    with np.load(tmp_path / 'prepared.npz') as data:
        np.testing.assert_array_equal(data['event_ids'], np.arange(5))
        assert (data['split'] < 2).all()
    assert not result['test_recurrence_computed'] and not result['test_model_scored']


@pytest.mark.parametrize('rows', ['1,8,2,0\n2,9,1,1\n3,8,3,2\n',
                                  '1,8,1,0\n2,9,2,2\n3,8,3,1\n',
                                  '-1,8,1,0\n2,9,2,1\n3,8,3,2\n'])
def test_invalid_stream_is_rejected(tmp_path, rows):
    path = tmp_path / 'events.csv'; path.write_text('src,dst,time,split\n' + rows)
    with pytest.raises(ValueError): audit_stream(path, chunk_size=1)


def test_negative_header_is_checked_without_loading_values(tmp_path, monkeypatch):
    path = tmp_path / 'negatives.npy'; np.save(path, np.zeros((3, 100), dtype=np.int64))
    monkeypatch.setattr(np, 'load', lambda *_args, **_kwargs: pytest.fail('Candidate values must not be loaded'))
    result = inspect_npy_header(path)
    assert result['shape'] == [3, 100] and not result['array_values_loaded']
    with path.open('ab') as stream: stream.write(b'extra')
    with pytest.raises(ValueError, match='size'): inspect_npy_header(path)
