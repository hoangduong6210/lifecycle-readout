"""Independent auditor's sparse replay must not observe simultaneous events."""
from pathlib import Path
import runpy

import numpy as np


def test_sparse_audit_replay_handles_source_timestamp_ties():
    reference = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                  'scripts/audit_stream_screen.py'))['reference_scores']
    data = dict(src=np.array([1, 1, 2, 1, 2, 3, 1, 1, 2, 3]),
                dst=np.array([5, 6, 6, 7, 5, 7, 5, 6, 7, 5]),
                time=np.array([1., 1., 1., 2., 2., 2., 3., 3., 3., 4.]),
                split=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]))
    candidates = dict(sources=data['src'][6:], timestamps=data['time'][6:],
                      positive_destinations=data['dst'][6:],
                      negative_destinations=np.array([[6, 7], [5, 7], [5, 6], [6, 7]]))
    actual = reference(data, candidates, [0, 1, 2, 3])
    for row in range(4):
        np.testing.assert_allclose(actual[row]['popularity'], [1/3] * 3)
    np.testing.assert_array_equal(actual[2]['edge_memory'], [0, 1, 1])
    # At t=3 both source-1 events follow context 7, even though they are tied.
    # They become available to source 3 only at t=4.
    np.testing.assert_allclose(actual[0]['pooled_transition'], [1/3] * 3)
    np.testing.assert_allclose(actual[3]['pooled_transition'], [4/3, 4/3, 1/3])
