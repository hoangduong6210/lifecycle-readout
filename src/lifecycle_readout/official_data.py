"""Structural audit of supplied TGB-Seq splits; never replace official masks."""
from pathlib import Path
import csv

import numpy as np
import pandas as pd


def inspect_npy_header(path):
    """Read shape/dtype only; do not load or inspect candidate values."""
    with Path(path).open('rb') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError('Unsupported NPY header version')
        header_size = stream.tell()
    if dtype.hasobject or len(shape) != 2 or not np.issubdtype(dtype, np.integer):
        raise ValueError('Expected a plain two-dimensional integer candidate array')
    expected_size = header_size + int(np.prod(shape)) * dtype.itemsize
    if Path(path).stat().st_size != expected_size:
        raise ValueError('Candidate file size disagrees with header')
    return {'shape': list(shape), 'dtype': str(dtype), 'fortran_order': fortran,
            'header_bytes': header_size, 'array_values_loaded': False}


def audit_stream(path, *, chunk_size=200000, prepared_output=None):
    """Audit full CSV structure, with recurrence calculated on train/val only.

    Test CSV rows are used only for schema, range, chronology and split counts.
    No test recurrence or model metric is calculated. Ordered source/destination
    roles are retained even if their integer ID ranges overlap.
    """
    with Path(path).open(newline='') as stream:
        header = next(csv.reader(stream))
    if header != ['src', 'dst', 'time', 'split']:
        raise ValueError('Unexpected official CSV columns or column order')
    columns = {k: [] for k in header}
    ids = []
    counts = {str(k): 0 for k in (0, 1, 2)}
    previous_time, previous_split, cursor = -np.inf, 0, 0
    source_min, source_max, dest_min, dest_max = np.inf, -np.inf, np.inf, -np.inf
    split_ranges = {str(k): {'min_time': None, 'max_time': None} for k in (0, 1, 2)}
    for chunk in pd.read_csv(path, dtype={'src': np.int64, 'dst': np.int64,
                                         'time': np.float64, 'split': np.int64}, chunksize=chunk_size):
        times = chunk['time'].to_numpy(); phases = chunk['split'].to_numpy()
        if not np.isfinite(times).all() or (np.diff(times) < 0).any() or times[0] < previous_time:
            raise ValueError('Nonfinite or nonchronological official events')
        if not np.isin(phases, [0, 1, 2]).all() or (np.diff(phases) < 0).any() or phases[0] < previous_split:
            raise ValueError('Unexpected or nonchronological official split labels')
        previous_time, previous_split = times[-1], phases[-1]
        for key in ('src', 'dst'):
            values = chunk[key].to_numpy()
            if (values < 0).any() or (values > np.iinfo(np.uint32).max).any():
                raise ValueError('Node ID outside supported nonnegative 32-bit range')
        source_min = min(source_min, chunk['src'].min()); source_max = max(source_max, chunk['src'].max())
        dest_min = min(dest_min, chunk['dst'].min()); dest_max = max(dest_max, chunk['dst'].max())
        for phase in (0, 1, 2):
            selected = phases == phase; count = int(selected.sum()); counts[str(phase)] += count
            if count:
                first, last = float(times[selected][0]), float(times[selected][-1])
                if split_ranges[str(phase)]['min_time'] is None:
                    split_ranges[str(phase)]['min_time'] = first
                split_ranges[str(phase)]['max_time'] = last
        selected = phases < 2
        for key in header:
            columns[key].append(chunk[key].to_numpy()[selected])
        ids.append(np.arange(cursor, cursor + len(chunk), dtype=np.int64)[selected])
        cursor += len(chunk)
    if not all(counts.values()):
        raise ValueError('Missing declared train, validation or test split')
    arrays = {k: np.concatenate(v) for k, v in columns.items()}
    arrays['event_ids'] = np.concatenate(ids)
    keys = (arrays['src'].astype(np.uint64) << np.uint64(32)) | arrays['dst'].astype(np.uint64)
    order = np.argsort(keys, kind='stable')
    sorted_keys, sorted_times = keys[order], arrays['time'][order]
    starts = np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
    group = np.cumsum(starts) - 1
    # The stream is chronological, including tied events. Repeats at the first
    # timestamp have no strictly past pair history and must remain unsupported.
    first_times = sorted_times[starts][group]
    has_past = sorted_times > first_times
    phases = arrays['split'][order]
    recurrence = {}
    for phase in (0, 1):
        selected = phases == phase
        recurrence[str(phase)] = {'events': int(selected.sum()),
            'strictly_past_pair_events': int(has_past[selected].sum()),
            'no_strictly_past_pair_events': int((~has_past[selected]).sum())}
    if prepared_output is not None:
        prepared_output = Path(prepared_output)
        if prepared_output.exists():
            raise FileExistsError('Use a fresh prepared-data path')
        prepared_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prepared_output, **arrays)
    return {'csv_columns': header, 'total_events': cursor, 'official_split_counts': counts,
            'official_split_time_ranges': split_ranges, 'timestamps_nondecreasing': True,
            'splits_nondecreasing': True, 'source_id_range': [int(source_min), int(source_max)],
            'destination_id_range': [int(dest_min), int(dest_max)],
            'development_pair_recurrence': recurrence, 'development_unique_pairs': int(starts.sum()),
            'test_recurrence_computed': False, 'test_model_scored': False,
            'prepared_scope': 'supplied train and validation rows only; original zero-based event IDs preserved'}
