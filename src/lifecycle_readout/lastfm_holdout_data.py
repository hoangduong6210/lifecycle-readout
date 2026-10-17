"""Chronological holdout extension with frozen identity and embedding semantics."""
from datetime import datetime

import numpy as np
import pandas as pd

from .lastfm_intake import parse_selected_line


def append_frozen_window(prior, identities, raw_events, start, end):
    """Append only [start,end); unknown artists stay outside the training lookup."""
    start_ms = int(datetime.fromisoformat(start).timestamp()) * 1000
    end_ms = int(datetime.fromisoformat(end).timestamp()) * 1000
    if start_ms >= end_ms or not len(prior['sources']):
        raise ValueError('Nonempty prior and increasing window required')
    if np.max(prior['timestamp_ms']) >= start_ms:
        raise ValueError('Prior must end strictly before appended window')
    vocabulary = int(prior['training_artist_vocabulary'][0])
    if identities['training_artist_vocabulary'] != vocabulary:
        raise ValueError('Training vocabulary mismatch')
    users = list(identities['sampled_users'])
    tracks = list(identities['track_uuids'])
    artists = list(identities['artist_uuids'])
    if not 0 <= vocabulary <= len(artists):
        raise ValueError('Invalid frozen training vocabulary')
    for values in (users, tracks, artists):
        if len(set(values)) != len(values):
            raise ValueError('Duplicate identities')
    chosen = set(users)
    rows = []
    lines = 0
    with raw_events.open(encoding='utf-8') as handle:
        for line in handle:
            lines += 1
            head = line.split('\t', 2)
            if head[0] not in chosen:
                continue
            if len(head) < 2:
                raise ValueError('Missing selected timestamp')
            # Do not parse feedback or identities outside this fixed window.
            if not start <= head[1] < end:
                continue
            row = parse_selected_line(line, chosen, end)
            if row is not None:
                rows.append(row)
    if not rows:
        raise ValueError('No events in fixed window; no replacement')
    tracks.extend(sorted({r[2] for r in rows if r[2]} - set(tracks)))
    artists.extend(sorted({r[3] for r in rows if r[3]} - set(artists)))
    user_map = {value: i for i, value in enumerate(users)}
    track_map = {value: i for i, value in enumerate(tracks)}
    artist_map = {value: i for i, value in enumerate(artists)}
    window = pd.DataFrame(dict(
        source=[user_map[r[0]] for r in rows],
        destination=[track_map.get(r[2], -1) for r in rows],
        timestamp_ms=[r[1] for r in rows],
        artist=[artist_map.get(r[3], -1) for r in rows],
    )).sort_values(['timestamp_ms', 'source', 'destination'], kind='stable')
    parent = pd.DataFrame(dict(source=prior['sources'], destination=prior['destinations'],
                               timestamp_ms=prior['timestamp_ms'], artist=prior['artist_ids']))
    frame = pd.concat([parent, window], ignore_index=True)
    if np.any(np.diff(frame.timestamp_ms.to_numpy()) < 0):
        raise ValueError('Nonchronological prior')
    artist = frame.artist.to_numpy()
    features = np.column_stack((np.where((artist >= 0) & (artist < vocabulary), artist + 1, 0),
                                frame.destination.to_numpy() >= 0, artist >= 0)).astype(np.float32)
    data = dict(sources=frame.source.to_numpy(), destinations=frame.destination.to_numpy(),
                artist_ids=artist, timestamp_ms=frame.timestamp_ms.to_numpy(),
                timestamps=frame.timestamp_ms.to_numpy() / 1000, features=features,
                training_artist_vocabulary=np.array([vocabulary]))
    for name, old in prior.items():
        observed = data[name] if name == 'training_artist_vocabulary' else data[name][:len(old)]
        np.testing.assert_array_equal(old, observed, err_msg=f'Prior data changed: {name}')
    result_maps = dict(sampled_users=users, track_uuids=tracks, artist_uuids=artists,
                       training_artist_vocabulary=vocabulary)
    report = dict(source_lines=lines, prior_rows=len(parent), appended_rows=len(window),
                  training_artist_vocabulary=vocabulary, appended_missing_track_rows=int((window.destination < 0).sum()),
                  appended_unknown_lookup_rows=int(((window.artist < 0) | (window.artist >= vocabulary)).sum()),
                  invalid_nonempty_track_rows=sum(r[4] for r in rows),
                  invalid_nonempty_artist_rows=sum(r[5] for r in rows),
                  all_prior_data_fields_equal=True)
    return frame, data, result_maps, report


def check_stream_prefix(prior, extended, prior_extra, extended_extra, prior_events):
    """Check every old candidate/history field; only the end boundary may change."""
    old_n = len(prior.event_ids)
    if not np.array_equal(np.flatnonzero(extended.event_ids < prior_events), np.arange(old_n)):
        raise AssertionError('Old candidate population changed')
    per_query = ('event_ids', 'negative_destinations', 'neighbor_edges', 'pair_features',
                 'elapsed_times', 'split', 'inductive')
    checked = []
    for name in per_query:
        np.testing.assert_array_equal(getattr(prior, name), getattr(extended, name)[:old_n], err_msg=name)
        checked.append(name)
    for name in ('excluded_training_ids', 'skipped_no_negative_ids'):
        current = getattr(extended, name)
        np.testing.assert_array_equal(getattr(prior, name), current[current < prior_events], err_msg=name)
        checked.append(name)
    np.testing.assert_array_equal(prior.unseen_nodes, extended.unseen_nodes)
    if prior.train_end != extended.train_end or prior.validation_end >= extended.validation_end:
        raise AssertionError('Unexpected chronological boundary change')
    for name in ('raw', 'alternate_exposure', 'reason'):
        n = prior_events if name == 'reason' else old_n
        np.testing.assert_array_equal(prior_extra[name], extended_extra[name][:n], err_msg=name)
        checked.append(name)
    return dict(status='PASS', prior_queries=old_n, prior_events=prior_events,
                checked_fields=checked + ['unseen_nodes', 'train_end'],
                only_allowed_scalar_change='validation_end')
