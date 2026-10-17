"""Inference from retained January checkpoints; no optimizer or selection."""
from pathlib import Path
import hashlib
import platform
import time

import numpy as np
import torch

from .benchmark import save_json, sha256
from .hard_controls import score
from .lastfm_diagnostics import paired_metrics
from .lastfm_model import RecordedArtistControl, artist_arm_stream
from .lastfm_model_training import load_stream


def state_identity(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        a = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode()); digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode()); digest.update(a.tobytes())
    return digest.hexdigest()


def frozen_checkpoint_scores(checkpoint, kind, seed, stream, extra, data, rows, device):
    """Use only saved settings, normalizers, physical units and embedding vocabulary."""
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['kind'] != kind or saved['seed'] != seed:
        raise ValueError('Checkpoint identity mismatch')
    if saved['artist_vocabulary'] != int(data['training_artist_vocabulary'][0]):
        raise ValueError('Frozen training vocabulary mismatch')
    if np.max(data['features'][:,0]) > saved['artist_vocabulary']:
        raise ValueError('Unknown artist entered learned vocabulary')
    model = RecordedArtistControl(kind,saved['settings'],saved['phi_mean'],saved['phi_std'],
        saved['time_unit'],saved['exposure_unit'],artist_vocabulary=saved['artist_vocabulary'],seed=seed)
    model.load_state_dict(saved['model'],strict=True)
    model.to(device).requires_grad_(False).eval()
    before = state_identity(model)
    arm = artist_arm_stream(stream,extra,kind)
    positive,negative = score(model,rows,arm,data,saved['settings'],device)
    if before != state_identity(model) or any(p.grad is not None for p in model.parameters()):
        raise AssertionError('Frozen inference changed checkpoint state or gradients')
    if any(not np.isfinite(a).all() or (a<0).any() or (a>1).any() for a in (positive,negative)):
        raise ValueError('Invalid prediction vectors')
    metadata = dict(kind=kind,seed=seed,selected_epoch=saved['selected_epoch'],
        checkpoint_sha256=sha256(checkpoint),frozen_state_sha256=before,
        unchanged_state=True,model_fit=False,queries=len(rows),
        time_unit=saved['time_unit'],exposure_unit=saved['exposure_unit'],
        artist_vocabulary=saved['artist_vocabulary'],all_test=paired_metrics(positive,negative))
    return positive,negative,metadata


def score_holdout(root,settings,protocol,output,device):
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    if device != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Declared real inference requires owned CUDA runtime')
    root = Path(root)
    for item in settings['input_files']:
        path=(root/item['path']).resolve()
        assert path.is_relative_to(root) and sha256(path)==item['sha256']
    prepared=root/settings['prepared_directory']
    checkpoints=root/settings['checkpoint_directory']
    prep=__import__('json').loads((prepared/'results.json').read_text())
    assert prep['status']=='PASS' and prep['all_support_gates_passed']
    assert settings['models']==protocol['models'] and settings['seeds']==protocol['seeds']
    with np.load(prepared/'data.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    result=dict(protocol_id=protocol['protocol_id'],dataset='lastfm-february',model_fit=False,
        evaluation_split='test_february',test_scored=True,publication_eligible=False,per_seed_runs=[],
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                         device=device,gpu=torch.cuda.get_device_name(0)))
    for seed in settings['seeds']:
        stream=load_stream(prepared/f'seed{seed}/candidate_stream.npz')
        with np.load(prepared/f'seed{seed}/artist_features.npz',allow_pickle=False) as z:extra={k:z[k] for k in z.files}
        with np.load(prepared/f'seed{seed}/test_rows.npz',allow_pickle=False) as z:selection={k:z[k] for k in z.files}
        rows=selection['rows'];np.testing.assert_array_equal(selection['event_ids'],stream.event_ids[rows])
        # The extended cache's split=1 includes January; only explicit February rows are scored.
        assert (data['timestamp_ms'][stream.event_ids[rows]]>=settings['test_start_ms']).all()
        assert (data['timestamp_ms'][stream.event_ids[rows]]<settings['test_end_ms']).all()
        folder=output/f'seed{seed}';folder.mkdir()
        entry=dict(seed=seed,candidate_digest=stream.digest(),test_queries=len(rows),models={})
        for kind in settings['models']:
            started=time.monotonic();checkpoint=checkpoints/f'seed{seed}/{kind}_seed{seed}.pt'
            positive,negative,report=frozen_checkpoint_scores(checkpoint,kind,seed,stream,extra,data,rows,torch.device(device))
            predictions=folder/f'{kind}_predictions.npz'
            np.savez_compressed(predictions,positive=positive,negative=negative,
                event_ids=stream.event_ids[rows],negative_destinations=stream.negative_destinations[rows])
            report.update(predictions_sha256=sha256(predictions),elapsed_seconds=time.monotonic()-started)
            save_json(folder/f'{kind}.json',report);entry['models'][kind]=report
            save_json(output/'partial_results.json',dict(result,current_seed=entry))
            print('FROZEN_FEBRUARY_SCORED',seed,kind,'queries',len(rows),flush=True)
        result['per_seed_runs'].append(entry);save_json(output/'partial_results.json',result)
    for item in settings['input_files']:assert sha256(root/item['path'])==item['sha256']
    result['complete_declared_models_and_seeds']=True
    save_json(output/'results.json',result)
    return result
