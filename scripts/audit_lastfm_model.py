"""All retained Last.fm validation checkpoints replayed on CPU, with saved-score metrics."""
from pathlib import Path
import hashlib,io,json,sys,zipfile
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifecycle_readout.lastfm_model import RecordedArtistControl,artist_arm_stream
from lifecycle_readout.lastfm_model_training import common_identity,load_stream
from lifecycle_readout.hard_controls import score
ROOT=Path(__file__).resolve().parents[1]


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def restore_initial_common(model,state,expected):
    keys={name for name in model.state_dict() if name.startswith(('base.','history_head.','artist_embedding.'))}
    assert set(state)==keys,'Only the complete common initialization may be restored'
    current=model.state_dict()
    assert all(state[name].shape==current[name].shape and state[name].dtype==current[name].dtype for name in keys)
    model.load_state_dict(state,strict=False)
    assert common_identity(model)==expected,'Independent initial-state digest mismatch'


def audit_models(folder,settings,data,streams,*,initial_common_states=None):
    result=json.loads((folder/'results.json').read_text());assert result['test_scored'] is False
    assert [x['seed'] for x in result['per_seed_runs']]==settings['seeds']
    records=[];byseed={};torch.set_num_threads(1)
    for entry in result['per_seed_runs']:
        seed=entry['seed'];stream,extra=streams[seed];assert stream.digest()==entry['candidate_digest'];assert set(entry['models'])==set(settings['models']);val=np.flatnonzero(stream.split==1);train=np.flatnonzero(stream.split==0);common=None;byseed[seed]={}
        for kind in settings['models']:
            report=entry['models'][kind];path=folder/f'seed{seed}/{kind}_seed{seed}.pt';pred=folder/f'seed{seed}/{kind}_seed{seed}_predictions.npz';assert sha(path)==report['checkpoint_sha256'] and sha(pred)==report['predictions_sha256'];saved=torch.load(path,map_location='cpu',weights_only=True);assert saved['settings']==settings
            arm=artist_arm_stream(stream,extra,kind);values=arm.pair_features[train].reshape(-1,settings['pair_feature_dim']);np.testing.assert_allclose(saved['phi_mean'],values.mean(0),rtol=2e-5,atol=2e-5);np.testing.assert_allclose(saved['phi_std'],values.std(0).clip(min=1e-5),rtol=2e-5,atol=2e-5)
            times=arm.elapsed_times[train].ravel();exposure=np.expm1(arm.pair_features[train,:,7]).ravel();positive=exposure[exposure>0];assert saved['time_unit']==float(np.median(times));assert saved['exposure_unit']==(float(np.median(positive)) if len(positive) else 1.)
            torch.manual_seed(seed);model=RecordedArtistControl(kind,settings,saved['phi_mean'],saved['phi_std'],saved['time_unit'],saved['exposure_unit'],artist_vocabulary=int(data['training_artist_vocabulary'][0]),seed=seed);local_identity=common_identity(model)
            if initial_common_states is not None:restore_initial_common(model,initial_common_states[seed],saved['initial_common_sha256'])
            identity=common_identity(model);assert identity==saved['initial_common_sha256']==report['initial_common_sha256']
            if common is None:common=identity
            assert identity==common;model.load_state_dict(saved['model']);model.eval();replayed=score(model,val,arm,data,settings,torch.device('cpu'))
            with np.load(pred,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['event_ids'],stream.event_ids[val]);np.testing.assert_array_equal(z['negative_destinations'],stream.negative_destinations[val]);pos=z['positive'];neg=z['negative']
            max_error=0.
            for observed,expected in zip(replayed,[pos,neg]):
                np.testing.assert_allclose(observed,expected,rtol=2e-4,atol=2e-5);max_error=max(max_error,float(np.max(np.abs(observed-expected))))
            labels=np.r_[np.ones(len(pos)),np.zeros(len(neg))];scores=np.r_[pos,neg];metrics=dict(ap=float(average_precision_score(labels,scores)),auc=float(roc_auc_score(labels,scores)),brier=float(np.mean(np.r_[(1.-pos)**2,neg**2])),pair_accuracy=float(np.mean((pos>neg)+.5*(pos==neg))))
            for name,value in metrics.items():assert abs(value-report['all_validation'][name])<2e-7
            history=report['validation_history'];assert len(history)==settings['epochs'];assert [x['epoch'] for x in history]==list(range(1,settings['epochs']+1));best=max(x['validation']['ap'] for x in history);chosen=next(x['epoch'] for x in history if x['validation']['ap']==best);assert chosen==saved['selected_epoch']==report['selected_epoch'];assert abs(metrics['ap']-best)<1e-12
            parameters=sum(p.numel() for p in model.parameters());branch=sum(p.numel() for n,p in model.named_parameters() if not n.startswith(('base.','history_head.','artist_embedding.')));assert parameters==report['parameter_count'] and branch==report['branch_parameter_count'];assert branch==({'history_only':0,'scalar_clock':10}.get(kind,13))
            if model.phase_kind:assert float(model.amplitude.detach())>=0
            records.append(dict(seed=seed,kind=kind,status='PASS',validation_rows=len(val),selected_epoch=chosen,max_probability_abs_error=max_error,parameter_count=parameters,branch_parameter_count=branch,metrics=metrics,local_initial_common_sha256=local_identity,verified_initial_common_sha256=identity,initialization_reconstruction=('local' if initial_common_states is None else 'independent_linux_cpu')));byseed[seed][kind]=metrics
        a,b=(entry['models'][k] for k in ['phase_competition','dose_mlp']);assert a['parameter_count']==b['parameter_count']
    comparisons={}
    for name,contrast in result['paired_validation_ap'].items():
        values=[byseed[seed][contrast['treatment']]['ap']-byseed[seed][contrast['control']]['ap'] for seed in settings['seeds']];np.testing.assert_allclose(values,contrast['values'],rtol=0,atol=1e-12);assert abs(float(np.mean(values))-contrast['mean'])<1e-12;comparisons[name]=dict(values=values,mean=float(np.mean(values)))
    return dict(status='PASS',records=records,comparisons=comparisons,scope='Complete retained validation checkpoint inference replay and saved-score metrics, units, identities/capacity and declared selection history; no optimizer replay or independent raw-history reconstruction')


def audit_attempt(attempt,output,*,initialization_attempt=None):
    folder=attempt/'artifacts';result=json.loads((folder/'results.json').read_text());settings=result['settings']
    with zipfile.ZipFile(attempt/'source.zip') as archive:
        manifest=json.loads(archive.read('source_manifest.json'))
        # Check local replay implementation against the actual uploaded source.
        for name,h in manifest.items():
            if name.startswith('src/') and name.endswith('.py'):assert sha(ROOT/name)==h
        def raw(name):
            value=archive.read(name);assert hashlib.sha256(value).hexdigest()==manifest[name];return value
        prepared=settings['prepared_directory'];data=dict(np.load(io.BytesIO(raw(prepared+'/data.npz')),allow_pickle=False));streams={}
        for seed in settings['seeds']:
            stream=load_stream(io.BytesIO(raw(f'{prepared}/seed{seed}/candidate_stream.npz')));extra=dict(np.load(io.BytesIO(raw(f'{prepared}/seed{seed}/artist_features.npz')),allow_pickle=False));streams[seed]=(stream,extra)
    initial_states=None;initial_report=None
    if initialization_attempt is not None:
        initializer=Path(initialization_attempt);job=json.loads((initializer/'job.json').read_text());assert job['status']=='completed' and job['cleanup_verified'] and job['remote_outcome_verified'] and job['artifacts_verified'] and job['requested_accelerator']=='CPU' and job['gpu_requested'] is False
        artifacts=initializer/'artifacts';identity=json.loads((artifacts/'source_identity.json').read_text());assert identity['bundle_sha256']==sha(initializer/'source.zip') and identity['worker_sha256']==sha(initializer/'worker.py')
        for name,h in json.loads((artifacts/'artifact_manifest.json').read_text()).items():
            target=(artifacts/name).resolve();assert target.is_relative_to(artifacts.resolve()) and sha(target)==h
        initial_report=json.loads((artifacts/'results.json').read_text());assert initial_report['status']=='PASS' and initial_report['model_fit'] is False and initial_report['test_scored'] is False
        expected={(entry['seed'],kind):report['initial_common_sha256'] for entry in result['per_seed_runs'] for kind,report in entry['models'].items()}
        observed={(x['seed'],x['kind']):x['initial_common_sha256'] for x in initial_report['records']};assert observed==expected and len(initial_report['records'])==len(expected)
        with zipfile.ZipFile(initializer/'source.zip') as z:
            pins=json.loads(z.read('manifest.json'))
            for name,h in manifest.items():
                if name.startswith('src/') and name.endswith('.py'):assert pins[name]==h==hashlib.sha256(z.read(name)).hexdigest()
        initial_states={seed:torch.load(artifacts/f'initial_common_seed{seed}.pt',map_location='cpu',weights_only=True) for seed in settings['seeds']}
    report=audit_models(folder,settings,data,streams,initial_common_states=initial_states)
    report['initialization_replay']=initial_report
    report['local_environment']=dict(torch=torch.__version__,numpy=np.__version__)
    output.write_text(json.dumps(report,indent=2)+'\n');return report
