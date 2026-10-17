"""Synthetic local envelopes test auditing, not real Colab execution or efficacy."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import tarfile
import tomllib
import zipfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from lifecycle_readout import hard_controls
import audit_hard_controls
from audit_pair_cadence_results import audit_study


def test_actual_three_arm_artifacts_pass_full_cadence_audit(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    root=tmp_path/'project';root.mkdir()
    shutil.copytree(ROOT/'src',root/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (root/'configs').mkdir();(root/'protocols').mkdir()
    protocol_text=(ROOT/'protocols/pair_cadence_development_v1.toml').read_text().replace('inductive_unseen_node_ratio = 0.10','inductive_unseen_node_ratio = 0.0')
    (root/'protocols/pair_cadence_development_v1.toml').write_text(protocol_text)
    protocol=tomllib.loads(protocol_text)
    for arm in ['original','features','relative']:
        text=(ROOT/f'configs/pair_cadence_{arm}_v1.toml').read_text()
        for old,new in [('epochs = 10','epochs = 1'),('hidden_dim = 128','hidden_dim = 8'),
                        ('time_dim = 32','time_dim = 4'),('max_neighbors = 20','max_neighbors = 3'),
                        ('batch_size = 256','batch_size = 64')]:text=text.replace(old,new)
        (root/f'configs/pair_cadence_{arm}_v1.toml').write_text(text)
    n=600
    data=dict(sources=np.arange(n)%3,destinations=10+(np.arange(n)//3)%8,
        timestamps=np.arange(n,dtype=float)*11,
        features=np.random.RandomState(4).normal(size=(n,2)).astype(np.float32))
    prepared=root/'resources/corpora/wikipedia_causal_v2.npz';prepared.parent.mkdir(parents=True)
    np.savez_compressed(prepared,**data)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    identity=dict(dataset='synthetic-auditor-fixture',derived_sha256=digest(prepared))
    monkeypatch.setattr(hard_controls,'load_dataset',lambda *_:(data,identity))
    monkeypatch.setattr(audit_hard_controls,'ROOT',root)
    files=sorted((root/'src').rglob('*.py'))+sorted((root/'configs').glob('*.toml'))+sorted((root/'protocols').glob('*.toml'))
    manifest={str(p.relative_to(root)):digest(p) for p in files}
    attempts={}
    for arm in ['original','features','relative']:
        attempt=root/f'attempt-{arm}';attempt.mkdir();attempts[arm]=attempt
        config=root/f'configs/pair_cadence_{arm}_v1.toml'
        settings=tomllib.loads(config.read_text())
        hard_controls.run_controls('wikipedia',settings,protocol,data_dir=prepared.parent,
            output_dir=attempt/'artifacts',device='cpu',smoke=False)
        with zipfile.ZipFile(attempt/'source.zip','w') as archive:
            for p in files:archive.write(p,str(p.relative_to(root)))
            archive.writestr('source_manifest.json',json.dumps(manifest))
        with tarfile.open(attempt/'artifacts.tar.gz','w:gz') as archive:
            archive.add(attempt/'artifacts',arcname='artifacts')
        # Explicitly synthetic job records exercise the envelope checks only.
        job=dict(execution_kind='synthetic-local-auditor-fixture',status='completed',cleanup_verified=True,
            archive_sha256=digest(attempt/'artifacts.tar.gz'),source_bundle_sha256=digest(attempt/'source.zip'),
            protocol_sha256=digest(root/settings['protocol']),config_sha256=digest(config))
        (attempt/'job.json').write_text(json.dumps(job))
    result=audit_study(attempts)
    assert result['status']=='PASS' and not result['test_scored'] and not result['publication_eligible']
    assert result['primary_comparison']=='clock_residual_scalar'
    assert all(r['all_checks_pass'] for r in result['arm_audits'].values())
