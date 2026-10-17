"""Retained-byte/metric/selection and full validation checkpoint inference audit.

Feature-history auditing is separately performed remotely on declared anchors.
This script does not replay optimizer steps or independently reconstruct every
raw history record. Current source must match the archived launch source.
"""
from pathlib import Path
import argparse
import json
import io
import hashlib
import sys
import zipfile

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from lifecycle_readout.benchmark import load_dataset, save_json, sha256, tensor_batch
from lifecycle_readout.causal_stream import CandidateStream
from lifecycle_readout.event_coupling_study import event_stream,CONTRASTS
from lifecycle_readout.event_coupling_study import EventCouplingControl


def audit(attempt, output):
    attempt, output = Path(attempt), Path(output)
    if output.exists():
        raise FileExistsError("Fresh audit output required")
    job = json.loads((attempt/"job.json").read_text())
    if job["status"] != "completed" or not job["cleanup_verified"]:
        raise ValueError("Attempt incomplete or owned runtime release unverified")
    if sha256(attempt/"source.zip") != job["source_bundle_sha256"]:
        raise ValueError("Source archive hash mismatch")
    with zipfile.ZipFile(attempt/"source.zip") as archive:
        manifest = json.loads(archive.read("source_manifest.json"))
        for rel, digest in manifest.items():
            if rel.startswith("src/") or rel in ("configs/event_coupling_v1.toml", "protocols/event_coupling_v1.toml"):
                if sha256(ROOT/rel) != digest:
                    raise ValueError("Current inference source differs from archive: "+rel)
    base = attempt/"artifacts"
    result = json.loads((base/"results.json").read_text())
    if result["test_scored"] or not result["complete_declared_seeds"] or result["smoke"]:
        raise ValueError("Not a complete validation-only real-data pilot")
    settings = result["settings"]
    if [entry["seed"] for entry in result["per_seed_runs"]] != settings["seeds"]:
        raise ValueError("Seed coverage mismatch")
    assert result['protocol_id']=='LCR-P-EVENT-COUPLING-001'
    with zipfile.ZipFile(attempt/'source.zip') as outer:
        assets={item['path']:outer.read(item['path']) for item in settings['input_files']}
        for item in settings['input_files']:
            assert hashlib.sha256(assets[item['path']]).hexdigest()==item['sha256']
        with zipfile.ZipFile(io.BytesIO(assets[settings['data_archive']])) as inner:
            raw=inner.read('resources/corpora/wikipedia_causal_v2.npz')
            assert hashlib.sha256(raw).hexdigest()==settings['prepared_data_sha256']
            data=dict(np.load(io.BytesIO(raw),allow_pickle=False))
    assert result['data']==dict(dataset='wikipedia',derived_sha256=settings['prepared_data_sha256'],data_archive=settings['data_archive'])
    torch.set_num_threads(1)
    records = []
    for entry in result["per_seed_runs"]:
        seed = entry["seed"]; folder = base/f"seed{seed}"
        candidates = np.load(folder/"candidate_identity.npz")
        arrays = dict(np.load(folder/"event_features.npz"))
        frozen=dict(np.load(io.BytesIO(assets[settings['feature_directory']+f'/seed{seed}_features.npz'])))
        for key,value in frozen.items():np.testing.assert_array_equal(value,arrays[key])
        parent=dict(np.load(io.BytesIO(assets[settings['candidate_directory']+f'/seed{seed}/candidate_identity.npz'])))
        for key,value in parent.items():np.testing.assert_array_equal(value,candidates[key])
        assert list(entry['models'])==settings['models']
        if arrays["raw"][candidates["split"] == 2].any() or arrays["queried"][candidates["split"] == 2].any():
            raise ValueError("Test exposure constructed")
        # Reconstruct only supported pre-event neighbor arrays from raw timestamps.
        # The parent independent history auditor remains responsible for sampler support.
        from lifecycle_readout.causal_stream import build_candidate_stream
        stream = build_candidate_stream(**data, seed=seed, split_seed=settings["split_seed"],
            max_neighbors=settings["max_neighbors"], history_decay=settings["history_decay"],
            negative_sampling="matched_historical", hard_time_unit=settings["hard_time_unit"])
        for key, value in (("event_ids",stream.event_ids),("negative_destinations",stream.negative_destinations),
                           ("split",stream.split),("pair_features",stream.pair_features),
                           ("excluded_training_ids",stream.excluded_training_ids)):
            np.testing.assert_array_equal(candidates[key], value)
        rows = np.flatnonzero(stream.split == 1)
        counts = []
        if set(entry["models"]) != set(settings["models"]):
            raise ValueError("Arm coverage mismatch")
        for arm in settings["models"]:
            arm_stream = event_stream(stream, arrays, arm)
            report = entry["models"][arm]
            path = folder/f"{arm}_seed{seed}.pt"
            predictions_path = folder/f"{arm}_seed{seed}_predictions.npz"
            if sha256(path) != report["checkpoint_sha256"] or sha256(predictions_path) != report["predictions_sha256"]:
                raise ValueError("Model artifact identity mismatch")
            saved = torch.load(path, weights_only=True, map_location="cpu")
            maximum = max(x["validation"]["ap"] for x in report["validation_history"])
            selected = next(x["epoch"] for x in report["validation_history"] if x["validation"]["ap"] == maximum)
            if selected != saved["selected_epoch"] or selected != report["selected_epoch"]:
                raise ValueError("Earliest validation maximum not selected")
            train_values = arm_stream.pair_features[stream.split == 0].reshape(-1, settings["pair_feature_dim"])
            np.testing.assert_allclose(saved["phi_mean"],train_values.mean(0),rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(saved["phi_std"],train_values.std(0).clip(min=1e-5),rtol=1e-6,atol=1e-7)
            clock=stream.elapsed_times[stream.split==0].ravel()
            exposure=np.expm1(arm_stream.pair_features[stream.split==0,:,7]).ravel()
            positive_exposure=exposure[exposure>0]
            unit=float(np.median(clock));exposure_unit=float(np.median(positive_exposure)) if len(positive_exposure) else 1.
            assert saved['time_unit']==unit==report['time_unit']
            assert saved['exposure_unit']==exposure_unit==report['exposure_unit']
            model = EventCouplingControl(arm, data["features"].shape[1], settings, saved["phi_mean"], saved["phi_std"],unit,exposure_unit).eval()
            model.load_state_dict(saved["model"])
            count=sum(p.numel() for p in model.parameters());counts.append(count)
            common=sum(p.numel() for module in (model.base,model.history_head) for p in module.parameters())
            branch=dict(history_only=0,dose_mlp=13,scalar_clock=10,phase_competition=13,shuffled_phase=13,activity_mlp=13)[arm]
            assert count==report['parameter_count'] and count-common==report['branch_parameter_count']==branch
            assert float(model.amplitude)==report['amplitude']
            if model.phase_kind:
                assert float(model.amplitude)>=0
                d=torch.nn.functional.softplus(model.raw_clock).detach().numpy()
                g=torch.nn.functional.softplus(model.raw_event).detach().numpy()
                if arm=='scalar_clock':g=g*d
                for name,value in dict(clock_rates=d,event_rates=g,readout_gaps=torch.nn.functional.softplus(model.raw_gap).detach().numpy()).items():
                    np.testing.assert_allclose(value,report['state_parameters'][name],rtol=1e-6,atol=1e-8)
            stored = np.load(predictions_path)
            np.testing.assert_array_equal(stored["event_ids"], stream.event_ids[rows])
            np.testing.assert_array_equal(stored["negative_destinations"], stream.negative_destinations[rows])
            positive, negative = [], []
            with torch.no_grad():
                for start in range(0,len(rows),settings["batch_size"]):
                    p,q = model(*tensor_batch(arm_stream, rows[start:start+settings["batch_size"]], data, torch.device("cpu")))
                    positive.extend(p["prob"].numpy());negative.extend(q["prob"].numpy())
            positive,negative = np.array(positive),np.array(negative)
            np.testing.assert_allclose(positive,stored["positive"],rtol=settings["audit_probability_rtol"],atol=settings["audit_probability_atol"])
            np.testing.assert_allclose(negative,stored["negative"],rtol=settings["audit_probability_rtol"],atol=settings["audit_probability_atol"])
            p,q = stored["positive"].astype(float),stored["negative"].astype(float)
            y = np.r_[np.ones(len(p)),np.zeros(len(q))]; scores = np.r_[p,q]
            recomputed=dict(ap=float(average_precision_score(y,scores)),auc=float(roc_auc_score(y,scores)),
                brier=float(np.mean(np.r_[(1-p)**2,q**2])), pair_accuracy=float(np.mean((p>q)+.5*(p==q))))
            for metric, value in recomputed.items():
                np.testing.assert_allclose(value,report["all_validation"][metric],rtol=1e-6,atol=1e-8)
            records.append(dict(seed=seed,arm=arm,queries=len(rows),metrics=recomputed,
                prediction_max_absolute_error=float(max(np.max(np.abs(positive-p)),np.max(np.abs(negative-q))))))
        assert entry['models']['dose_mlp']['parameter_count']==entry['models']['phase_competition']['parameter_count']
    assert set(result['paired_validation_ap'])==set(CONTRASTS)
    for contrast,record in result["paired_validation_ap"].items():
        values=[entry["models"][record["treatment"]]["all_validation"]["ap"]-entry["models"][record["control"]]["all_validation"]["ap"] for entry in result["per_seed_runs"]]
        np.testing.assert_array_equal(values,record["values"])
        np.testing.assert_allclose(np.mean(values),record["mean"],rtol=0,atol=1e-15)
        np.testing.assert_allclose(np.std(values,ddof=1),record["std_ddof1"],rtol=0,atol=1e-15)
    audit_report=dict(status="PASS",records=records,comparisons=result["paired_validation_ap"],
        scope="Archived-source identity, metric/selection/normalization, common candidate reproduction and full validation inference replay; remote feature history inherits the prior fixed input-audit anchors. No optimizer replay or fresh-environment reproduction.",
        runtime_release_verified=job["cleanup_verified"],test_scored=False,publication_eligible=False)
    save_json(output,audit_report)
    return audit_report


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("attempt",type=Path);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();report=audit(args.attempt,args.output)
    print("Event-coupling audit:",report["status"],"checkpoint records:",len(report["records"]))
