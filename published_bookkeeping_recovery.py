"""Filesystem/JSON recovery only. No experiment settings, labels or weights edited."""
from pathlib import Path
import json,shutil
import numpy as np,pandas as pd,joblib
from fog_pipeline import save_json as original_save,sha256


def python_values(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {python_values(k):python_values(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [python_values(x) for x in value]
    return value


def safe_save(path,value):return original_save(path,python_values(value))


def archive_folder(folder,out,reason):
    folder=Path(folder).resolve();out=Path(out).resolve();base=out/'technical_attempts';base.mkdir(exist_ok=True)
    stem='_'.join(folder.relative_to(out).parts)+'_'+reason;target=base/stem;n=2
    while target.exists():target=base/(stem+f'_{n}');n+=1
    assert folder.is_relative_to(out) and target.resolve().is_relative_to(out)
    folder.rename(target)
    return target


def archive_root_attempt(folder,out):
    folder=Path(folder);out=Path(out)
    if not (folder/'results.json').exists():return None
    frozen=json.loads((folder/'FROZEN_CASCADE.json').read_text())
    if not frozen.get('failures'):return None
    base=out/'technical_attempts';base.mkdir(exist_ok=True);archive=base/(folder.name+'_metadata_failure');n=2
    while archive.exists():archive=base/(folder.name+f'_metadata_failure_{n}');n+=1
    archive.mkdir()
    for p in list(folder.iterdir()):
        if not p.is_file():continue
        dest=archive/p.name
        if p.name=='results.json' or p.name.endswith('_FAILURE.json') or p.suffix=='.npz':
            assert p.resolve().is_relative_to(out.resolve()) and dest.resolve().is_relative_to(out.resolve());p.rename(dest)
        else:shutil.copy2(p,dest)
    return archive


def recovered_fit(original_fit,family,x,seq,candidate,meta,folder,out,records):
    folder=Path(folder)
    if not (folder/'FIT.json').exists() and (folder/'training_candidate_ledger.csv').exists():
        actual=None
        if (folder/'model.keras').exists():actual='tcn'
        elif (folder/'model.joblib').exists():
            model=joblib.load(folder/'model.joblib')
            actual='logistic' if hasattr(model,'named_steps') else 'hgb'
        if actual is not None:
            old=pd.read_csv(folder/'training_candidate_ledger.csv',float_precision='round_trip')
            y=meta.target.to_numpy().copy();y[meta.active_fog]=0
            mask=candidate&meta.known_followup.to_numpy()&((meta.target.to_numpy()>=0)|meta.active_fog.to_numpy())
            matches=(len(old)==len(meta) and np.array_equal(old.subject.to_numpy(),meta.subject.to_numpy())
                and np.array_equal(old.time.to_numpy(),meta.time.to_numpy())
                and np.array_equal(old.candidate.to_numpy(bool),candidate)
                and np.array_equal(old.known.to_numpy(bool),meta.known_followup.to_numpy())
                and np.array_equal(old.used.to_numpy(bool),mask) and np.array_equal(old.label.to_numpy(),y))
            if actual==family and matches:
                files={p.name:sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='FIT.json'}
                safe_save(folder/'FIT.json',dict(family=family,subjects=sorted(set(meta.subject.to_numpy()[mask])),
                    known_candidates=int(mask.sum()),positive_candidates=int(y[mask].sum()),reload_parity=True,
                    artifact_hashes=files,recovery='Model was saved and reload checked before NumPy-int metadata serialization failed; weights unchanged'))
                records.append(dict(folder=str(folder),action='reused_saved_fitted_weights',family=family,weight_hashes={k:v for k,v in files.items() if k.endswith(('.joblib','.keras'))}))
            else:
                archive=archive_folder(folder,out,'initial_fallback_family' if actual!=family else 'checkpoint_ledger_mismatch')
                records.append(dict(folder=str(folder),action='preserved_initial_invalid_attempt',archive=str(archive),actual_family=actual,required_family=family))
    return original_fit(family,x,seq,candidate,meta,folder)
