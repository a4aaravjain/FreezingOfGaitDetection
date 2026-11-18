"""Complete all frozen folds after directory/metadata failures; no new trial."""
import json
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from threadpoolctl import threadpool_limits
from run_published_cascade import OUT,load_export,freeze,evaluate_fold
from fog_pipeline import save_json
import run_published_cascade as runner
import published_verifiers as verifiers
from published_bookkeeping_recovery import archive_root_attempt,recovered_fit,safe_save


def main():
    assert (OUT/'execution_completed.json').exists(),'Wait for the ongoing main execution to finish before this completion'
    frame=load_export();plan=freeze(frame)
    for spec in plan['folds']:
        folder=OUT/f"fold_{spec['fold']}"
        for name in ['verifiers','ablations']:(folder/name).mkdir(parents=True,exist_ok=True)
    # Restore cached decimal scores to the original binary floats exactly.
    # This is checkpoint fidelity, not recalibration or an experimental change.
    original_reader=pd.read_csv
    def checkpoint_reader(path,*args,**kwargs):
        if isinstance(path,(str,Path)) and Path(path).name.endswith('_scores.csv'):
            kwargs.setdefault('float_precision','round_trip')
        return original_reader(path,*args,**kwargs)
    recovery=[];original_fit=runner.fit_verifier
    def fit_with_recovery(family,x,seq,candidate,meta,folder):
        return recovered_fit(original_fit,family,x,seq,candidate,meta,folder,OUT,recovery)
    with threadpool_limits(limits=2),patch.object(pd,'read_csv',side_effect=checkpoint_reader),patch.object(verifiers,'save_json',side_effect=safe_save),patch.object(runner,'fit_verifier',side_effect=fit_with_recovery):
        for spec in plan['folds']:
            archive=archive_root_attempt(OUT/f"fold_{spec['fold']}",OUT)
            if archive:recovery.append(dict(fold=spec['fold'],initial_attempt=str(archive)))
            evaluate_fold(frame,spec)
            safe_save(OUT/'bookkeeping_recovery_ledger.json',recovery)
    actual=[]
    for spec in plan['folds']:
        path=OUT/f"fold_{spec['fold']}"/'results.json'
        if path.exists():actual.append(dict(fold=spec['fold'],outer=spec['outer'],results=json.loads(path.read_text())))
        else:actual.append(dict(fold=spec['fold'],outer=spec['outer'],failure='See retained fold failure record'))
    save_json(OUT/'all_folds.json',actual)
    save_json(OUT/'execution_completed.json',dict(attempted=4,completed=sum('results' in f for f in actual),
        exact_setting_bookkeeping_failures_completed=True,upstream_retraining=False,new_experiments_added=False,
        reserved_participants_excluded=True,locked_assessment_executed=False,promoted=False))


if __name__=='__main__':main()
