"""Verify and index completed scientific artifacts; never trains or selects models."""
from pathlib import Path
import json,re,shutil,hashlib

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/published_cascade_20261111_01'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    pipeline=json.loads((OUT/'pipeline_status.json').read_text());assert pipeline['complete']
    plan=json.loads((OUT/'preregistered_plan.json').read_text())
    sources={n:digest(ROOT/n)==h and digest(OUT/'source'/n)==h for n,h in plan['source_sha256'].items()}
    assert all(sources.values()),sources
    assert digest(ROOT/plan['data'])==plan['data_sha256']
    original=ROOT/'runs/right_ankle_waveform_v2'
    original_manifest=json.loads((original/'manifest.json').read_text())
    original_checks={n:digest(original/n)==h for n,h in original_manifest['artifact_sha256'].items()}
    assert all(original_checks.values()),original_checks
    folds=json.loads((OUT/'all_folds.json').read_text())
    assert len(folds)==4 and all('results' in f for f in folds)
    active={str(f['fold']):json.loads((OUT/f"fold_{f['fold']}"/'FROZEN_CASCADE.json').read_text())['failures'] for f in folds}
    assert not any(active.values()),active
    fits=list(OUT.glob('fold_*/*/*_DONE.json'));assert len(fits)==84,len(fits)
    verifiers=list(OUT.glob('fold_*/verifiers/*/FIT.json'));ablations=list(OUT.glob('fold_*/ablations/*/FIT.json'))
    assert len(verifiers)==12 and len(ablations)==20,(len(verifiers),len(ablations))
    tests=(OUT/'final_tests.log').read_text(errors='replace')
    assert '\nOK' in tests and 'FAILED (' not in tests
    count=int(re.search(r'Ran (\d+) tests?',tests).group(1))
    native=json.loads((OUT/'native_verification.json').read_text());assert len(native['folds'])==4
    assert all(f['upstream_score_parity'] and f['selected_physical_parity'] and f['all_full_verifier_combined_decisions_exact'] for f in native['folds'])
    recovery=json.loads((OUT/'bookkeeping_recovery_ledger.json').read_text());recovered_checks={}
    old_locks=[(p.parent.name,json.loads(p.read_text())) for p in (OUT/'technical_attempts').rglob('FROZEN_CASCADE.json')]
    for row in recovery:
        if row.get('action')!='reused_saved_fitted_weights':continue
        folder=Path(row['folder']);relative=folder.relative_to(OUT)
        for name,expected in row['weight_hashes'].items():
            key=str(Path(*relative.parts[1:])/name)
            recorded=any(archive.startswith(relative.parts[0]+'_') and lock.get('model_hashes',{}).get(key)==expected for archive,lock in old_locks)
            recovered_checks[str(relative/name)]=recorded and digest(folder/name)==expected
    assert all(recovered_checks.values()),recovered_checks
    support=['published_cascade_stream.py','verify_published_cascade.py','report_published_cascade.py',
        'published_bookkeeping_recovery.py','complete_published_fold1.py','finish_published_pipeline.py',
        'test_published_bookkeeping.py','test_published_split_audit.py','finalize_published_study.py']
    snapshots=OUT/'implementation_support';snapshots.mkdir(exist_ok=True)
    for name in support:shutil.copy2(ROOT/name,snapshots/name)
    command_path=OUT/'commands.json';commands=json.loads(command_path.read_text())
    commands['integrity']='python finalize_published_study.py';write(command_path,commands)
    write(OUT/'final_integrity_audit.json',dict(frozen_source_hash_checks=sources,original_v2_artifact_hash_checks=original_checks,
        export_hash_verified=True,upstream_neural_fits=len(fits),full_verifier_fits=len(verifiers),ablation_fits=len(ablations),
        tests_passed=count,native_windows=sum(f['windows'] for f in native['folds']),
        recovered_weights_match_initial_frozen_artifacts=recovered_checks,
        support_source_hashes={n:digest(ROOT/n) for n in support},reserved_participants_excluded=True,locked_assessment_executed=False))
    summaries=json.loads((OUT/'aggregate_results.json').read_text());promotion=json.loads((OUT/'promotion_gate.json').read_text())
    assert not promotion['promoted'],'Physical delivered-lead validation was not authorized/performed'
    base=summaries['matched_v2'];final=summaries['baseline_OR_best_rescue']
    relative='runs/published_cascade_20261111_01/REPORT.md'
    line=(f"2026-11-11 published-model cascade completed: all four nested development folds, 84 upstream neural fits, "
        f"12 full verifier fits and 20 ablations. Matched v2 recipe {base['detected']}/{base['episodes']} early episodes; "
        f"selected final OR {final['detected']}/{final['episodes']}, {final['unsuccessful_starts']} unsuccessful starts "
        f"over {final['known_monitored_hours']:.4f} known monitored hours ({final['unsuccessful_per_known_hour']:.2f}/hour), "
        f"{final['unknown_starts']} unknown starts. No promotion. {count} tests passed; "
        f"{sum(f['windows'] for f in native['folds'])} native windows passed replay checks. "
        f"Original saved v2 weights unchanged; its historical cohort is reported separately. "
        f"Actual sensor-to-cue lead remains unmeasured. Reserved participants excluded; no locked assessment. "
        f"See [completed study]({relative}).")
    latest=ROOT/'LATEST_RESULTS.md';old=latest.read_text(encoding='utf-8')
    if not old.startswith(line):latest.write_text(line+'\n\nHistorical entries below describe earlier protocols; the continuous-session correction supersedes task-reset performance claims.\n\n'+old,encoding='utf-8')
    status_path=ROOT/'CURRENT_RESEARCH_STATUS.json';status=json.loads(status_path.read_text())
    operational=json.loads((OUT/'original_saved_v2_operational_comparison.json').read_text())
    if operational.get('original_v2_metrics'):status['current_metrics']=operational['original_v2_metrics']
    status.update(latest_study=relative,published_cascade=dict(matched_baseline=base,final=final,promotion=promotion),
        tests_passed=count,native_windows=sum(f['windows'] for f in native['folds']),promoted=False,
        actual_delivered_lead_verified=False,reserved_participants_excluded=True,locked_assessment_executed=False)
    write(status_path,status)
    registry={str(p.relative_to(OUT)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_registry.json'}
    write(OUT/'artifact_registry.json',dict(excludes_self=True,files=registry))
    print(line)


if __name__=='__main__':main()
