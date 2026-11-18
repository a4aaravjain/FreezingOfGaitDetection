"""Sequential continuation of the frozen study; no scheduling service or new trials."""
from pathlib import Path
import json,subprocess,sys,time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/published_cascade_20261111_01'


def stage(script,logname,args=()):
    path=OUT/logname
    if path.exists():
        suffix=2
        while path.with_name(path.stem+f'_attempt{suffix}'+path.suffix).exists():suffix+=1
        path=path.with_name(path.stem+f'_attempt{suffix}'+path.suffix)
    print('Starting',script,flush=True)
    (OUT/'pipeline_status.json').write_text(json.dumps(dict(stage=script,log=str(path),complete=False),indent=2))
    with path.open('x') as f:
        result=subprocess.run([sys.executable,script,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        (OUT/'PIPELINE_FAILURE.json').write_text(json.dumps(dict(stage=script,returncode=result.returncode,log=str(path)),indent=2))
        raise RuntimeError('Stage failed: '+script+'; inspect '+str(path))
    print('Completed',script,flush=True)


def main():
    print('Waiting for the already-running four-fold model fits.',flush=True)
    while not (OUT/'execution_completed.json').exists():time.sleep(5)
    stage('complete_published_fold1.py','fold1_directory_completion.log')
    stage('verify_published_cascade.py','native_replay.log')
    # Test process is separate from latency measurement and uses synthetic recipe smoke fits.
    stage('-m','final_tests.log',('unittest','discover','-s','.','-p','test_*.py','-v'))
    stage('report_published_cascade.py','report_generation.log')
    (OUT/'pipeline_status.json').write_text(json.dumps(dict(stage='report completed',complete=True),indent=2))
    print('Full development experiment, native replay, tests and report complete.',flush=True)


if __name__=='__main__':main()
