"""FoG research entry point. Default runs development only, never S09/S10.
See AUDIT.md and README.md. Historical code/models are preserved in legacy/.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import sys
import warnings
import numpy as np
import pandas as pd
from fog_pipeline import (Config, load_subjects, build_windows, fit_normalizer,
                          normalize, verify_split, save_json, sha256,
                          select_threshold, evaluate, OnlinePredictor)

ROOT = Path(__file__).resolve().parent
TRAIN = ["S01","S02","S03","S04","S05","S06"]
VALIDATION = ["S07","S08"]
HISTORICAL = ["S09","S10"]
# All local Daphnet subjects have development exposure, including old test users.
EXPOSED = [f"S{i:02}" for i in range(1,11)]

def build_model(cfg,architecture="cnn"):
    if architecture != "cnn":
        from temporal_models import build_temporal_model
        return build_temporal_model(cfg,architecture)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(cfg.seed)
    tf.config.experimental.enable_op_determinism()
    model = tf.keras.Sequential([
        tf.keras.layers.Input((cfg.history_rows,3)),
        tf.keras.layers.Conv1D(16,7,padding="causal",activation="relu"),
        tf.keras.layers.Conv1D(32,5,dilation_rate=2,padding="causal",activation="relu"),
        tf.keras.layers.Conv1D(32,5,dilation_rate=4,padding="causal",activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(.25),
        tf.keras.layers.Dense(1,activation="sigmoid")
    ], name="ankle_cnn_v3")
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(curve="PR",name="pr_auc")])
    return model

def predict(model,x,batch_size):
    # Direct batched calls also reproduce the callable used by OnlinePredictor.
    return np.concatenate([np.asarray(model(x[i:i+batch_size],training=False)).ravel()
                           for i in range(0,len(x),batch_size)])

def save_evaluation(out,name,data,probabilities,threshold,cfg):
    metrics,replayed,events = evaluate(data,probabilities,threshold,cfg)
    save_json(out/f"{name}_metrics.json",metrics)
    replayed.to_csv(out/f"{name}_predictions.csv",index=False)
    pd.DataFrame(events).to_csv(out/f"{name}_events.csv",index=False)
    per_subject = []
    from fog_pipeline import Windows
    for subject in sorted(data.meta.subject.unique()):
        mask = data.meta.subject.to_numpy() == subject
        subset = Windows(data.x[mask],data.y[mask],data.meta[mask].reset_index(drop=True),
                         [s for s in data.segments if s.subject == subject])
        m,_,_ = evaluate(subset,probabilities[mask],threshold,cfg)
        per_subject.append(dict(subject=subject,
                                **{f"window_{k}":v for k,v in m["window"].items() if not isinstance(v,(dict,list))},
                                **{k:v for k,v in m["device"].items() if not isinstance(v,list)}))
    pd.DataFrame(per_subject).to_csv(out/f"{name}_subjects.csv",index=False)
    d = metrics["device"]
    if d["smoothed_positive_fraction"] < .001 or d["smoothed_positive_fraction"] > .95:
        warnings.warn(f"{name}: almost all predictions have the same decision.")
    if d["alarm_active_fraction"] > cfg.max_alarm_fraction:
        warnings.warn(f"{name}: alarm-active budget exceeded.")
    if d["maximum_alarm_duration_seconds"] > cfg.max_alarm_duration_seconds + 1e-9:
        warnings.warn(f"{name}: continuous alarm exceeds {cfg.max_alarm_duration_seconds:g}s budget.")
    w = metrics["window"]
    if w.get("precision") is not None and w["precision"] > .9 and w["recall"] > .9:
        warnings.warn("Unexpectedly high prediction performance: inspect leakage.")
    return metrics

def run_development(root,out,cfg,train=TRAIN,validation=VALIDATION,outer=(),smoke=False,architecture="cnn"):
    verify_split(train,validation,outer)
    if (set(train)|set(validation)|set(outer)) & set(HISTORICAL):
        raise ValueError("Historical S09/S10 cannot enter development mode.")
    out.mkdir(parents=True,exist_ok=False)
    cfg.validate()
    train_segments,train_files = load_subjects(root,train,cfg)
    val_segments,val_files = load_subjects(root,validation,cfg)
    train_data, val_data = build_windows(train_segments,cfg), build_windows(val_segments,cfg)
    for name,data in [("train",train_data),("validation",val_data)]:
        if set(data.y[data.y >= 0]) != {0,1}:
            raise ValueError(f"{name} requires both classes.")
    params = fit_normalizer(train_segments)
    mask = train_data.y >= 0
    x,y = normalize(train_data.x[mask],params),train_data.y[mask].astype(np.float32)
    rng = np.random.default_rng(cfg.seed)
    if smoke:
        # Plumbing check only: deterministic subset, not performance evidence.
        take = np.r_[rng.choice(np.flatnonzero(y==0),min(384,int((y==0).sum())),replace=False),
                     rng.choice(np.flatnonzero(y==1),min(128,int((y==1).sum())),replace=False)]
        x,y = x[take],y[take]
    # Single imbalance correction, no undersampling in scientific runs.
    positive_weight = min(float((y==0).sum()/(y==1).sum()),cfg.positive_weight_cap)
    weights = np.where(y==1,positive_weight,1.).astype(np.float32)
    weights /= weights.mean()
    val_x = normalize(val_data.x,params)
    vm = val_data.y >= 0
    manifest = dict(architecture=architecture,schema_version=4,mode="smoke" if smoke else "development",
        created_utc=datetime.now(timezone.utc).isoformat(),config=asdict(cfg),
        train_subjects=list(train),validation_subjects=list(validation),
        outer_subjects=list(outer),historically_exposed_subjects=EXPOSED,
        legitimate_final_test_available=False,external_datasets_enabled=False,
        normalization=params,positive_class_weight=positive_weight,
        imbalance="single capped sample weighting; no resampling in research runs",
        target=f"next onset in [t+{cfg.minimum_lead_seconds}s,t+{cfg.horizon_seconds}s]; active/censored excluded from loss only",
        training_windows=len(y),validation_windows=len(val_data.y),
        dataset_sha256={p.name:sha256(p) for p in train_files+val_files},
        source_sha256={n:sha256(ROOT/n) for n in ["ankles.py","fog_pipeline.py","temporal_models.py"]},
        python=sys.version,platform=platform.platform(),
        packages={n:importlib.metadata.version(n) for n in ["numpy","pandas","scikit-learn","tensorflow","keras"]})
    save_json(out/"manifest.json",manifest)
    # Keep exact source with new runs so later code revisions remain reproducible.
    import shutil
    (out/"source").mkdir()
    for name in manifest["source_sha256"]:
        shutil.copy2(ROOT/name,out/"source"/name)
    print(f"Training subjects: {train}; validation subjects: {validation}",flush=True)
    print(f"Output: {out}",flush=True)
    model = build_model(cfg,architecture)
    (out/"architecture.json").write_text(model.to_json(),encoding="utf-8")
    import tensorflow as tf
    history = model.fit(x,y,sample_weight=weights,
        validation_data=(val_x[vm],val_data.y[vm]),
        batch_size=cfg.batch_size,epochs=1 if smoke else cfg.epochs,shuffle=True,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc",mode="max",patience=5,restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(str(out/"best.keras"),monitor="val_pr_auc",mode="max",save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=.5,patience=2,min_lr=1e-6),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.CSVLogger(str(out/"training.csv"))],verbose=2)
    if not all(np.isfinite(v).all() for v in history.history.values()):
        raise ValueError("Nonfinite training metrics.")
    model.save(out/"model.keras")
    probs = predict(model,val_x,cfg.batch_size)
    threshold,candidates = select_threshold(val_data,probs,cfg)
    pd.DataFrame(candidates).to_csv(out/"threshold_candidates.csv",index=False)
    metrics = save_evaluation(out,"validation",val_data,probs,threshold,cfg)
    manifest.update(threshold=threshold,
                    policy_status="feasible" if threshold <= 1 else "disabled_no_feasible_policy",
                    model_sha256=sha256(out/"model.keras"))
    save_json(out/"manifest.json",manifest)
    # Reload parity and label-free replay parity on a real validation segment.
    restored = tf.keras.models.load_model(out/"model.keras")
    np.testing.assert_allclose(predict(restored,val_x[:8],cfg.batch_size),probs[:8],rtol=1e-5,atol=1e-6)
    s = next(s for s in val_segments if len(s.time) >= cfg.history_rows+3*cfg.interval_rows)
    online = OnlinePredictor(lambda z: restored(z,training=False),params,threshold,cfg)
    outputs = [v for t,xyz in zip(s.time[:cfg.history_rows+3*cfg.interval_rows],s.xyz_g)
               if (v:=online.step(float(t),xyz)) is not None]
    indices = val_data.meta.index[val_data.meta.segment==s.key][:len(outputs)]
    np.testing.assert_allclose([v["probability"] for v in outputs],probs[indices],rtol=1e-5,atol=1e-6)
    save_json(out/"verification.json",dict(model_reload=True,online_probability_parity=True))
    # Outer subjects are loaded only AFTER model and threshold are frozen.
    if outer:
        segments,files = load_subjects(root,outer,cfg)
        data = build_windows(segments,cfg)
        p = predict(model,normalize(data.x,params),cfg.batch_size)
        outer_metrics = save_evaluation(out,"outer",data,p,threshold,cfg)
        save_json(out/"outer_data_sha256.json",{f.name:sha256(f) for f in files})
    else:
        outer_metrics = None
    print("RUN:",out)
    print("POLICY:",manifest["policy_status"],"THRESHOLD:",threshold)
    print("VALIDATION:",metrics)
    return outer_metrics

def audit(root,out,cfg):
    out.mkdir(parents=True,exist_ok=False)
    rows = []
    for path in sorted(root.glob("S??R??.txt")):
        segments = load_subjects(root,[path.name[:3]],cfg)[0]
        segments = [s for s in segments if s.key.startswith(path.name+":")]
        d = pd.read_csv(path,sep=r"\s+",header=None)
        t = d[0].to_numpy()/1000
        rows.append(dict(file=path.name,subject=path.name[:3],sha256=sha256(path),
                         rows=len(d),sample_rate_hz=1/np.diff(t).mean(),
                         label0=int((d[10]==0).sum()),label1=int((d[10]==1).sum()),
                         label2=int((d[10]==2).sum()),segments=len(segments),
                         observed_onsets=sum(len(s.onsets) for s in segments)))
    pd.DataFrame(rows).to_csv(out/"dataset_inventory.csv",index=False)
    # All external files inspected, but never mixed with the training adapter.
    ext = []
    for f in sorted((root/"imu").glob("*.txt")):
        d = pd.read_csv(f,sep="\t")
        ext.append(dict(file=f.name,sha256=sha256(f),rows=len(d),columns=list(d.columns),
                        finite=bool(np.isfinite(d.to_numpy()).all()),
                        sample_rate_hz=float(1/d["Time [s]"].diff().mean()),
                        labels=d["Freezing event [flag]"].value_counts().to_dict()
                        if "Freezing event [flag]" in d else None))
    f = root/"sensor_data.csv"
    if f.exists():
        d = pd.read_csv(f)
        ext.append(dict(file=f.name,sha256=sha256(f),rows=len(d),columns=list(d.columns),
            subjects=sorted(d.subjectID.unique().tolist()),
            missing_by_column=d.isna().sum().to_dict(),
            median_group_timestamp_step=float(d.groupby(["subjectID","sessionID","taskID"]).timestamp.diff().median())))
    save_json(out/"external_inventory.json",ext)
    segments,_ = load_subjects(root,TRAIN+VALIDATION,cfg)
    data = build_windows(segments,cfg)
    verify_split(TRAIN,VALIDATION,HISTORICAL)
    save_json(out/"checks.json",dict(shape=list(data.x.shape),positives=int((data.y==1).sum()),
              negatives=int((data.y==0).sum()),excluded_from_loss=int((data.y<0).sum()),
              finite=bool(np.isfinite(data.x).all()),split_verified=True,
              historical_subjects_used_for_training_or_selection=False))
    print("AUDIT:",out)

def final_evaluation(root,run,subjects,attest):
    import json
    m = json.loads((run/"manifest.json").read_text())
    if m["mode"] == "smoke" or m["policy_status"] != "feasible":
        raise ValueError("Freeze a non-smoke, feasible development model first.")
    if not attest or not subjects or set(subjects)&set(m["historically_exposed_subjects"]):
        raise ValueError("Final testing requires genuinely new subjects and --attest-unseen. S01-S10 are exposed.")
    verify_split(m["train_subjects"],m["validation_subjects"],subjects)
    for name,digest in m["source_sha256"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError("Evaluation code differs from frozen run; do not silently change protocol.")
    if sha256(run/"model.keras") != m["model_sha256"]:
        raise ValueError("Model hash changed.")
    marker=run/"FINAL_EVALUATION_STARTED.json"
    # Exclusive marker prevents accidental repeat evaluation of the frozen run.
    with marker.open("x",encoding="utf-8") as f:
        json.dump(dict(subjects=subjects,attested_unseen=True),f)
    cfg=Config(**m["config"])
    segments,files=load_subjects(root,subjects,cfg)
    data=build_windows(segments,cfg)
    import tensorflow as tf
    model=tf.keras.models.load_model(run/"model.keras")
    p=predict(model,normalize(data.x,m["normalization"]),cfg.batch_size)
    save_evaluation(run,"final",data,p,m["threshold"],cfg)
    save_json(run/"final_data_sha256.json",{f.name:sha256(f) for f in files})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["audit","develop","cv","final","summary","ensemble"],nargs="?",default="develop")
    parser.add_argument("--data-dir",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--epochs",type=int,default=Config.epochs)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--run",type=Path)
    parser.add_argument("--subjects",nargs="+")
    parser.add_argument("--attest-unseen",action="store_true")
    args=parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    cfg=replace(Config(),epochs=args.epochs)
    out=args.output or ROOT/"runs"/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+args.mode)
    if args.mode=="ensemble":
        from continuous_session_experiment import main as continuous_replay
        run=args.run or ROOT/"runs/right_ankle_waveform_v2"
        print("RIGHT-ANKLE ENSEMBLE: continuous-session development replay; not independent validation")
        print("Uses the verified development export. Task labels do not reset physical sensor history.")
        continuous_replay(output=out,model_run=run)
    elif args.mode=="summary":
        from verify_summary_stream import verify
        import json
        run=args.run or ROOT/"runs/right_ankle_pulse_candidate"
        print("HISTORICAL TASK-RESET SUMMARY CHECK; superseded for continuous-session performance")
        print(json.dumps(json.loads((run/"results.json").read_text())[0]["metrics"]["device"],indent=2))
        verify(run)
    elif args.mode=="audit":
        audit(args.data_dir,out,cfg)
    elif args.mode=="develop":
        run_development(args.data_dir,out,cfg,smoke=args.smoke)
    elif args.mode=="cv":
        # Grouped outer assessment; disjoint inner validation selects epochs/policy.
        # Development CV, not a recovered pristine final test.
        groups=[["S01","S02"],["S03","S04"],["S05","S06"],["S07","S08"]]
        out.mkdir(parents=True,exist_ok=False)
        results=[]
        for i,outer in enumerate(groups):
            val=groups[(i+1)%len(groups)]
            train=sorted(set(sum(groups,[]))-set(outer)-set(val))
            m=run_development(args.data_dir,out/f"fold_{i+1}",cfg,train,val,outer,args.smoke)
            results.append(dict(fold=i+1,subjects=outer,metrics=m))
        save_json(out/"cv_results.json",results)
        print(f"CV COMPLETE: {out}",flush=True)
        # Macro across folds; per-subject CSVs retained, no window-level CIs.
        sensitivity=[r["metrics"]["device"]["episode_sensitivity"] for r in results]
        save_json(out/"cv_summary.json",dict(kind="development grouped outer CV",
                    macro_episode_sensitivity=float(np.mean([x for x in sensitivity if x is not None])),
                    folds=len(results),untouched_final_test=False))
    else:
        if args.run is None:
            parser.error("final requires --run.")
        final_evaluation(args.data_dir,args.run,args.subjects,args.attest_unseen)

if __name__=="__main__":
    main()

