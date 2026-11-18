"""Review a completed grouped CV run without fitting or selecting any policy.
Outputs go to a separate new folder; original run files are never edited.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from fog_pipeline import Config, policy_rejection_reasons, verify_split, save_json

def review(run, output):
    run,output=Path(run),Path(output)
    if not (run/"cv_results.json").exists():
        raise ValueError("CV is incomplete: cv_results.json is missing.")
    summary,subjects,diagnostics=[],[],[]
    for fold in sorted(run.glob("fold_*")):
        m=json.loads((fold/"manifest.json").read_text())
        verify_split(m["train_subjects"],m["validation_subjects"],m["outer_subjects"])
        if m["mode"]=="smoke":
            raise ValueError("Smoke runs are not scientific performance estimates.")
        outer=json.loads((fold/"outer_metrics.json").read_text())
        d,w=outer["device"],outer["window"]
        summary.append(dict(fold=fold.name,subjects=",".join(m["outer_subjects"]),
            policy=m["policy_status"],threshold=m["threshold"],
            window_f1=w["f1"],window_average_precision=w["pr_auc_average_precision"],
            window_prevalence=w["positives"]/w["n"],**{
                k:d[k] for k in ["total_fog_episodes","episodes_predicted_before_onset",
                "true_warning_events","total_false_alarms","monitored_seconds",
                "total_alarm_seconds","alarm_active_percentage","false_alarms_per_minute",
                "episode_sensitivity","maximum_alarm_duration_seconds"]}))
        sub=pd.read_csv(fold/"outer_subjects.csv")
        sub.insert(0,"fold",fold.name);subjects.append(sub)
        candidates=pd.read_csv(fold/"threshold_candidates.csv")
        predictions=pd.read_csv(fold/"validation_predictions.csv")
        raw=predictions.loc[predictions.target>=0,"probability"].to_numpy()
        cfg=Config(**m["config"])
        for row in candidates.to_dict("records"):
            raw_rate=float(np.mean(raw>=row["threshold"]))
            metrics={"device":row,"window":{"positive_fraction":raw_rate}}
            reasons=policy_rejection_reasons(metrics,cfg)
            original_reasons=reasons if "max_alarm_duration_seconds" in m["config"] else [
                reason for reason in reasons if reason!="continuous_alarm_too_long"]
            # Check the saved decision using its ORIGINAL protocol.
            if bool(row["feasible"]) != (len(original_reasons)==0):
                raise ValueError(f"{fold.name}: original feasibility cannot be reproduced.")
            diagnostics.append(dict(fold=fold.name,threshold=row["threshold"],
                original_feasible=bool(row["feasible"]),
                original_rejection_reasons=";".join(original_reasons),
                duration_budget_seconds=cfg.max_alarm_duration_seconds,
                passes_duration_guard=row["maximum_alarm_duration_seconds"]<=cfg.max_alarm_duration_seconds+1e-9,
                original_warning_precision=row["warning_precision"],
                maximum_alarm_duration_seconds=row["maximum_alarm_duration_seconds"],
                raw_positive_fraction=raw_rate))
    frame=pd.DataFrame(summary)
    people=pd.concat(subjects,ignore_index=True)
    if people.subject.duplicated().any():
        raise ValueError("Subject appears in multiple outer folds.")
    totals=frame[["total_fog_episodes","episodes_predicted_before_onset",
                  "total_false_alarms","monitored_seconds","total_alarm_seconds"]].sum()
    result=dict(run=str(run.resolve()),folds=len(frame),
        disabled_folds=int((frame.policy=="disabled_no_feasible_policy").sum()),
        episodes=int(totals.total_fog_episodes),
        predicted_episodes=int(totals.episodes_predicted_before_onset),
        pooled_episode_sensitivity=float(totals.episodes_predicted_before_onset/totals.total_fog_episodes),
        macro_fold_episode_sensitivity=float(frame.episode_sensitivity.mean()),
        false_alarm_starts=int(totals.total_false_alarms),
        pooled_false_alarms_per_minute=float(totals.total_false_alarms*60/totals.monitored_seconds),
        pooled_alarm_active_percentage=float(100*totals.total_alarm_seconds/totals.monitored_seconds),
        successful_subjects=people.loc[people.episodes_predicted_before_onset>0,"subject"].tolist(),
        completed_original_protocol=True,new_model_fit=False,thresholds_changed=False,
        note="Development CV; disabled folds remain in all aggregate denominators.")
    output.mkdir(parents=True,exist_ok=False)
    frame.to_csv(output/"outer_fold_summary.csv",index=False)
    people.to_csv(output/"outer_subject_summary.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(output/"validation_rejection_diagnostics.csv",index=False)
    save_json(output/"summary.json",result)
    lines=["# Completed development CV review","",
        f"Original run: {run.resolve()}","",
        f"All {len(frame)} folds completed. {result['disabled_folds']} disabled their policies on inner validation.",
        f"Outer episodes predicted: {result['predicted_episodes']}/{result['episodes']} "
        f"({100*result['pooled_episode_sensitivity']:.2f}%).",
        f"Mean fold sensitivity: {100*result['macro_fold_episode_sensitivity']:.2f}%.",
        f"Successful warnings came from: {', '.join(result['successful_subjects']) or 'none'}.","",
        "| Outer subjects | Policy | Episodes warned | Window F1 |",
        "|---|---|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['subjects']} | {r['policy']} | "
                     f"{r['episodes_predicted_before_onset']}/{r['total_fog_episodes']} | {r['window_f1']:.3f} |")
    lines += ["","The low aggregate false-alarm rate partly reflects disabled policies; it is not evidence of a useful wearable.",
        "Per-subject burden is in outer_subject_summary.csv. Pooled burden can hide poor outcomes for an individual.","",
        "validation_rejection_diagnostics.csv reproduces the original threshold constraints from INNER validation only.",
        "It separately checks continuous duration against the current default 5-second engineering budget when the old manifest has no duration limit.",
        "This is a diagnostic flag, not a retroactive threshold selection or re-evaluation of outer predictions.",
        "A duration guard rejects selection; it does not truncate alarms, prove clinical acceptability, or guarantee bounded actuator behavior on new data.","",
        "The completed v3 run is preserved. No model was refit, no original threshold was changed, and no outer result was used to optimize a threshold."]
    (output/"REVIEW.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
    print("REVIEW:",output)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    review(args.run,args.output)

