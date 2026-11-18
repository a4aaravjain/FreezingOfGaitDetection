# Nested rescue-cascade development results

Source: verified development_only_20261111_01.csv; original combined CSV was not used for fitting/evaluation. Reserved rows were excluded during the separately authorized metadata-only export. This is historical nested development evidence, not an independent assessment.

## Comparison (completed folds only; failed folds remain unassessed)

| Arm | Early episodes | Scored unsuccessful starts | Monitored hours | Starts/hour | 95% Poisson interval /h | Unknown alerts |
|---|---:|---:|---:|---:|---:|---:|
| matched_v2 | 5/44 | 18 | 0.7897 | 22.793492598563727 | [13.508868289997375, 36.02354517242302] | 3 |
| level1_direct | 0/44 | 0 | 0.7897 | 0.0 | [0.0, 4.671247029685544] | 0 |
| logistic_rescue | 0/44 | 0 | 0.7897 | 0.0 | [0.0, 4.671247029685544] | 0 |
| baseline_OR_selected_rescue | 5/44 | 18 | 0.7897 | 22.793492598563727 | [13.508868289997375, 36.02354517242302] | 3 |
| hgb_rescue | 0/44 | 0 | 0.7897 | 0.0 | [0.0, 4.671247029685544] | 0 |
| no_score_derivatives | 0/44 | 0 | 0.7897 | 0.0 | [0.0, 4.671247029685544] | 0 |
| baseline_OR_no_score_derivatives | 5/44 | 18 | 0.7897 | 22.793492598563727 | [13.508868289997375, 36.02354517242302] | 3 |
| no_motion_transition | 0/44 | 0 | 0.7897 | 0.0 | [0.0, 4.671247029685544] | 0 |
| baseline_OR_no_motion_transition | 5/44 | 18 | 0.7897 | 22.793492598563727 | [13.508868289997375, 36.02354517242302] | 3 |

The matched baseline is fold-fitted v2 architecture with unchanged alert settings, not the original weights evaluated on people they trained on. Original saved v2 remains unchanged and its historical 7/16 result is not directly comparable to this all-development nested aggregate. Inner models use two disjoint participant groups for cross-fitted verifier training; every upstream fit excludes the scored participant, all outer people and all inner-calibration people. Head normalization is training-only. All final policies and model hashes are written before outer prediction.

## Promotion decision

{
  "promoted": false,
  "eligible_for_streaming_check": false,
  "reasons": [
    "added_episodes_not_in_two_people",
    "aggregate_false_rate_not_below_one",
    "fewer_than_two_added_episodes",
    "no_participant_macro_gain",
    "one_or_more_incomplete_folds",
    "participant_false_rate_not_below_one",
    "physical_transport_and_actuator_latency_not_measured"
  ],
  "additional_episodes": 0,
  "additional_participants": 0,
  "participant_macro_sensitivity_change": 0.0,
  "limited_nested_development": true,
  "clinical_validation": false
}

No cascade is promoted. Disabled policies are recorded failures to find a rescue, not successful zero-alert predictors. The selected-family ablations were chosen using inner calibration only; outer folds did not drive family/threshold revision. No fallback threshold search was conducted.

Exact original annotation onset denominators include those hidden by missing sensors. Per-fold candidate opportunity recall, known-followup occupancy, candidate runs/hour, negative candidate labels and boundary exclusions are saved. All alerts are timed at the actual physical OR rising edge. Logical baseline requests are independent, but OR overlap can hide a physical rising edge; these cases are rejected by the preservation check.

Poisson 95% intervals are descriptive homogeneous-rate calculations and do not correct clustering, historical tuning or distribution shift. Short exposure cannot establish a dependable sub-one/hour rate. Per-person false rates are also subjected to a conservative <1/hour gate; this numerical concentration rule is a conservative reporting check, not an independently preregistered confirmatory threshold. Nominal window-end lead is not physical delivered lead: acquisition/communication/actuator delays are not measured. Any successful matched event obeys the 2ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“5-second scoring interval; that is not clinical usefulness.

## Reproducibility

Preregistration, exact partitions, source snapshots, upstream weights/normalization and histories, OOF provenance, training-candidate ledgers, threshold grids, outer score ledgers, per-window decisions, per-alert outcomes, per-episode and per-person results, and reload checks are retained in fold folders. Fold exceptions are preserved in FAILURE.json and make the aggregate incomplete. fold_coverage.json lists the entire development onset inventory and explicitly identifies unassessed participants; partial-fold ratios must not be presented as full-cohort sensitivity. Core causal/reset/leakage/export tests passed before fitting. Batch-versus-one-tick verifier score parity is checked for every outer window; this is not complete native-sample streaming parity.

Commands: `python export_development_only.py --source sensor_data.csv --output development_only_20261111_01.csv` (already completed; never overwrite export); `python nested_rescue_experiment.py` (new run refuses overwrite); `python nested_rescue_report.py`. Export manifest records allowed IDs, permitted row count, column schema, command, script and output SHA-256. Original file content hash was deliberately not calculated because it would read reserved signal/label bytes.


## Stage feasibility and limitations of the ablation table

In all three completed folds, every prespecified Level 1 threshold exceeded the 25% inner-calibration occupancy cap. The selected Level 1 threshold is therefore the disabled sentinel 1.1. No real-data logistic or boosting verifier was fitted in these folds: the empty candidate set correctly failed the both-class prerequisite. The table's zero-alert Level 1/verifier arms are DISABLED/NOT TRAINED, not measured evidence about a learned verifier's predictive quality. The requested high-recall permissive ablation cannot be a cap-compliant selected system here; raw opportunity/occupancy tradeoffs for every prespecified rule are retained in each level1_candidates.json. No infeasible rule was promoted merely to fit Level 2. The feature ablations similarly remain disabled; they do not establish an effect of removing those features. stage_completion.json and complete_oof_candidate_ledger.csv make this explicit.

All four outer folds were attempted. One failed because a cross-fitting training half lacked both classes. It was not rearranged or retried. The table summarizes only completed folds (44 onsets); the full permitted development inventory is 72 onsets. The remaining 28 onsets are unassessed, not dropped, and there is no full-cohort sensitivity estimate. Thus the run is an incomplete, unsuccessful bounded feasibility experiment, not a successful complete model comparison.

A native-sample replay subsequently passed all 11,396 decision windows in the three completed folds, including all three upstream scores, baseline starts, rescue state and actual combined physical starts. The separate original-v2 regression check passed all 2,622 native windows on the verified export and preserved baseline output dictionaries exactly inside a disabled-rescue wrapper. See native_streaming_parity.json and baseline_native_parity.json. These checks do not certify an active learned rescue verifier because none was fitted. The successful lead times minus measured desktop computation are in compute_adjusted_successful_leads.csv; acquisition, transport and actuator latency remain unmeasured.

The final promotion decision is NO. No additional episode or participant is warned, participant-macro sensitivity does not improve, the matched baseline already exceeds the final unsuccessful-start budget, and one fold remains incomplete. Preserving an already-unsuccessful baseline channel prevents an additive rescue from removing those alerts. The original saved waveform-v2 candidate and default command remain unchanged. No locked assessment occurred.
