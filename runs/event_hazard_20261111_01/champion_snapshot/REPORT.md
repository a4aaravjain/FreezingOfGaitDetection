The cascade did not improve development performance.

# Published-model rescue cascade: complete development results

This is corrected physical-session, nested participant development. Reserved participants remained excluded; no locked assessment was run. All architectures, recipes, folds, calibration splits, thresholds and ablations were frozen before fitting. No additional experiment was selected after outer results.

## Main paired comparison

| System | Early episodes | Participant macro | Unsuccessful starts | Known hours | Starts/known hour | Unknown starts |
|---|---:|---:|---:|---:|---:|---:|
| matched_v2 | 15/72 | 0.1294261294261294 | 58 | 1.1739 | 49.408 | 0 |
| baseline_OR_best_rescue | 15/72 | 0.1294261294261294 | 58 | 1.1739 | 49.408 | 0 |

The nested baseline is the fixed v2 architecture/training recipe refitted without each outer group. Original saved v2 trained people1-14 and therefore cannot be a subject-independent baseline on those people. Its unchanged weights are separately compared with the selected rescue only on historical people15-18; see fold4 original_saved_v2 results. The frozen baseline path is logically independent; physical OR collisions and extended pulses are explicitly tested rather than called preserved merely because requests were ORed.

## Candidate and verifier results

candidate_metrics.csv reports opportunity recall against every original onset, eligible-window recall and known-time candidate occupancy. risk_blind_capacity_control.json in each fold exposes whether candidate coverage exceeds a trivial equal-capacity selector. A causal25%/33.3% capacity gate prevents distribution shift from producing an always-positive candidate stream. Failure to reach95% is retained and does not skip verifier training.

All prespecified standalone components, training recipes, fixed.5 diagnostic policies, raw/Platt verifiers, calibrated standalone/combined policies and external/motion/history ablations are tabulated in all_comparisons.csv. verifier_rejection_rates.csv distinguishes classifier rejection at fixed.5 from calibration-disabled policies. Missing arms remain explicitly partial with paired baseline coverage; they are never silently scored as zero or compared to a different denominator. Disabled safe policies are failures to find useful warnings, not successful low-false-alarm models.

calibration_policy_constraints.csv identifies whether the preserved baseline already exceeded the inner warning budget or whether no adding rescue policy met the complete constraints. These are distinct failure modes: an additive verifier cannot repair an already excessive preserved baseline warning rate.

## Technical recovery

Initial fold-one verifier attempts stopped before fitting because parent output directories were absent. Later attempts saved fitted weights and passed reload checks, then failed while serializing NumPy participant IDs to JSON. These failed attempts are retained under technical_attempts. Completion creates the missing directories and converts metadata scalars to native JSON values. It reuses saved weights only when model family and the complete training-membership/label ledger match; any initial fallback-family ablation is archived and the originally prespecified calibration-selected family is fitted. Cached decimal scores are read with round-trip precision. No upstream weights, training labels, hyperparameters, folds, threshold grids or experiment choices were changed. See bookkeeping_recovery_ledger.json.

## Source adaptations and training

The external_source_license_audit.json and EXTERNAL_SOURCE_AUDIT.md record URLs, exact commits, papers, licenses, sensor assumptions and adaptations. Stanford BSD3 and ResNet MIT notices are retained. Unlicensed FoGSTAR/GaitFM source is not imported or copied; architectural/masked-learning ideas are independently implemented. External detection scores and externally trained weights are not used as early-warning evidence.

FoGSTAR/Stanford causal CNNs and compact ResNet variants use right-ankle acceleration(g) and gyroscope(degrees/s) at60Hz, with fit-only normalization. ResNet recipe comparisons add participant/episode-balanced focal training, fit-only hard-negative mining, rotation/noise/scale augmentation, then fit-only masked pretraining. They never change event timing. The two-stage TCN verifier replaces symmetric refinement with left-causal convolutions over current/past score features.

Sensor placement: these fitted models require the right ankle and the same six-channel axis/unit convention used by the export. The official single-ankle architecture originally used a left-ankle signal; adapting its topology does not make left/right inputs interchangeable. This experiment does not compare ankle sides, so it cannot establish which side is intrinsically best. Neither ankle has been established as a dependable real-world early-warning placement by this development experiment.

Each verifier fitting candidate comes from an upstream model excluding that participant, the outer group and both calibration groups. Its candidate cutoff is calibrated using that same upstream model on separate policy-calibration people. Unknown tails remain unknown. Active/late/postFoG/stopping/turning negatives are training annotations only; they are not inference inputs. Platt calibration uses separate people from operating-policy selection.

## Promotion and uncertainty

```json
{
  "promoted": false,
  "gates": {
    "all_folds_complete": true,
    "all_prespecified_fit_comparisons_completed": true,
    "baseline_actual_starts_preserved": true,
    "prior_episode_warnings_not_lost_or_later": true,
    "two_added_episodes": false,
    "additions_in_two_people": false,
    "unsuccessful_rate_below_one": false,
    "unknown_alerts_not_increased": true,
    "participant_macro_improved": false,
    "pulses_one_second": true,
    "candidate_occupancy_below_40": true,
    "native_reload_parity": true,
    "streaming_latency": false,
    "desktop_adjusted_lead_two_seconds": true,
    "actual_delivered_lead_verified": false,
    "original_saved_v2_starts_preserved": true,
    "original_saved_v2_combined_below_one": false,
    "original_saved_v2_two_additions_two_people": false,
    "causality_split_reset_leakage_tests": true
  },
  "failed": [
    "two_added_episodes",
    "additions_in_two_people",
    "unsuccessful_rate_below_one",
    "participant_macro_improved",
    "streaming_latency",
    "actual_delivered_lead_verified",
    "original_saved_v2_combined_below_one",
    "original_saved_v2_two_additions_two_people"
  ],
  "added_episodes": 0,
  "added_people": [],
  "locked_assessment_executed": false,
  "default_v2_weights_changed": false
}
```

All uncertainty estimates are descriptive. Poisson intervals assume a homogeneous warning rate; the participant bootstrap does not undo repeated historical tuning or overlapping fits. Short known-monitor exposure cannot establish a dependable population rate below one/hour even if observed unsuccessful starts are zero.

Training-strategy contrasts are prespecified sequential recipe comparisons, not a factorial decomposition. Balancing/focal loss is one bundle, and the mining recipe includes two additional hard-negative epochs. Improvements cannot be attributed to a single bundled ingredient in isolation.

Lead times are actual algorithm-output start timestamps relative to annotations, with measured desktop computation additionally subtracted in compute_adjusted_leads.csv. Sensor/communication/actuator delays are unmeasured, so the required two-second delivered warning is unverified. Model setup/compilation on synthetic zeros is reported separately from live-stream inference latency.

All failures, training histories, model configurations/provenance, candidate ledgers, per-window scores, per-alert and per-episode outcomes, per-person metrics, threshold grids, exposure totals and reload/native checks are retained in this directory. The saved v2 weights remain unchanged.

## Streaming latency

Selected live cascade median 32.711ms; p95 48.324ms; maximum 374.860ms. Setup/compilation and sensor/transport/actuator delay are excluded and reported separately.

## Constraint imposed by preserving baseline warnings

The original saved baseline already emits 3 unsuccessful starts on this historical cohort (15.944 per known monitored hour). Under fixed annotations and the same warning horizon, preserving every physical baseline start prevents an additive rescue path from removing that existing unsuccessful-warning burden. A useful rescue can add true warnings, but this OR-only design cannot reach a lower false-start rate than its preserved baseline floor on these recordings. Changing the baseline warning policy would require a separately prespecified study; no such change was made here.

## All frozen comparisons

| model | folds | episodes | detected | macro | unsuccessful | known_hours | unsuccessful_per_known_hour | unknown |
|---|---|---|---|---|---|---|---|---|
| base_alone_calibrated | 4 | 72 | 0 | 0 | 0 | 1.1739 | 0 | 0 |
| baseline_OR_best_rescue | 4 | 72 | 15 | 0.12943 | 58 | 1.1739 | 49.408 | 0 |
| extended_alone_calibrated | 4 | 72 | 0 | 0 | 0 | 1.1739 | 0 | 0 |
| fogstar_alone_calibrated | 4 | 72 | 0 | 0 | 2 | 1.1739 | 1.7037 | 0 |
| fogstar_alone_fixed05 | 4 | 72 | 25 | 0.30705 | 250 | 1.1739 | 212.97 | 2 |
| hgb_platt_alone | 4 | 72 | 0 | 0 | 0 | 1.1739 | 0 | 0 |
| hgb_platt_combined | 4 | 72 | 15 | 0.12943 | 58 | 1.1739 | 49.408 | 0 |
| hgb_platt_fixed05 | 4 | 72 | 7 | 0.13333 | 72 | 1.1739 | 61.334 | 2 |
| hgb_raw_alone | 4 | 72 | 0 | 0 | 0 | 1.1739 | 0 | 0 |
| hgb_raw_combined | 4 | 72 | 15 | 0.12943 | 58 | 1.1739 | 49.408 | 0 |
| hgb_raw_fixed05 | 4 | 72 | 1 | 0.0083333 | 6 | 1.1739 | 5.1112 | 0 |
| level1_direct | 4 | 72 | 65 | 0.84498 | 1132 | 1.1739 | 964.31 | 22 |
| logistic_platt_alone | 4 | 72 | 0 | 0 | 0 | 1.1739 | 0 | 0 |
| logistic_platt_combined | 4 | 72 | 15 | 0.12943 | 58 | 1.1739 | 49.408 | 0 |
| logistic_platt_fixed05 | 4 | 72 | 1 | 0.0069444 | 19 | 1.1739 | 16.185 | 1 |
| logistic_raw_alone | 4 | 72 | 0 | 0 | 0 | 1.1739 | 0 | 0 |
| logistic_raw_combined | 4 | 72 | 15 | 0.12943 | 58 | 1.1739 | 49.408 | 0 |
| logistic_raw_fixed05 | 4 | 72 | 3 | 0.020833 | 12 | 1.1739 | 10.222 | 1 |
| matched_v2 | 4 | 72 | 15 | 0.12943 | 58 | 1.1739 | 49.408 | 0 |
| no_fogstar | 4 | 72 | 15 | 0.12943 | 58 | 1.1739 | 49.408 | 0 |
| no_history | 4 | 72 | 15 | 0.12943 | 58 | 1.1739 | 49.408 | 0 |
| no_motion | 4 | 72 | 15 | 0.12943 | 58 | 1.1739 | 49.408 | 0 |
| no_resnet | 4 | 72 | 15 | 0.12943 | 58 | 1.1739 | 49.408 | 0 |
| no_stanford | 4 | 72 | 15 | 0.12943 | 58 | 1.1739 | 49.408 | 0 |
| original_saved_v2 | 1 | 16 | 6 | 0.23333 | 3 | 0.18815 | 15.944 | 1 |
| original_saved_v2_OR_rescue | 1 | 16 | 6 | 0.23333 | 3 | 0.18815 | 15.944 | 1 |
| resnet_augmented_alone_calibrated | 4 | 72 | 0 | 0 | 0 | 1.1739 | 0 | 0 |
| resnet_augmented_alone_fixed05 | 4 | 72 | 12 | 0.13889 | 111 | 1.1739 | 94.557 | 3 |
| resnet_balanced_alone_calibrated | 4 | 72 | 0 | 0 | 0 | 1.1739 | 0 | 0 |
| resnet_balanced_alone_fixed05 | 4 | 72 | 19 | 0.21806 | 170 | 1.1739 | 144.82 | 1 |
| resnet_core_alone_calibrated | 4 | 72 | 0 | 0 | 0 | 1.1739 | 0 | 0 |
| resnet_core_alone_fixed05 | 4 | 72 | 17 | 0.18994 | 172 | 1.1739 | 146.52 | 1 |
| resnet_mined_alone_calibrated | 4 | 72 | 0 | 0 | 0 | 1.1739 | 0 | 0 |
| resnet_mined_alone_fixed05 | 4 | 72 | 20 | 0.26944 | 165 | 1.1739 | 140.56 | 1 |
| resnet_pretrained_alone_calibrated | 4 | 72 | 0 | 0 | 0 | 1.1739 | 0 | 0 |
| resnet_pretrained_alone_fixed05 | 4 | 72 | 26 | 0.19494 | 140 | 1.1739 | 119.26 | 0 |
| stanford_alone_calibrated | 4 | 72 | 0 | 0 | 0 | 1.1739 | 0 | 0 |
| stanford_alone_fixed05 | 4 | 72 | 45 | 0.54308 | 497 | 1.1739 | 423.38 | 8 |
| tcn_platt_alone | 4 | 72 | 0 | 0 | 0 | 1.1739 | 0 | 0 |
| tcn_platt_combined | 4 | 72 | 15 | 0.12943 | 58 | 1.1739 | 49.408 | 0 |
| tcn_platt_fixed05 | 4 | 72 | 0 | 0 | 0 | 1.1739 | 0 | 0 |
| tcn_raw_alone | 4 | 72 | 0 | 0 | 0 | 1.1739 | 0 | 0 |
| tcn_raw_combined | 4 | 72 | 15 | 0.12943 | 58 | 1.1739 | 49.408 | 0 |
| tcn_raw_fixed05 | 4 | 72 | 0 | 0 | 0 | 1.1739 | 0 | 0 |
| waveform_alone_calibrated | 4 | 72 | 0 | 0 | 0 | 1.1739 | 0 | 0 |
