# Causal rescue cascade: implementation and safe preflight

Status: prototype implemented and tested; proper nested participant experiment BLOCKED by missing eligible inputs. No model improvement or promotion is claimed. Frozen v2 remains unchanged/default. No raw or reserved dataset was opened.

## Implemented
- Label-free same-tick rescue feature extraction (65 numeric features), independent baseline channel, latched rescue alarm with causal rearming, and physical OR output with rising edges measured at output time.
- Three preregistered Level 1 thresholds and two fixed verifier families: regularized logistic regression and small histogram boosting. The verifier ledger consumer rejects missing provenance, held-person training overlap, outer-person upstream overlap and inner-calibration upstream overlap. Unknown candidate labels are excluded, including unknown active-state tails. Current labels/IDs never enter the feature vector.
- Full, no-score-derivative and no-motion-transition feature masks are implemented in the verifier fitter. No best family or ablation winner has been selected.
- Physical preservation checks explicitly detect rescue/baseline pulse collisions. Logical OR preserves the baseline channel but cannot guarantee a distinct physical rising edge if rescue is already active. Any such matched-development collision blocks promotion, as does excessive physical pulse length. No universal unseen-person preservation claim is justified for physical OR pulses.

## Tests and experiments actually run
105 tests passed with command:
`python -m unittest test_rescue_cascade test_locked_assessment test_development_evidence_review test_candidate_gate test_waveform_ensemble test_waveform_bank test_fast_temporal_bank test_pipeline`.
Ten tests concern the new cascade: prefix state, label independence, unknown labels, leakage provenance, same-tick decisions, baseline-channel equality, reset behavior, collision detection, Poisson calculation and both model-family save/reload on synthetic candidates. Synthetic fits are software checks, not participant results; temporary fixtures are not research checkpoints.

The real-data preflight read only the existing saved OOF development score ledger, restricted analysis to people11–14, and tested the three fixed thresholds on its ONE available dense head. Occupancy proxies were 44.94%, 29.19% and 16.79% respectively. These are partial-input, eligible/active-window exposure proxies, not full three-head candidate occupancy or complete-followup denominator estimates. Exact onset inventories and other heads are absent, so candidate opportunity recall is unreported and no threshold is selected. Do not interpret these results as the required nested experiment or as evidence of 95% candidate recall.

The retained baseline has zero scored unsuccessful starts over0.18156 hours. A descriptive exact two-sided95% homogeneous-Poisson interval is [0,20.32] starts/hour. With zero events, about3.689 total hours would be needed merely to put that model-based upper limit below1/hour, about3.507 additional hours beyond the current exposure. This does not correct development selection, clustering or distribution shift, and does not establish clinical performance.

## Why the nested experiment is not run
The only FoG-STAR raw file found is combined with reserved participants. The instructions prohibit opening reserved rows before filtering. The old loader is therefore not used. Existing OOF cached predictions do not provide all three heads/motion features and cannot enforce exclusion of a proposed outer group from every upstream verifier-training fit. Reusing them as if fully nested would leak outer information. Development-only raw data or complete, verified nested score/motion ledgers are required.

Not yet completed: fitting fold-specific three-head upstream models, generating real nested candidate ledgers, inner selection of final verifier thresholds using episode/burden metrics, held-out outer evaluation, all real-data ablations, full cascade native streaming parity, target-hardware latency and promotion assessment. The ledger consumer is an implementation component, not a complete nested upstream-model trainer. No missing outcome is reported as zero or successful.

## Next required input
A development-only FoG-STAR export for people1–18, preserving all rows/annotations for those people and containing no reserved people, is needed. No export from the combined raw file was attempted under the present access restriction. An explicit clarification allowing subject-ID-only partitioning without inspecting reserved sensor/label values would provide another path; the current instruction is interpreted strictly. Stanford6–7 remain sealed.

## Preservation
Baseline artifact hashes checked against the stored manifest; the weights, configuration and source were not edited. The experiment plan was written before fitting or selecting a rescue model. Model configurations, feature list, fold groups, required comparisons, source snapshots, preflight CSV/JSON, test outcome and this report are retained. This experiment is incomplete, not an unsuccessful fully evaluated cascade, and not a newly verified model.
