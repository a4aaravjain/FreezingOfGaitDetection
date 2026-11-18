Work autonomously as the lead researcher and machine-learning engineer for:

`C:\Users\victo\Desktop\Aarav\Science Fair 2026-2027`

The previous rescue cascade was implemented but never trained or evaluated. Therefore, performance did not get worse; there is simply no new performance result.

Your mission is now to build, train, and rigorously evaluate the strongest possible causal freezing-of-gait early-warning system using:

1. The existing verified v2 model.
2. The implemented same-tick rescue cascade.
3. Credible published open-source FoG models and feature encoders.
4. A high-recall first level followed by a high-precision verification level.
5. Subject-independent evaluation and strict preservation of the ≥2-second prediction horizon.

Do not stop after planning, preflight, or implementation. Complete the development-only experiment unless a genuine technical failure prevents it.

## Data-export authorization

You are authorized to create a development-only export from the combined FoG-STAR CSV.

Proceed without asking again under these restrictions:

- Inspect the header to locate `subjectID`.
- Stream through the file and read only the minimum information necessary to determine participant ID before filtering.
- Retain complete rows only for participants 1–18.
- Do not parse, analyze, summarize, count labels, or retain sensor/label values for participants 19–22.
- Reading the raw row bytes and participant-ID field solely to exclude reserved participants is authorized.
- Do not report anything about participants 19–22 except that their rows were excluded.
- Write a new immutable development-only file.
- Preserve the original combined file unchanged.
- Record the filtering source, command, input hash, output hash, schema, row count, and included participant IDs.
- Add tests proving that the export contains only participants 1–18.
- All subsequent development loaders must use this export, not the combined file.

This does not authorize evaluating participants 19–22.

## Study established open-source work

Search the internet, papers, and GitHub for credible FoG code. Prioritize official repositories connected to peer-reviewed work, including:

- FoG-STAR single-ankle CNN:  
  `https://github.com/Lu1g1n0/FoG-detection-using-a-single-ankle-mounted-sensor`

- Stanford DeepFOG:  
  `https://github.com/stanfordnmbl/imu-fog-detection`

- Stanford/Mobilize DeepFOG documentation:  
  `https://mobilize.stanford.edu/software/deepfog/`

- FoG-STAR dataset utilities:  
  `https://github.com/CristianTuretta/FoG-STAR`

- Published two-block initial-prediction/refinement method:  
  `https://pmc.ncbi.nlm.nih.gov/articles/PMC10865632/`

- Public 1D-ResNet FoG implementation:  
  `https://github.com/janbrederecke/fog`

- GaitFM self-supervised gait representation code:  
  `https://github.com/saiketa/GaitFM`

You may find additional sources, but prefer published, official, reproducible repositories over unsupported or one-commit projects.

For every external source:

1. Record its URL, paper, license, commit hash, data assumptions, sensor placement, sampling rate, window definition, and prediction target.
2. Inspect the code before executing it.
3. Do not copy code without a compatible license.
4. Preserve required attribution.
5. Do not execute downloaded preprocessing that may overwrite local data.
6. Place external code in a clearly separated read-only directory.
7. Snapshot only the adapted components used by this project.
8. Explain every change needed to make the method causal and compatible with right-ankle 60-Hz acceleration and gyroscope data.

A published FoG detection score is not evidence of early prediction. Many repositories classify FoG that is already occurring. Use their encoders, training methods, refinement blocks, augmentation, and feature extraction only after adapting them to the fixed 2–5-second pre-onset target.

Reject any method that requires:

- Future samples.
- Bidirectional inference.
- Centered or zero-phase filtering.
- Whole-recording normalization.
- Test-participant statistics.
- Active-FoG labels at inference.
- Backward sequence refinement.
- Subject or task identity as predictive input.

## Target architecture

Build a baseline-preserving multi-model cascade:

```text
                           +--> Frozen v2 baseline --------+
Right-ankle causal IMU ----|                               |
                           +--> High-recall ensemble        |
                                  |                        |
                                  v                        |
                           Learned verifier/refiner         |
                                  |                        |
                                  +------ final OR --------+
                                             |
                                     one-second alert
```

The final output is:

`frozen v2 alert OR verified rescue alert`

The frozen v2 path must remain unchanged. Rescue logic may add warnings but may never suppress or delay a v2 warning.

## Level 1: high-recall ensemble

Build Level 1 from a small ensemble of complementary, proven components:

1. Existing v2 waveform and motion heads.
2. An adapted version of the official FoG-STAR single-ankle CNN encoder.
3. An adapted causal Stanford DeepFOG-style CNN.
4. One strong causal temporal model chosen from:
   - causal TCN,
   - 1D ResNet,
   - InceptionTime-style network,
   - or a causal GRU.

Do not use a bidirectional model.

Use published architecture and preprocessing choices where compatible, but change the output target to:

- Positive: an onset will occur 2–5 seconds after the current window endpoint.
- Negative: known follow-up exists and no onset occurs in that interval.
- Excluded from binary training: unknown recording tails and active FoG, unless active FoG is deliberately used as a negative hard example in a separately declared comparison.
- Included in continuous replay: every valid inference window, including active FoG and recording tails.

Level 1 should be deliberately open. On inner development folds, target:

- Preferably 100% episode opportunity recall.
- At least 95% episode opportunity recall if 100% would require trivial always-positive output.
- At least one candidate within the valid 2–5-second interval.
- Candidate-positive occupancy preferably below 25% and never above 40%.
- No use of final false-alarm constraints at this level.

Use the union or learned stacking of model scores. Level 1 is not the final alarm and may produce extra candidates.

## Level 2: learned verifier and refinement block

Level 2 must reject ordinary walking changes, stopping, turning, active FoG, post-FoG recovery, and isolated noisy Level 1 peaks.

Adapt ideas from published prediction-refinement work, but make the verifier strictly causal.

At each Level 1 candidate, provide Level 2 with:

- Every Level 1 component score.
- Cross-model agreement and disagreement.
- Score slopes and curvature over the past 0.5, 1, 2, and 3 seconds.
- Duration and consistency above permissive thresholds.
- Causal gait periodicity and loss-of-periodicity measures.
- Acceleration and gyroscope energy changes.
- Frequency-band energy computed only from the trailing window.
- Freeze-index-style features computed causally.
- Step-cycle consistency and autocorrelation.
- Waveform-shape features.
- Current-versus-earlier-window differences.
- Sensor-validity and reset-state indicators.
- Time since the previous candidate or alert.

The verifier must decide on the same inference tick. It may not wait for future samples.

Train it using cross-fitted candidates:

- Generate each training candidate using a Level 1 model that did not train on that participant.
- Positive candidates precede onset by 2–5 seconds.
- Negative candidates have sufficient follow-up and no onset in that interval.
- Unknown candidates remain unknown.
- Hard-negative mining must include stopping, turning, late 0–2-second precursors, active FoG, post-FoG motion, and high-scoring non-FoG segments.
- Prevent recordings with many windows from dominating participants with fewer recordings.

Compare a bounded set of verifier models:

1. Regularized logistic regression.
2. Small histogram gradient boosting.
3. A compact causal TCN refinement block based on a credible published implementation.

Select among these using inner calibration only.

## Training improvements from published systems

Test these training strategies in a controlled way:

- Subject-balanced batches.
- Episode-balanced positive sampling.
- Focal loss or class-balanced loss.
- Hard-negative mining.
- Rotation augmentation applied only to training data.
- Small realistic sensor noise and scale augmentation.
- Masked-signal or contrastive pretraining using only allowed training participants.
- Encoder initialization from compatible published code, followed by causal pre-FoG fine-tuning.
- Model-score calibration on separate inner participants.
- Deep ensembles or seed ensembles only if every seed is retained and reported.

Do not use augmentation that changes event timing or accesses future data.

## Evaluation

Use nested, subject-grouped development evaluation across participants 1–18.

Every outer participant must be excluded from:

- Model fitting.
- Normalization fitting.
- Pretraining.
- Threshold selection.
- Verifier training.
- Hard-negative selection.
- Early stopping.
- Model-family selection.

All model and threshold selection occurs inside each outer fold.

Compare:

1. Frozen v2.
2. Each published-code adaptation alone.
3. Level 1 directly.
4. Level 1 plus each verifier.
5. Frozen v2 OR the best rescue cascade.
6. Important ablations showing which external components actually help.

Report both episode-weighted and participant-macro results.

## Final performance target

The combined system should aim for:

- More successfully warned episodes than v2.
- Additional detections across at least two participants.
- Final unsuccessful warnings below 1 per known monitored hour.
- Every successful final warning beginning at least 2 seconds before onset.
- No suppression or delay of existing v2 alerts.
- One-second output pulses.
- No increase in unknown boundary alerts.
- Better participant-macro sensitivity.
- Acceptable streaming latency.

Report:

- Episode sensitivity.
- Participant-macro sensitivity.
- Per-person outcomes.
- Unsuccessful starts/hour.
- Total monitored hours.
- False-alert occupancy.
- Warning precision.
- Unknown alerts.
- Lead-time distribution.
- Candidate recall.
- Candidate occupancy.
- Verifier rejection rate.
- Median, p95, and maximum latency.
- Confidence intervals or honest uncertainty estimates.

If monitored exposure is too short to establish a dependable sub-one-per-hour rate, say so explicitly.

## Promotion rule

Promote the cascade only if it:

- Preserves every matched v2 alert.
- Adds at least two successfully predicted episodes.
- Adds detections across at least two people.
- Keeps final unsuccessful starts below 1/hour.
- Maintains at least a two-second actual output lead.
- Does not increase unknown alerts.
- Improves participant-macro sensitivity.
- Passes reload and streaming parity.
- Passes every causality, split, reset, and leakage test.

Otherwise, keep v2 and identify exactly which component failed.

## Complete the run

Do not stop after:

- Writing code.
- Passing unit tests.
- Downloading repositories.
- Creating the development export.
- Producing a preflight report.
- Training only one fold.
- Discovering that the first model failed.

Complete the predefined nested experiment and all prespecified comparisons. Do not add new experiments after examining outer-fold outcomes.

Use available GPU acceleration if present. If only CPU is available, use efficient models and checkpoint every fold so work can resume safely.

## Required deliverables

Create a new uniquely named run directory containing:

- Development-export audit.
- External-source and license audit.
- Exact repository commit hashes.
- Preregistered experiment plan.
- Subject splits.
- Adapted source snapshots.
- Model configurations.
- Training histories.
- Candidate and verifier ledgers.
- Per-window predictions.
- Per-alert and per-episode outcomes.
- Per-person metrics.
- Threshold grids.
- False-alarm exposure calculations.
- Comparison and promotion decisions.
- Saved models.
- Reload and streaming verification.
- All failed results.
- A final Markdown report.

The report must begin with one of these statements:

- “The cascade passed every promotion requirement and improved development performance.”
- “The cascade improved some metrics but failed one or more promotion requirements.”
- “The cascade did not improve development performance.”

Do not say “improved” based solely on training metrics, window accuracy, F1, or results from a FoG-detection target. Improvement requires additional valid 2–5-second episode warnings under the complete final alarm protocol.

Begin now by creating the authorized development-only export, auditing the strongest published repositories, freezing the experiment matrix, and then running the complete development experiment.