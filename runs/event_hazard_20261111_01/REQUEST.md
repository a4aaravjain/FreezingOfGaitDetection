Act as the lead machine-learning scientist and implementation engineer for:

`C:\Users\victo\Desktop\Aarav\Science Fair 2026-2027`

Your task is to produce a system that is demonstrably better than the current champion under the same corrected, participant-held-out development protocol.

Do not claim improvement unless every mandatory criterion below passes. If no challenger passes, retain the current champion and state that strict improvement was not achieved.

## Critical findings from the failed cascade

Read the complete published-model rescue report and all its artifacts before changing anything.

The previous cascade failed because:

- Matched v2 detected 15 of 72 episodes.
- Participant-macro sensitivity was 0.12943.
- It produced 58 unsuccessful starts during 1.1739 known hours: 49.408 unsuccessful starts/hour.
- The original saved v2 produced 3 unsuccessful starts in 0.18815 known hours: 15.944/hour.
- An additive `v2 OR rescue` system necessarily preserved this false-alarm floor.
- Therefore, an OR-only rescue system could never reach less than one unsuccessful start/hour.
- Permissive Level 1 found opportunities for 65 of 72 episodes, demonstrating that considerable precursor signal exists.
- However, Level 1 produced 1,132 unsuccessful starts and 22 unknown starts.
- Logistic, boosting, Platt-calibrated, and TCN verifiers could not separate useful precursors from false candidates at the required operating point.
- Published CNN and ResNet adaptations improved raw candidate recall but produced unacceptable false-alarm rates.
- Calibrated models generally became disabled rather than finding a useful operating point.
- The selected cascade had a maximum desktop latency of 374.86 milliseconds.
- Preserving every baseline alert was the wrong architectural constraint.

Do not repeat that design.

## The only valid guarantee

Implement a strict champion–challenger system:

- Freeze all current champion artifacts and results.
- Never overwrite them.
- A challenger becomes the new champion only if it passes every mandatory promotion criterion on the same fixed evaluation.
- If it fails any criterion, the current champion remains the official best system.
- Do not repeatedly tune on outer-fold results until something wins.
- Do not describe training, calibration, window-level, or fixed-threshold improvements as final improvement.

This guarantees non-regression in model selection. It does not guarantee that a superior biological signal exists in the available data.

## Mandatory definition of “better”

A new system is better only if it simultaneously achieves, on the corrected nested participant-held-out evaluation:

- More than 15 of 72 valid early episode warnings.
- Participant-macro sensitivity greater than 0.12943.
- Additional detections in at least two participants.
- Fewer than 1 unsuccessful final alert start per known monitored hour.
- Every credited warning begins 2–5 seconds before onset.
- Zero unknown starts, or no more unknown starts than the matched comparison if denominators change for a documented reason.
- No denominator, exposure, or episode-definition changes.
- One-second final alert pulses.
- Maximum streaming computation below 250 milliseconds, with lower median and p95 also reported.
- Complete reload and streaming parity.
- No future-data, subject-split, reset, or scoring leakage.

The target is not satisfied by a zero-alert or disabled system.

## Architectural correction

Do not preserve every v2 alert.

Use v2 and all published-model scores as inputs to a new verifier, but allow the verifier to reject both v2 and rescue candidates.

The final system must be:

```text
Causal sensor history
        |
        v
High-recall candidate ensemble
        |
        v
Causal event-level verifier
        |
        v
Budget-aware alert controller
        |
        v
One-second final alert
```

The verifier—not an OR operation—makes the final decision.

Retain a separate untouched v2 replay only as the comparison champion.

## Why the previous verifier failed

The previous verifier treated candidates too much like independent classification examples. A successful warning is instead a rare temporal event constrained by:

- its position within a recording;
- preceding gait behavior;
- persistence and change in risk;
- competition with many false candidate peaks;
- a limited alert budget per monitored hour;
- and a future onset 2–5 seconds away.

Replace pointwise candidate classification with direct event-level learning and causal hazard ranking.

## Model the actual question

At every current timestamp \(t\), estimate separate causal hazards for:

- onset in 0–2 seconds: late and not alertable;
- onset in 2–3 seconds;
- onset in 3–4 seconds;
- onset in 4–5 seconds;
- no onset within 5 seconds.

The final alert score is the combined probability of the 2–5-second bins.

This forces the model to distinguish a valid early warning from a late FoG detector.

Use only samples available at or before \(t\).

## High-recall candidate layer

Reuse the completed Level 1 ensemble rather than fitting another collection of nearly identical classifiers.

Its 65/72 opportunity coverage is sufficient for the next experiment.

Freeze its component scores and candidate-generation procedure. Analyze the seven uncovered episodes separately, but do not lower the candidate threshold until nearly every window becomes positive.

The primary research problem is now eliminating false candidates while retaining true opportunities.

## New event-level verifier

Build a compact causal sequence model that examines 10–20 seconds of observed history ending at the current timestamp.

Compare only these prespecified verifier families:

1. A causal dilated TCN.
2. A compact state-space or gated recurrent model.
3. A causal temporal refinement model adapted from credible published FoG segmentation/refinement code.

No bidirectional layers, centered convolutions, backward refinement, or future context are allowed.

The verifier receives:

- Right-ankle raw acceleration and gyroscope history.
- Existing v2 head scores.
- Published-model component scores.
- Level 1 candidate state.
- Score slopes, curvature, persistence, and cross-model agreement.
- Past-only periodicity and loss-of-periodicity features.
- Gait-cycle autocorrelation.
- Freeze-index-style frequency ratios computed from trailing samples.
- Rotation-invariant magnitudes plus directional channels.
- Motion-energy changes over 0.5, 1, 2, 5, 10, and 20 seconds.
- Time since the previous candidate and final alert.
- Reset and sensor-validity state.

No subject, session, task, activity, current FoG, or future label may enter inference.

## Multi-task context learning

Use training-only annotations to teach the encoder why false candidates occur.

Add auxiliary training heads for:

- current walking versus non-walking motion;
- turning;
- voluntary stopping;
- active FoG;
- post-FoG recovery;
- gait periodicity disruption.

These labels may shape the shared representation during training, but they must not be supplied as inference inputs.

The primary output remains the 2–5-second future-onset hazard.

This is intended to distinguish pre-FoG changes from normal turning, stopping, and already-active FoG—the major sources of false alerts.

## Multi-sensor teacher, single-ankle student

FoG-STAR contains additional sensor positions. Use them only in a training-time teacher model.

Prespecify two systems:

- Teacher: all available non-missing IMU locations.
- Student: right-ankle acceleration and gyroscope only.

Train the student using:

- true future-hazard labels;
- teacher soft targets;
- intermediate-representation distillation where appropriate.

At deployment and evaluation, the student must use only the right ankle.

This tests whether richer training-time context can teach the right-ankle model to reject ambiguous movements without changing the intended wearable input.

Missing sensors, reserved people, or test-participant statistics must not enter teacher training.

## Direct event-level objective

Do not optimize ordinary window accuracy as the main objective.

Use a combined loss containing:

1. Future-hazard classification loss.
2. Positive-episode multiple-instance loss: at least one high score should occur inside each 2–5-second warning interval.
3. Hard-negative top-k loss: the highest false peaks per known monitored hour receive the strongest penalty.
4. Temporal ranking loss: valid precursor candidates should rank above false candidates from the same participant or recording.
5. Late-warning penalty for 0–2-second predictions.
6. Active-FoG and post-FoG false-alert penalties.
7. Sparse-alert penalty based on candidate starts, not positive windows.
8. Auxiliary context-task losses.
9. Teacher–student distillation loss for the single-ankle student.

All loss weights must be fixed using inner development only.

## Budget-aware alert controller

The controller must be trained or calibrated for complete alert starts—not window classifications.

It may use:

- causal hysteresis;
- risk slope;
- one-step confirmation;
- refractory state;
- a causal rolling risk baseline;
- and a limited alert budget.

It must not use labels or future information.

A one-step confirmation is allowed only if the actual final alert still begins at least two seconds before onset. Lead time is measured from the physical final output start.

Do not mechanically impose a one-hour cooldown merely to satisfy the false-alarm rate. That would destroy episode sensitivity and is not a valid solution.

## Personalized causal calibration

Cross-person score calibration is a documented weakness.

Test one prespecified causal calibration method:

- Use the first fixed-duration, chronological setup period.
- Select it without inspecting FoG labels.
- Compute robust sensor and feature baselines using only that completed setup period.
- Do not issue calibrated alerts during setup.
- Do not choose a replacement setup interval if the first one is inconvenient.
- Keep all setup-period onsets in the all-onset denominator as unwarned.
- Compare fixed population normalization against this predetermined causal calibration.

Do not use whole-recording or future-person statistics.

## External code

Do not rerun the same published CNN and ResNet models as standalone pointwise classifiers; that question has been answered.

Search credible primary papers and open-source repositories specifically for:

- causal temporal refinement;
- event segmentation;
- discrete-time survival or hazard models;
- multiple-instance temporal learning;
- false-positive reduction;
- teacher–student IMU distillation;
- subject-invariant representation learning;
- and alert-rate-constrained calibration.

For every source:

- record the paper, repository, exact commit, and license;
- inspect it before execution;
- use code only when its license permits;
- retain attribution;
- identify and remove noncausal operations;
- explain its original task and every adaptation;
- never present borrowed detection results as early-warning evidence.

## Development data authorization

Use the already created and verified participants-1–18 development export.

Do not reopen the combined file unless required to verify a stored hash.

Do not access participants 19–22.

Do not run the locked assessment.

## Nested evaluation

Use the exact corrected physical-session, subject-grouped nested folds from the failed cascade so results remain comparable.

For each outer fold:

1. Exclude outer participants from all training, pretraining, normalization, augmentation fitting, calibration, threshold selection, distillation, and hard-negative mining.
2. Fit all preprocessing on training participants only.
3. Use inner participants for checkpoint, model-family, loss-weight, and alert-policy selection.
4. Freeze the challenger.
5. Evaluate the outer fold once.
6. Preserve failed folds and disabled policies.
7. Never revise the experiment after viewing outer results.

All arms must use identical onsets, exposure, missing-data rules, physical-session boundaries, and event matching.

## Feasibility analysis before training

First determine whether existing Level 1 candidates contain enough separability to support the target.

Calculate, using development labels for diagnosis only:

- the oracle maximum episode coverage possible from Level 1;
- the minimum false-start rate achievable while retaining each number of detected episodes;
- per-person true-candidate versus false-candidate score distributions;
- whether true opportunities rank above false candidates within recordings;
- false-candidate categories;
- and the amount of known negative exposure.

This oracle analysis must never be used as the live decision rule.

If even an oracle operating on existing candidate features cannot exceed 15/72 at less than one unsuccessful start/hour, document that new raw information or additional sensors are necessary. Continue with the multi-sensor-teacher experiment because it introduces new training information.

## Bounded experiment matrix

Freeze this matrix before training:

- Causal TCN verifier with event-level loss.
- Recurrent/state-space verifier with event-level loss.
- Best verifier plus auxiliary context heads.
- Best verifier plus multi-sensor teacher distillation.
- Best verifier plus predetermined wearer calibration.
- Required ablations:
  - without event-level loss;
  - without hard-negative top-k penalty;
  - without context tasks;
  - without distillation;
  - without personalized calibration.

Do not add more arms after outer results.

## Testing

Add tests proving:

- Every convolution and sequence feature is left-causal.
- No teacher-only sensor enters student inference.
- No activity or FoG label enters inference.
- Outer participants are absent from every fitting ledger.
- The 0–2, 2–3, 3–4, and 4–5-second targets are correct at exact boundaries.
- Alert starts—not positive windows—determine false-alarm rate.
- Unknown tails are never labeled negative.
- Physical sessions never merge.
- Missing, reversed, repeated, or nonfinite samples reset state.
- Actual final output time determines lead.
- Saved-model reload reproduces outputs.
- Streaming and batch implementations agree.
- Every reported comparison uses identical denominators.

## Performance and latency engineering

Profile the complete live student system.

Optimize implementation only after the model and scoring policy are frozen.

Required timing report:

- setup/compilation time separately;
- median live inference;
- p95;
- maximum;
- scheduling delay;
- compute-adjusted lead;
- unmeasured transport and actuator delay.

The promoted model’s maximum desktop computation must be below 250 milliseconds.

## Required outcome

Create a new unique run directory and save:

- preregistration;
- oracle feasibility analysis;
- exact folds;
- source and license audit;
- configurations;
- fitting ledgers;
- training histories;
- candidate scores;
- hazard predictions;
- final alert replays;
- per-person and per-episode results;
- false-start exposure;
- latency;
- reload parity;
- every failed arm;
- and promotion-gate results.

The final report must begin with exactly one of:

- “The challenger strictly beat the current champion under every mandatory criterion.”
- “The challenger improved some results but did not strictly beat the champion.”
- “The available development evidence did not produce a better system.”

Do not use the first statement unless all mandatory criteria pass.

## Execution instruction

Proceed through implementation, training, nested evaluation, verification, and reporting. Do not stop after a preflight report or successful tests.

However, do not keep inventing experiments after viewing outer results. A failure of this frozen study is a valid scientific result, not permission to overfit.

The current champion remains official unless the challenger produces more than 15/72 valid early warnings, improves participant-macro sensitivity, and simultaneously reduces unsuccessful starts below one per known monitored hour with a genuine ≥2-second output lead.