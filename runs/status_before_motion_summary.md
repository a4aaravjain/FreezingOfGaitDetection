> Newest comparison: [NON_NEURAL_RESULTS.md](NON_NEURAL_RESULTS.md). Extra Trees and a signal-change rule were also tested; neither met the target. Current regression suite: 37 passing tests.

# Latest research status

The requested reliable >=2-second early-warning predictor has NOT been achieved.

This round downloaded and verified a 55,304,442-byte Twente ankle sample, audited additional public releases, trained four additional neural models, and tested a compiled label-free streaming path. Combined with the previous round, nine models have been fitted. No model produced a validation policy satisfying the declared false-warning constraints and useful episode coverage.

## Measured results

The pooled orientation-independent GRU's highest-coverage diagnostic candidates warned 9/38 Daphnet validation episodes with 78.3 unsuccessful warnings/hour, and 2/16 original FoG-STAR validation episodes with 68.0 unsuccessful warnings/hour. These are different cohort-specific diagnostic thresholds, not a recommended joint operating point. The selected joint policy is disabled. Both intervals permit warnings 2-5 seconds before annotated onset; the two-second minimum is unchanged.

The three other new networks are documented in [TEMPORAL_RESULTS.md](TEMPORAL_RESULTS.md). The pooled model is documented in [POOLED_MOTION_RESULTS.md](POOLED_MOTION_RESULTS.md). All results are development results, not final generalization evidence.

## Verified implementation

- 36 regression checks passed, including target boundaries, late/duplicate warnings, gap resets, causal feature inputs, rotation invariance and native sampling rates.
- The four new fitted models passed reload checks. The three temporal models passed real-stream probability checks; the pooled streaming class passed probability, smoothing and alarm-state parity at 60 and 64 Hz.
- Compiled pooled inference measured about 6 ms median and under 9 ms maximum in the final ten-call checks per rate on this computer. Earlier eager calls took roughly 500 ms. These measurements exclude communications, sensor buffering and actuator delay and do not establish delivered warning time on target hardware.
- `motion_stream.py` provides the compiled research stream. It retains the saved disabled threshold; it is not a successful warning device.

## Data findings

[ADDITIONAL_DATA_AUDIT.md](ADDITIONAL_DATA_AUDIT.md) documents the downloaded files, checksums, source links and exclusions. The Twente sample lacks the timing needed for continuous early-warning scoring. The inspected public raw multimodal release lacks accessible alignment labels, and its API returned HTTP 403. No newly audited dataset was admitted as a compatible continuous raw-ankle prediction source this round.

FoG-STAR gaps and required history reduce coverage: its six-second-history ideal-score diagnostic warns 9/16 original validation episodes. This diagnostic uses future labels intentionally and is not a trained model. The Daphnet equivalent warns 38/38, showing the learned scores, rather than the replay logic alone, limit that experiment.

[RAW_DATA_REQUEST_DRAFT.md](RAW_DATA_REQUEST_DRAFT.md) is an unsent request for continuous raw recordings, onset timing and participant mapping. Additional suitable recordings and genuinely unseen evaluation people remain necessary. Repeated threshold tuning on these same validation people cannot establish near-perfect performance.

## Files and execution

Actual project folder: `C:\Users\victo\Desktop\Aarav\Science Fair 2026-2027`.

New architecture definitions: `temporal_models.py`. Reproducible experiment entry points: `temporal_experiments.py`, `pooled_motion.py`. Streaming implementation: `motion_stream.py`. Run `py -3.11 test_pipeline.py` for regression tests and `py -3.11 verify_pooled.py` to recheck saved pooled-model streaming behavior. Training scripts preserve existing run folders and refuse to overwrite them.

Artifacts: `runs/temporal_round_1`, `runs/pooled_motion_round_1`, `external_downloads`. Prior models, experiments and source snapshots are retained.

