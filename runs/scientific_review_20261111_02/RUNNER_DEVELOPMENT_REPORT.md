# Reserved-safe runner development — 2026-11-11

The assessment runner and annotation inventory are implemented, tested, and frozen in review bundle 02. No candidate weights, normalization, thresholds, inference behavior or scoring targets were changed. Reserved participants and raw datasets were not opened. This work improves assessment integrity; it does not improve or revalidate model coverage.

## What changed
- locked_assessment.py requires a separate exact-bundle human approval and execution_ready=true. Neither exists for this review bundle. It checks frozen-file hashes and its own location before creating an exclusive bundle-level start marker, then checks an approved participant-only CSV checksum before reading it. It rejects mixed/unsupported partitions. Source imports must resolve to the frozen runtime. Failures retain the marker and write an error ledger; changing output does not allow retry.
- assessment_core.py inventories all original annotation transitions before splitting sensor-valid chunks. Missing sensors cannot delete onsets from the primary denominator. No-FoG participants and chunks too short for prediction remain represented. Timestamp gaps mark results incomplete. Label fields are used for scoring only, never passed to the predictor. Separate chunk pulse accounting prevents reset-separated alerts being counted as one long pulse.
- The runner reports all-onset, observed-contiguous and eligible denominators, participant outcomes/macro sensitivity, monitored/valid/recording durations, known target failures and separate unknown-alert status. Crash/corrupt partition errors are incomplete results, not exclusions that improve performance.

## Verification
26 tests passed: 13 runner/inventory tests, seven saved-audit tests and six comparison-gate tests. Exact command: `python -m unittest test_locked_assessment test_development_evidence_review test_candidate_gate`.
Synthetic cases cover missing sensors at onset, zero valid sensor input, no-FoG participants, annotation-independent alert generation, timestamp gaps, incompatible rates/scope, chunk pulse boundaries, old/new true/late/unknown matching parity, denied approval, frozen-file changes and failure/retry locks. All synthetic partition/checksum fixtures were generated in temporary directories; no real reserved file was used.
A fresh process imported frozen runtime sources, verified candidate artifacts and reproduced 2,622 saved-score alarm decisions. The result is in frozen_development_check.json. It is saved-score parity, not a new native-input model replay. Existing archived native parity remains historical evidence for unchanged model/runtime sources.

## Readiness and remaining gates
Bundle is intentionally NOT execution-ready. No approval exists, no reserved-data start marker exists, and approved_partition_sha256 is null. An owner-reviewed partition containing exactly FoG-STAR 19–22 and its checksum must be established under an explicit final-review decision before any such raw data are opened. The approved executable assessment would use `runtime/locked_assessment.py --bundle BUNDLE --approval REVIEW_JSON --data APPROVED_PARTITION --output NEW_UNIQUE_RUN`; do not execute it now. Scope approval is not created by the software or inferred from the instruction to build the runner.
Stanford 6–7 remain excluded because no separately frozen 128-Hz candidate/adapter exists. This bundle does not authorize an assessment, and any eventual evaluation is a limited locked assessment rather than clinical validation.

## Safe next development step
Check calibration feasibility only with development-only exports or saved development features. Establish whether a predetermined initial walking setup can be confirmed without label-based recording selection. Do not use the old combined-CSV loader during the embargo and do not launch a broad model search. The current verified v2 remains 7/16 on reused development data, with nine missed episodes.
