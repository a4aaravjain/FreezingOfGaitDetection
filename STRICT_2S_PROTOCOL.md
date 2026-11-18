# Strict two-second prediction protocol (revision 4)

Actual project folder: `C:\Users\victo\Desktop\Aarav\Science Fair 2026-2027`.

A successful warning starts between 2 and 3 seconds before the next annotated onset. A warning less than 2 seconds ahead is late, receives no detection credit, and counts as an unsuccessful warning. Normal samples in that late interval are negative training examples. Active-FoG samples remain excluded from binary loss but included in continuous alarm replay. Existing revision-3 reports used a different target and must not be presented as two-second prediction results.

The nominal research target is at least 90% of all observed episodes warned in this interval and at most one unsuccessful warning per hour. Duration and occupancy guards also apply. Disabled policies fail sensitivity and are not successful low-false-positive models. This research target is not a clinically validated safety criterion.

All timing is measured from sensor-window end to annotated onset. Compute, communication and actuator delay must be measured on the eventual device and subtracted before claiming usable lead time. Annotation uncertainty also matters, particularly near the exact two-second boundary. The models have not demonstrated real-world clinical usefulness.

## Bounded experiments

`strict_baselines.py` compares two prespecified models, standardized logistic regression (C=1) and histogram gradient boosting (100 iterations, 15 leaves), using trailing one- and three-second accelerometer features. No future filtering or label-derived features are used. Scaling and fitting use training subjects only. Thresholds use the existing fixed validation grid, now subject to one unsuccessful warning per hour. No repeated outer-test optimization is performed.

Daphnet uses S01-S06 for training and S07-S08 for validation. S09/S10 are not used in these experiments. All Daphnet people were historically exposed, so these are development results.

FoG-STAR is a separate experiment using the left ankle at native 60 Hz, with IDs 1-14 training, 15-18 validation and 19-22 reserved from fitting/model evaluation in this revision. The split is fixed before fitting. Historical use of external data prevents claiming these are pristine test subjects. Both axes and cohorts remain separate from Daphnet. Missing ankle samples and recording boundaries reset history; no interpolation or stitching is used. Event counts from valid continuous segments must be read alongside the inventory of all original annotations, including events whose onsets become unobservable across gaps.

The CNN experiment retains the existing architecture, training split and 18-epoch maximum, changing the prediction target and false-warning budget. Exact source, configuration and artifacts are stored with each run. Saved source snapshots reproduce older runs even if current adapter support has changed.

## External provenance

The local FoG-STAR sensor CSV exactly matches the publisher's v3 MD5 `952a37ab147da35e6d4e7a1e9bac44cb`; a duplicate download adds nothing. It has 329,027 rows, 22 people, 106 recordings and 101 normal-to-FoG onsets. Local timestamp increments are seconds at 60 Hz, despite the README calling them milliseconds. See [Zenodo v3](https://zenodo.org/records/17838806) and [dataset paper](https://www.nature.com/articles/s41597-026-06645-1). License: CC-BY-4.0. Missing-data and pre-onset-history inventory is saved under `runs/external_provenance`.

The local turning IMU files match the structure described by the [turning-in-place dataset paper](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.832463/full). The authors state the released accelerometer and gyroscope signals received offline zero-phase filtering. Such processing accesses future samples. Pending raw-signal provenance, these files are excluded from strict causal experiments. The source [Figshare release](https://doi.org/10.6084/m9.figshare.14984667) is documented by that paper; local files have not yet been checksum-matched to that release.

## Reproduce

Run from this actual project directory with Python 3.11:

```text
py -3.11 test_pipeline.py
py -3.11 audit_external.py
py -3.11 strict_baselines.py
py -3.11 strict_baselines.py --dataset fogstar
py -3.11 ankles.py develop --output runs/new_strict_cnn_run
```

Baseline run folders intentionally reject overwriting. Existing completed runs are retained. Previous code is archived under `legacy/before_strict_2s`.
