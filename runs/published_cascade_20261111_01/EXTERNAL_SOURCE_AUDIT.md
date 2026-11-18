# External source audit

The architecture/model code and relevant preprocessing interfaces used in the admission decisions were inspected as text. Other downloaded files remain reference material; no downloaded scripts or preprocessing were executed. External weights and external raw datasets were not used. Downloaded source is separated under external_sources/published_20261111_01 with read-only file attributes. Exact commits/file hashes are in external_source_license_audit.json. Working snapshots contain only this project’s adapted components and required notices.

## https://github.com/Lu1g1n0/FoG-detection-using-a-single-ankle-mounted-sensor

- commit: 2b2277eda53051142cc8198a8e0c0a3f61f2de0b
- paper: Borzì et al.2025, doi:10.1177/1877718X241302766
- license_decision: No repository license; do not copy/import its code or weights. Independently implement published CNN topology.
- sensor: Left ankle gyro3, degrees/s
- rate: 60Hz
- window: 120samples,2s,75% overlap
- target: Concurrent FoG majority label, NOT a2s-ahead forecast
- adaptation: Trailing2s within shared3s input; right ankle6channels; causal padding; fit-only normalization; fresh weights;2-5s onset target. Reject random-window train/test split and weight provenance.

## https://github.com/stanfordnmbl/imu-fog-detection

- commit: e95687842801ca3565463486a50d562fff44d182
- paper: O’Day et al.2022, doi:10.1186/s12984-022-00992-x
- license_decision: BSD-3-Clause admitted; full notice retained.
- sensor: Configurable multisensor6axis IMUs; SI units in source
- rate: 128Hz acquisition,64Hz preprocessing
- window: 2s, configurable overlap
- target: Concurrent FoG window-mode label
- adaptation: Single right ankle60Hz; two16filter17-tap CNN layers; causal padding; preserve fit-only six-channel scales in g and degrees/s; no resampling, external data or external weights.

## https://github.com/CristianTuretta/FoG-STAR

- commit: 90c7e5feccff46aa03b3e58878c86714af738999
- paper: FoG-STAR dataset: doi:10.1038/s41597-026-06645-1
- license_decision: Data described as CC-BY4.0; repository API has no code license. No utility code/notebook copied or executed.
- sensor: Ankles/back/wrist; current study right ankle only
- rate: 60Hz
- window: Utilities are dataset-oriented, not an early-warning definition
- target: Sample FoG annotations/activity metadata
- adaptation: Use only authorized export; verify local timestamps in seconds by1/60 increments despite README ms header. Continuous physical sessions; task IDs never predictive inputs.

## https://github.com/janbrederecke/fog

- commit: 02c00b86d0d987b07a84902cdc41026d9eba8002
- paper: Brederecke2023 preprint, arxiv:2307.03475; based on published residual-network architecture
- license_decision: MIT admitted; full notice retained. Kaggle/preprint source, not an independent peer-reviewed early-warning result.
- sensor: Lower-back3axis acceleration
- rate: Challenge sources vary; no rate is specified by architecture itself
- window: Architecture input1000samples, sequence prediction
- target: Concurrent FoG event labels; competition name does not establish early prediction
- adaptation: Compact three residual blocks16/24/32channels,6axis right ankle,180samples60Hz. Replace symmetric convolutions with left-causal padding and time-mixing batch norm with per-timestep layer norm. No competition weights or time-reversal augmentation.

## https://github.com/saiketa/GaitFM

- commit: 825017ced7ac7aa713617fac17a0b2d5675e21b2
- paper: Cai and Hong2026 preprint arxiv:2611.01684; not established peer-reviewed evidence
- license_decision: No repository license: source/weights not copied or executed.
- sensor: Multisource wearable IMU, six-channel configurations
- rate: Pretraining launchers50Hz
- window: 250samples,5s;75% masking
- target: Masked reconstruction and downstream gait outcomes, not this early-warning target
- adaptation: Only masked-reconstruction principle retained; custom causal ResNet autoencoder,3s native60Hz, fitting participants only. Full-attention/backbone weights and outside corpora rejected for incompatible causality/provenance.

## https://github.com/yabufarha/ms-tcn

- commit: 33ed91c0c7576650a2367efc602553af4c5295b1
- paper: Abu Farha and Gall,CVPR2019; Yang et al.2024 doi:10.1186/s12984-024-01320-1 applies refinement to FoG detection
- license_decision: MIT plus Commons Clause; read-only inspection. No PyTorch source imported; independently written compact Keras mathematical topology; reference license retained.
- sensor: Original action features; Yang uses five6axis IMUs
- rate: Yang64Hz; this refiner consumes4Hz causal score/features
- window: Original variable-length full sequence; this study current+12previous ticks
- target: Concurrent segmentation refinement, not a future FoG predictor
- adaptation: All convolutions left-causal; two stages,16channels,dilations1/2/4; final current tick only. No backward refinement, centered smoothing or future-dependent segment cleanup.

Official Stanford documentation: https://mobilize.stanford.edu/software/deepfog/ (same BSD-licensed repository, no independent commit). Yang refinement paper: https://link.springer.com/article/10.1186/s12984-024-01320-1 (published detection method, architecture reference only). No detection score is imported as early-warning performance.
