## Subsequent Stanford raw-data addition

The complete 60-recording Stanford raw seven-person directory is now downloaded and checksum-verified (392,947,843 bytes including three documentation/source files). This is a newly admitted continuous raw-ankle development source, superseding the earlier no-new-raw-source status below. See ANKLE_AND_NEW_DATA_RESULTS.md for units, subject reservations, development inventory and unsuccessful pooled-model results. Publisher: https://github.com/stanfordnmbl/imu-fog-detection . Full hashes and pinned revision: external_downloads/stanford/download_manifest.json.

---

## Subsequent full turning archive download and auxiliary-only use

Downloaded `external_downloads/turning_IMU.zip`, 159,910,531 bytes. Publisher MD5 `3b3ed59f0043b99bddcebb771be7599b` matched. Source: https://doi.org/10.6084/m9.figshare.14984667 . The 128 Hz files supply acceleration, rotation and FoG flags. Seventy-one turning trials from 35 people produced 33,299 auxiliary training windows; duplicate CSV/TXT copies were not double-counted.

Because preprocessing includes noncausal filtering, this release was used only for auxiliary representation pretraining, never as causal evaluation evidence. All early-warning scoring stayed on raw FoG-STAR validation. Paired from-scratch versus pretrained models both failed the useful prediction/false-warning target; high auxiliary detection accuracy does not imply early prediction. Audit, inventory, checksums, checkpoints and exact source snapshots are saved in external_downloads and runs/pretraining_round_1. This limited pretraining use updates the earlier no-new-training-source status below; no new raw continuous causal evaluation source was admitted.

---

# Additional dataset search and download audit

Actual downloads: Twente participant sample (55,304,442 bytes, verified publisher MD5), Twente README and landing-page metadata, Pardoel release metadata and description, and the multimodal dataset author README. No downloaded Python code was executed.

- Twente: right-ankle accelerometer and gyroscope. Downloaded PD001_AllActivities_FOG.npz contains 9,096 windows of 120 samples and six channels. Its metadata identifies PD019, illustrating why filenames alone cannot establish participant splits. The release uses label-dependent overlap and lacks precise sample/end/onset timestamps in this sample. Filtering direction is unresolved. Not admitted to strict continuous prediction; arbitrary adjacent windows must not be treated as a recording. Source: https://doi.org/10.4121/40e06061-f441-43b5-9235-006829206509

- Pardoel: the released tables contain selected pressure and thigh features. Prediction labels include some windows overlapping actual FoG; they do not establish >=2-second warning. Not admitted as raw ankle training. Source: https://doi.org/10.5683/SP2/UFZTLV

- Multimodal PD dataset: the authors state raw files lack the private label marks needed to reproduce annotated alignment. Public API access also returned HTTP 403. No raw files downloaded. Source: https://github.com/AccSrd/multimodal-Parkinson-data-processing

- MyGait footwear recordings: published column descriptions do not provide time-resolved FoG onset labels, and placement differs from ankle. Not downloaded for supervised early-warning training. Source: https://zenodo.org/records/15672744

- Kaggle TLVMC: lower-back placement, not ankle. Not pooled into an ankle deployment claim. Source: https://www.kaggle.com/competitions/tlvmc-parkinsons-freezing-gait-prediction/data

No new release audited in this round has yet supplied a verified, directly compatible continuous raw ankle + precise onset training set. Existing Daphnet and FoG-STAR remain the admitted labeled development sources. This does not prove no suitable dataset exists.

New model round: two causal residual temporal convolution networks and a recurrent GRU network, six-second trailing history; minimum lead stays two seconds. One TCN uses the old upper horizon of three seconds; the other TCN and GRU allow warnings up to five seconds early. The latter changes the acceptable warning interval, so compare their results with that distinction. Trials and splits were written before training in runs/temporal_round_1/planned_experiments.json.
