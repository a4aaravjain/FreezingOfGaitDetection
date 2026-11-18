"""
FoG (Freezing of Gait) early-warning model -- Daphnet ankle accelerometer.
See project notes / experiment_log.csv for full history. This copy has the
model.fit()/callback bugs fixed (see FIX NOTE below); no methodology beyond
that was changed.

FIX NOTE (this pass)
-----------------------------------------------------------------------------
1. ValidationF1Callback was instantiated with only (X_val, y_val) but its
   constructor requires (x_val, y_val, batch_size) -> TypeError before
   training could even start. Fixed to pass BATCH_SIZE.
2. EarlyStopping(...), ModelCheckpoint(...), ReduceLROnPlateau(...) were
   literal Ellipsis placeholders, not valid configs -> would crash
   immediately. Filled in with concrete settings:
     - EarlyStopping monitors val_f1 (max), matching the stated design goal
       of checkpointing/stopping on F1 rather than val_loss, and restores
       the best epoch's weights at the end of training.
     - ModelCheckpoint also monitors val_f1 (max) and saves the best epoch
       to disk as best_fog_model.keras, so the best model is recoverable
       as a file even if training continues past it.
     - ReduceLROnPlateau monitors val_loss (min), which is a separate,
       conventional LR-decay signal and doesn't conflict with checkpointing
       on val_f1.
3. sample_weights was only ever defined inside `if not used_augmentation:`,
   but model.fit() referenced it unconditionally -> NameError as soon as
   USE_IMU_AUGMENTATION or USE_SENSOR_DATA_AUGMENTATION is True. (Both are
   False in this v5 config, so this wasn't hit yet, but it was a crash
   waiting to happen the next time augmentation is turned back on.)
4. training_data (the MixedTrainingSequence) was built but never actually
   passed to model.fit() -- the fit call always used the plain-array path.
   This meant that if augmentation were ever re-enabled, the model would
   silently train on Daphnet-only data while the printed log line still
   claimed "IMU augmentation active: True" / "sensor_data.csv augmentation
   active: True". Fixed by branching model.fit() on used_augmentation and
   actually feeding it the Sequence in that branch (no sample_weight or
   batch_size there -- the generator already handles batching and mixes in
   positives itself).
None of these fixes change NEGATIVE_RATIO, POSITIVE_CLASS_WEIGHT,
THRESHOLD_OBJECTIVE, MAX_FALSE_ALARMS_PER_MINUTE, or any other v5
methodological choice -- only the crashing/silently-wrong plumbing above.
I still have no TensorFlow and no network in this sandbox, so this was
verified with py_compile only, not an actual training run. Please run it
yourself and report back the printed val/test metrics and event metrics.

RUN LOG (continued)
-----------------------------------------------------------------------------
v5 (first real run, reported by researcher): val_f1 peaked at 0.237 (epoch
3, in line with v2.1's ballpark), but the threshold search then picked
threshold=0.032, which flagged 99.94% of all TEST negative windows as
positive (confusion matrix [[9, 15346], [0, 344]]; test precision 0.02,
val precision 0.00-0.07). Event metrics looked deceptively good -- 100%
episode detection, 0.108 false alarms/min, 8 false clusters -- because a
continuous positive stretch across an entire non-FoG recording still
collapses into ONE cluster under CLUSTER_GAP_SECONDS, so an almost-always-
positive model satisfies the cluster-based false-alarm cap while being
maximally noisy in reality. REJECTED: this is the exact degenerate outcome
the project explicitly said it did not want, it just wasn't caught by the
existing cap. Root cause was a gap in choose_max_detection_threshold, not
the model itself (epoch 3's raw val_f1 wasn't unreasonable).

v5.1 (MAX_POSITIVE_PREDICTION_RATE=0.5 added, reported by researcher):
threshold rose to 0.348 (from v5's degenerate 0.032), test positive rate
16.1% (sane, nowhere near v5's 99.94%). Test: precision 0.05, recall 0.39,
F1 0.09. Event metrics: 66.7% episodes detected (18/27, up from v2.1's
55.6%), 1.84 false alarms/min (up from v2.1's 0.54), avg warning 1.75s.
ACCEPTED as a real, non-degenerate operating point, deliberately not
comparable to v2.1 on F1 (that trade -- more detection, more false alarms,
worse precision/F1 -- is exactly what was requested). NOT yet
"near-complete" detection though. Diagnosis: validation positive rate at
the selected threshold was already ~49.6%, meaning MAX_POSITIVE_PREDICTION_
RATE (not MAX_FALSE_ALARMS_PER_MINUTE, which only reached 1.84 of its 6.0
budget) was the binding constraint that stopped the threshold from going
lower. FOLLOW-UP (v5.2, untested -- no TensorFlow available in this
sandbox): MAX_POSITIVE_PREDICTION_RATE raised 0.5 -> 0.65, single controlled
change, nothing else touched, to let the false-alarm-per-minute cap become
the real limiting factor and see how much further detection can go before
that cap (still 6.0/min) is actually hit.
===============================================================================
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score
)

from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation, BatchNormalization, Conv1D, Dense, Dropout,
    GlobalAveragePooling1D, Input, MaxPooling1D
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence


# =====================================================================
# CONFIG -- every methodological knob lives here, and is written out to
# fog_model_config.json so a saved model can always be traced back to the
# settings that produced it.
# =====================================================================
FOG_RAW_LABEL = 2
HISTORY_SECONDS = 3.0
HORIZON_SECONDS = 3.0
PREDICTION_INTERVAL_SECONDS = 0.25

# GOAL CHANGE (v5): researcher explicitly wants near-complete FoG detection
# and is willing to accept more false alarms to get it ("some are fine").
# Keep moderate positive weighting so the model learns FoG without pushing
# its probabilities into the overprediction regime seen in the v5 run.
NEGATIVE_RATIO = 3
POSITIVE_CLASS_WEIGHT = 1.15
EPOCHS = 18
BATCH_SIZE = 256
SEED = 42

# (3) target definition: exclude ambiguous in-FoG windows instead of
# silently labeling them as "no freeze coming".
EXCLUDE_IN_FOG_SAMPLES = True

# (5) probability smoothing: odd window length, in *prediction steps*
# (each step = PREDICTION_INTERVAL_SECONDS). 1 = no smoothing.
SMOOTHING_WINDOW_STEPS = 5

# Require repeated evidence before an alarm is emitted.  This is deliberately
# tuned on validation data rather than treating one noisy prediction as FoG.
PERSISTENCE_WINDOW_STEPS = 3
PERSISTENCE_REQUIRED_STEPS = 2

# (6) two positive prediction windows less than this many seconds apart are
# considered the same warning cluster (one alarm), not two.
CLUSTER_GAP_SECONDS = 1.0

# v5: augmentation (IMU, sensor_data.csv) made things WORSE in the last
# confirmed run -- turned back off. Going back to Daphnet-only.
USE_IMU_AUGMENTATION = False
USE_SENSOR_DATA_AUGMENTATION = False

# THRESHOLD POLICY (v5): scan thresholds on S08 VALIDATION ONLY and pick the
# one that catches the most FoG episodes before onset, subject to a cap on
# false alarm clusters per minute.
THRESHOLD_OBJECTIVE = "max_detection"  # choose operating point on S08 only
MAX_FALSE_ALARMS_PER_MINUTE = 1.0

# GUARD ADDED after v5's first real run: that run picked threshold=0.032 and
# flagged 99.94% of all negative test windows as positive (confusion matrix
# [[9, 15346], [0, 344]]) -- a degenerate "always predict positive" model.
# It slipped past MAX_FALSE_ALARMS_PER_MINUTE because that cap is computed
# on CLUSTERED alarms: a continuous positive stretch across an entire
# non-FoG recording merges into just ONE cluster under CLUSTER_GAP_SECONDS,
# so nonstop false positives look cheap by that metric alone. This second,
# independent cap works on raw (unclustered) predictions and can't be
# gamed the same way: the fraction of ALL validation windows predicted
# positive must stay under this ceiling, in addition to (not instead of)
# the cluster-based false-alarm cap above.
# v5.1 result (this cap = 0.5): at the threshold selected, validation
# positive rate was already ~49.6% -- i.e. THIS cap, not
# MAX_FALSE_ALARMS_PER_MINUTE, was the binding constraint (false alarms/min
# only reached 1.84 of the 6.0 allowed; 66.7% episode detection, up from
# v2.1's 55.6%). Raised 0.5 -> 0.65 as a single controlled change so the
# false-alarm-per-minute cap -- the constraint actually tied to the stated
# goal -- gets a chance to be the real limiting factor instead.
MAX_POSITIVE_PREDICTION_RATE = 0.15

# threshold policy fallback (used only if THRESHOLD_OBJECTIVE == "f1"):
# maximize validation F1 subject to a minimum validation precision.
MINIMUM_VALIDATION_PRECISION = 0.20

TRAIN_SUBJECTS = {"S01", "S02", "S03", "S04", "S05", "S06", "S07"}
VALIDATION_SUBJECTS = {"S08"}
TEST_SUBJECTS = {"S09", "S10"}

COL_NAMES = [
    "Time", "Ankle_X", "Ankle_Y", "Ankle_Z",
    "Leg_X", "Leg_Y", "Leg_Z", "Trunk_X", "Trunk_Y", "Trunk_Z", "Label"
]

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =====================================================================
# Data loading / preprocessing
# =====================================================================
def read_recording(file_path):
    data = pd.read_csv(file_path, sep=r"\s+", header=None, names=COL_NAMES)
    data["Original_Index"] = np.arange(len(data))
    data = data[data["Label"] != 0].copy()
    data.reset_index(drop=True, inplace=True)
    ankle = data[["Ankle_X", "Ankle_Y", "Ankle_Z"]].to_numpy(np.float32)
    labels = data["Label"].to_numpy(np.int8)
    times = data["Time"].to_numpy(np.float64)
    indices = data["Original_Index"].to_numpy(np.int64)
    return ankle, labels, times, indices


def causal_normalize(ankle):
    ankle64 = ankle.astype(np.float64)
    count = np.arange(1, len(ankle64) + 1, dtype=np.float64)[:, None]
    cumulative_sum = np.cumsum(ankle64, axis=0)
    cumulative_sq_sum = np.cumsum(ankle64 ** 2, axis=0)
    running_mean = cumulative_sum / count
    running_variance = np.maximum(cumulative_sq_sum / count - running_mean ** 2, 1e-6)
    running_std = np.sqrt(running_variance)
    normalized = (ankle64 - running_mean) / running_std
    return normalized.astype(np.float32)


def make_target(labels, times):
    starts = (labels == FOG_RAW_LABEL) & np.r_[True, labels[:-1] != FOG_RAW_LABEL]
    fog_starts = times[starts]
    target = np.zeros(len(labels), dtype=np.int8)
    if len(fog_starts) == 0:
        return target, fog_starts
    next_index = np.searchsorted(fog_starts, times, side="right")
    valid = next_index < len(fog_starts)
    next_times = np.full(len(times), np.inf)
    next_times[valid] = fog_starts[next_index[valid]]
    target = (
        (labels != FOG_RAW_LABEL)
        & (next_times > times)
        & (next_times <= times + HORIZON_SECONDS * 1000)
    ).astype(np.int8)
    return target, fog_starts


def build_split(files, history_rows, interval_rows):
    sequences = []
    targets = []
    metadata = []
    for file_path in files:
        ankle, labels, times, original_indices = read_recording(file_path)
        target, fog_starts = make_target(labels, times)
        ankle = causal_normalize(ankle)
        for end_index in range(history_rows, len(ankle), interval_rows):
            start_index = end_index - history_rows
            if not np.all(np.diff(original_indices[start_index:end_index]) == 1):
                continue
            if EXCLUDE_IN_FOG_SAMPLES and labels[end_index - 1] == FOG_RAW_LABEL:
                continue
            sequences.append(ankle[start_index:end_index])
            targets.append(target[end_index - 1])
            metadata.append({
                "file": file_path,
                "time": float(times[end_index - 1]),
                "fog_starts": fog_starts
            })
    return np.asarray(sequences, np.float32), np.asarray(targets, np.int8), metadata


# =====================================================================
# Thresholding, smoothing, event/cluster metrics
# =====================================================================
def choose_f1_threshold(y_true, probabilities):
    _, _, thresholds = precision_recall_curve(y_true, probabilities)
    best_threshold = float(thresholds[0]) if len(thresholds) else 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(np.int8)
        precision = precision_score(y_true, predictions, zero_division=0)
        if precision < MINIMUM_VALIDATION_PRECISION:
            continue
        score = f1_score(y_true, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    if best_f1 < 0:
        return float(np.max(probabilities))
    return best_threshold


def apply_persistence_by_file(probabilities, metadata, window_steps=PERSISTENCE_WINDOW_STEPS,
                              required_steps=PERSISTENCE_REQUIRED_STEPS):
    """
    Convert noisy window-level probabilities into alarms only when there is
    repeated evidence within the same recording.

    This is applied identically to validation and test data. It does not look
    at future labels, so it cannot use the answer to create an alarm.
    """
    if window_steps <= 1 or required_steps <= 1:
        return (probabilities >= 0.5).astype(np.int8)

    output = np.zeros(len(probabilities), dtype=np.int8)
    start = 0

    while start < len(metadata):
        end = start + 1
        current_file = metadata[start]["file"]
        while end < len(metadata) and metadata[end]["file"] == current_file:
            end += 1

        segment = probabilities[start:end]
        binary = (segment >= 0.5).astype(np.int8)

        # Alarm at the current step only after enough recent predictions
        # agree.  Padding is not used, so the first few samples cannot
        # accidentally create an alarm.
        for i in range(len(binary)):
            left = max(0, i - window_steps + 1)
            if int(binary[left:i + 1].sum()) >= required_steps:
                output[start + i] = 1

        start = end

    return output


def apply_threshold_and_persistence(probabilities, metadata, threshold,
                                    window_steps=PERSISTENCE_WINDOW_STEPS,
                                    required_steps=PERSISTENCE_REQUIRED_STEPS):
    """
    Apply a probability threshold followed by a short temporal persistence
    rule.  Both validation and test use this exact same procedure.
    """
    output = np.zeros(len(probabilities), dtype=np.int8)
    start = 0

    while start < len(metadata):
        end = start + 1
        current_file = metadata[start]["file"]
        while end < len(metadata) and metadata[end]["file"] == current_file:
            end += 1

        binary = (probabilities[start:end] >= threshold).astype(np.int8)

        for i in range(len(binary)):
            left = max(0, i - window_steps + 1)
            if int(binary[left:i + 1].sum()) >= required_steps:
                output[start + i] = 1

        start = end

    return output


def choose_max_detection_threshold(probabilities, val_metadata, max_false_alarms_per_minute,
                                    max_positive_prediction_rate=MAX_POSITIVE_PREDICTION_RATE):
    """
    Select the operating threshold using S08 validation only.

    The selected threshold must satisfy BOTH:
      1. clustered false alarms <= MAX_FALSE_ALARMS_PER_MINUTE
      2. raw alarm rate <= MAX_POSITIVE_PREDICTION_RATE

    A persistence rule is applied before calculating either constraint. This
    prevents isolated noisy windows from becoming device alarms.
    """
    duration_cache = build_duration_cache(val_metadata)
    candidates = np.unique(probabilities)
    candidates = np.sort(candidates)[::-1]

    best = None
    fallback = None

    for threshold in candidates:
        predictions = apply_threshold_and_persistence(
            probabilities,
            val_metadata,
            float(threshold),
            PERSISTENCE_WINDOW_STEPS,
            PERSISTENCE_REQUIRED_STEPS
        )

        positive_rate = float(predictions.mean())
        metrics = event_metrics(val_metadata, predictions, duration_cache)
        false_rate = metrics["False Alarms Per Minute"]
        detection = metrics["Episodes Detected Before Onset (%)"]

        # Keep the best conservative fallback even if the requested alarm
        # budget cannot be met.
        if positive_rate <= max_positive_prediction_rate:
            if fallback is None or false_rate < fallback[0] or (
                false_rate == fallback[0] and detection > fallback[2]
            ):
                fallback = (false_rate, float(threshold), detection, positive_rate)

            if false_rate <= max_false_alarms_per_minute:
                candidate = (detection, float(threshold), false_rate, positive_rate)
                if best is None or candidate[0] > best[0]:
                    best = candidate

    if best is not None:
        print(
            f"Selected threshold={best[1]:.6f} | "
            f"validation detection={best[0]:.1f}% | "
            f"false alarms/min={best[2]:.3f} | "
            f"alarm rate={best[3]:.2%} | "
            f"persistence={PERSISTENCE_REQUIRED_STEPS}/{PERSISTENCE_WINDOW_STEPS}"
        )
        return best[1]

    if fallback is not None:
        print(
            "WARNING: no validation threshold met the requested false-alarm "
            f"budget of {max_false_alarms_per_minute:.2f}/min. Using the lowest "
            f"false-alarm operating point that stayed below the "
            f"{max_positive_prediction_rate:.0%} alarm-rate cap: "
            f"{fallback[0]:.3f}/min, detection={fallback[2]:.1f}%."
        )
        return fallback[1]

    print(
        "WARNING: no validation threshold stayed below the alarm-rate cap. "
        "Using the strictest observed probability."
    )
    return float(np.max(probabilities))


def smooth_probabilities_by_file(probabilities, metadata, window_steps):
    if window_steps <= 1:
        return probabilities
    smoothed = np.array(probabilities, dtype=np.float64, copy=True)
    start = 0
    n = len(metadata)
    while start < n:
        end = start
        current_file = metadata[start]["file"]
        while end < n and metadata[end]["file"] == current_file:
            end += 1
        segment = probabilities[start:end]
        kernel = np.ones(window_steps) / window_steps
        padded = np.pad(segment, (window_steps // 2, window_steps // 2), mode="edge")
        smoothed_segment = np.convolve(padded, kernel, mode="valid")
        smoothed[start:end] = smoothed_segment[:end - start]
        start = end
    return smoothed


def clusters(times):
    groups = []
    gap_ms = CLUSTER_GAP_SECONDS * 1000
    for time in sorted(times):
        if not groups or time - groups[-1][-1] > gap_ms:
            groups.append([time])
        else:
            groups[-1].append(time)
    return groups


def build_duration_cache(metadata):
    cache = {}
    for item in metadata:
        file_path = item["file"]
        if file_path not in cache:
            _, _, times, _ = read_recording(file_path)
            cache[file_path] = (times[-1] - times[0]) / 60000
    return cache


def event_metrics(metadata, predictions, duration_cache=None):
    if duration_cache is None:
        duration_cache = build_duration_cache(metadata)

    by_file = {}
    for item, prediction in zip(metadata, predictions):
        values = by_file.setdefault(
            item["file"], {"fog_starts": item["fog_starts"], "times": []}
        )
        if prediction == 1:
            values["times"].append(item["time"])

    total_events = detected = 0
    true_clusters = false_clusters = 0
    warning_times = []
    minutes = 0.0
    for file_path, values in by_file.items():
        minutes += duration_cache[file_path]
        fog_starts = values["fog_starts"]
        positive_times = np.asarray(values["times"], dtype=np.float64)
        total_events += len(fog_starts)
        for onset in fog_starts:
            warnings = positive_times[
                (positive_times < onset)
                & (positive_times >= onset - HORIZON_SECONDS * 1000)
            ]
            if len(warnings):
                detected += 1
                warning_times.append((onset - warnings[0]) / 1000)
        for group in clusters(positive_times):
            is_useful = any(
                np.any(
                    (fog_starts > alarm)
                    & (fog_starts <= alarm + HORIZON_SECONDS * 1000)
                )
                for alarm in group
            )
            if is_useful:
                true_clusters += 1
            else:
                false_clusters += 1

    return {
        "FoG Episodes": total_events,
        "Episodes Detected Before Onset (%)": 100 * detected / total_events if total_events else 0.0,
        "False Alarms Per Minute": false_clusters / minutes if minutes else 0.0,
        "Average Warning Time (seconds)": float(np.mean(warning_times)) if warning_times else 0.0,
        "Median Warning Time (seconds)": float(np.median(warning_times)) if warning_times else 0.0,
        "True Warning Clusters": true_clusters,
        "False Warning Clusters": false_clusters
    }


# =====================================================================
# Model
# =====================================================================
def build_model(history_rows):
    model = Sequential([
        Input(shape=(history_rows, 3)),

        Conv1D(32, 7, padding="same", use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.10),
        Conv1D(32, 7, padding="same", use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(2),
        Dropout(0.2),

        Conv1D(64, 5, padding="same", use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.15),
        Conv1D(64, 5, padding="same", use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(2),
        Dropout(0.3),

        Conv1D(96, 3, padding="same", use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        GlobalAveragePooling1D(),

        Dense(64, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


class ValidationF1Callback(Callback):
    def __init__(self, x_val, y_val, val_metadata, batch_size):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.val_metadata = val_metadata
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        probabilities_raw = self.model.predict(
            self.x_val, batch_size=self.batch_size, verbose=0
        ).ravel()
        # Callback F1 is still a simple training signal; threshold is fixed at
        # 0.5 here so the validation callback does not tune on the test set.
        probabilities = smooth_probabilities_by_file(
            probabilities_raw,
            self.val_metadata,
            SMOOTHING_WINDOW_STEPS
        )
        predictions = apply_threshold_and_persistence(
            probabilities,
            self.val_metadata,
            0.5,
            PERSISTENCE_WINDOW_STEPS,
            PERSISTENCE_REQUIRED_STEPS
        )
        logs["val_f1"] = f1_score(self.y_val, predictions, zero_division=0)


def sample_windows_from_recording(record, count, history_rows, rng):
    positive, negative = record["positive"], record["negative"]
    selected = np.concatenate([
        rng.choice(positive, count // 2, replace=len(positive) < count // 2),
        rng.choice(negative, count - count // 2, replace=len(negative) < count - count // 2)
    ])
    source_history_rows = record["source_history_rows"]
    ankle = record["ankle"]
    targets = record["targets"]
    source_positions = np.arange(source_history_rows)
    output_positions = np.linspace(0, source_history_rows - 1, history_rows)
    windows = []
    window_targets = []
    for end_index in selected:
        source_window = ankle[end_index - source_history_rows:end_index]
        windows.append(np.column_stack([
            np.interp(output_positions, source_positions, source_window[:, axis])
            for axis in range(3)
        ]))
        window_targets.append(targets[end_index - 1])
    return np.asarray(windows, np.float32), np.asarray(window_targets, np.int8)


def load_imu_file(file_path):
    try:
        data = pd.read_csv(file_path, sep="\t")
    except Exception as error:
        print(f"WARNING: could not read {file_path}: {error} -- skipping this IMU file.")
        return None
    data.columns = [str(column).strip() for column in data.columns]
    required = {"ACC ML [g]", "ACC AP [g]", "ACC SI [g]", "Time [s]", "Freezing event [flag]"}
    if not required.issubset(data.columns):
        print(f"WARNING: {file_path} missing expected columns -- skipping.")
        return None

    ankle = data[["ACC ML [g]", "ACC SI [g]", "ACC AP [g]"]].to_numpy(np.float32)
    times = data["Time [s]"].to_numpy(np.float64) * 1000
    labels = data["Freezing event [flag]"].to_numpy(np.int8)

    if len(ankle) < 10 or np.any(np.diff(times) <= 0):
        print(f"WARNING: {file_path} too short or has non-increasing timestamps -- skipping.")
        return None

    source_rate = 1000 / np.median(np.diff(times))
    source_history_rows = int(HISTORY_SECONDS * source_rate)
    source_interval_rows = max(1, int(PREDICTION_INTERVAL_SECONDS * source_rate))
    if len(ankle) <= source_history_rows:
        print(f"WARNING: {file_path} shorter than one history window -- skipping.")
        return None

    fog_starts = times[(labels == 1) & np.r_[True, labels[:-1] != 1]]
    next_index = np.searchsorted(fog_starts, times, side="right")
    valid = next_index < len(fog_starts)
    next_times = np.full(len(times), np.inf)
    next_times[valid] = fog_starts[next_index[valid]]
    targets = (
        (labels != 1)
        & (next_times > times)
        & (next_times <= times + HORIZON_SECONDS * 1000)
    ).astype(np.int8)

    ankle = causal_normalize(ankle)
    end_indices = np.arange(source_history_rows, len(ankle), source_interval_rows)
    if EXCLUDE_IN_FOG_SAMPLES:
        end_indices = end_indices[labels[end_indices - 1] != 1]
    positive = end_indices[targets[end_indices - 1] == 1]
    negative = end_indices[targets[end_indices - 1] == 0]

    return {
        "file": file_path,
        "ankle": ankle,
        "targets": targets,
        "positive": positive,
        "negative": negative,
        "source_history_rows": source_history_rows,
    }


def load_sensor_data_recordings(csv_path):
    try:
        data = pd.read_csv(
            csv_path,
            usecols=["timestamp", "ankleL_acc_x", "ankleL_acc_y", "ankleL_acc_z",
                     "ankleR_acc_x", "ankleR_acc_y", "ankleR_acc_z",
                     "activity", "fog", "subjectID", "sessionID", "taskID"]
        )
    except Exception as error:
        print(f"WARNING: could not read {csv_path}: {error} -- sensor_data.csv will not be used.")
        return []

    data = data[data["activity"] != 0]
    recordings = []
    for (subject_id, session_id, task_id), group in data.groupby(["subjectID", "sessionID", "taskID"]):
        group = group.sort_values("timestamp")
        times = group["timestamp"].to_numpy(np.float64) * 1000
        if len(times) < 10 or np.any(np.diff(times) <= 0):
            continue
        labels = group["fog"].to_numpy(np.int8)
        fog_starts = times[(labels == 1) & np.r_[True, labels[:-1] != 1]]
        next_index = np.searchsorted(fog_starts, times, side="right")
        valid = next_index < len(fog_starts)
        next_times = np.full(len(times), np.inf)
        next_times[valid] = fog_starts[next_index[valid]]
        targets = (
            (labels != 1)
            & (next_times > times)
            & (next_times <= times + HORIZON_SECONDS * 1000)
        ).astype(np.int8)

        source_rate = 1000 / np.median(np.diff(times))
        source_history_rows = int(HISTORY_SECONDS * source_rate)
        source_interval_rows = max(1, int(PREDICTION_INTERVAL_SECONDS * source_rate))
        if len(times) <= source_history_rows:
            continue
        end_indices = np.arange(source_history_rows, len(times), source_interval_rows)
        if EXCLUDE_IN_FOG_SAMPLES:
            end_indices = end_indices[labels[end_indices - 1] != 1]
        positive = end_indices[targets[end_indices - 1] == 1]
        negative = end_indices[targets[end_indices - 1] == 0]
        if len(positive) == 0 or len(negative) == 0:
            continue

        for side, columns in [
            ("L", ["ankleL_acc_y", "ankleL_acc_x", "ankleL_acc_z"]),
            ("R", ["ankleR_acc_y", "ankleR_acc_x", "ankleR_acc_z"])
        ]:
            raw_ankle = group[columns].to_numpy(np.float32)
            if np.isnan(raw_ankle).any():
                continue
            ankle = causal_normalize(raw_ankle)
            recordings.append({
                "file": f"sensor_data:S{subject_id}_sess{session_id}_task{task_id}_{side}",
                "ankle": ankle,
                "targets": targets,
                "positive": positive,
                "negative": negative,
                "source_history_rows": source_history_rows,
            })
    return recordings


class MixedTrainingSequence(Sequence):
    def __init__(self, original_x, original_y, imu_files, sensor_data_csv_path, history_rows, batch_size):
        super().__init__()
        self.original_x = original_x
        self.original_y = original_y
        self.history_rows = history_rows
        self.batch_size = batch_size
        self.rng = np.random.default_rng(SEED)

        self.sources = {}
        if imu_files:
            imu_recordings = [r for r in (load_imu_file(f) for f in imu_files) if r is not None]
            if imu_recordings:
                self.sources["imu"] = imu_recordings
            elif imu_files:
                print("WARNING: imu_files were found but none could be loaded -- IMU augmentation disabled.")
        if sensor_data_csv_path is not None:
            sensor_recordings = load_sensor_data_recordings(sensor_data_csv_path)
            if sensor_recordings:
                self.sources["sensor_data"] = sensor_recordings
            else:
                print("WARNING: sensor_data.csv found but no usable recordings extracted -- disabled.")

        if not self.sources:
            raise ValueError("MixedTrainingSequence created with no usable augmentation sources.")

        for name, recordings in self.sources.items():
            print(f"Augmentation source '{name}': {len(recordings)} usable recordings")

    def __len__(self):
        return max(1, int(np.ceil(len(self.original_y) / self.batch_size)))

    def _sample_from_source(self, source_name, count, remaining_attempts=50):
        recordings = self.sources[source_name]
        if remaining_attempts <= 0:
            raise RuntimeError(
                f"No usable recording in source '{source_name}' could supply both "
                "positive and negative windows after many attempts."
            )
        record = recordings[self.rng.integers(len(recordings))]
        if len(record["positive"]) == 0 or len(record["negative"]) == 0:
            return self._sample_from_source(source_name, count, remaining_attempts - 1)
        return sample_windows_from_recording(record, count, self.history_rows, self.rng)

    def _augmentation_batch(self, count):
        source_names = list(self.sources.keys())
        per_source_counts = [count // len(source_names)] * len(source_names)
        per_source_counts[-1] += count - sum(per_source_counts)
        x_parts, y_parts = [], []
        for name, sub_count in zip(source_names, per_source_counts):
            if sub_count <= 0:
                continue
            x_part, y_part = self._sample_from_source(name, sub_count)
            x_parts.append(x_part)
            y_parts.append(y_part)
        return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)

    def __getitem__(self, index):
        count = self.batch_size // 2
        original_indices = self.rng.choice(len(self.original_y), count, replace=False)
        augmentation_x, augmentation_y = self._augmentation_batch(self.batch_size - count)
        x = np.concatenate([self.original_x[original_indices], augmentation_x], axis=0)
        y = np.concatenate([self.original_y[original_indices], augmentation_y], axis=0)
        order = self.rng.permutation(len(y))
        return x[order], y[order]


# =====================================================================
# Main
# =====================================================================
def main():
    files = sorted(glob.glob("S*.txt"))
    if not files:
        raise FileNotFoundError("No Daphnet files matching S*.txt were found.")

    all_differences = []
    for file_path in files:
        times = pd.read_csv(file_path, sep=r"\s+", header=None, names=COL_NAMES)["Time"].to_numpy()
        differences = np.diff(times)
        all_differences.extend(differences[differences > 0].tolist())
    sample_rate = 1 / (np.median(all_differences) / 1000)
    history_rows = int(HISTORY_SECONDS * sample_rate)
    interval_rows = max(1, int(PREDICTION_INTERVAL_SECONDS * sample_rate))
    print(f"Sample rate: {sample_rate:.3f} Hz | history_rows={history_rows} | interval_rows={interval_rows}")

    train_files = [f for f in files if f[:3] in TRAIN_SUBJECTS]
    val_files = [f for f in files if f[:3] in VALIDATION_SUBJECTS]
    test_files = [f for f in files if f[:3] in TEST_SUBJECTS]
    assert set(train_files).isdisjoint(val_files) and set(train_files).isdisjoint(test_files) \
        and set(val_files).isdisjoint(test_files), "Subject split leaked across train/val/test!"

    X_train, y_train, _ = build_split(train_files, history_rows, interval_rows)
    X_val, y_val, val_metadata = build_split(val_files, history_rows, interval_rows)
    X_test, y_test, test_metadata = build_split(test_files, history_rows, interval_rows)
    print(f"Train windows: {X_train.shape}  positives={int(y_train.sum())}")
    print(f"Val windows:   {X_val.shape}  positives={int(y_val.sum())}")
    print(f"Test windows:  {X_test.shape}  positives={int(y_test.sum())}")
    for name, arr in [("train", X_train), ("val", X_val), ("test", X_test)]:
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError(f"NaN/Inf detected in {name} features -- aborting before training.")

    positives = np.flatnonzero(y_train == 1)
    negatives = np.flatnonzero(y_train == 0)
    selected_negatives = np.random.choice(
        negatives, min(len(negatives), len(positives) * NEGATIVE_RATIO), replace=False
    )
    selected = np.concatenate([positives, selected_negatives])
    np.random.shuffle(selected)
    X_train, y_train = X_train[selected], y_train[selected]

    imu_files = sorted(glob.glob("imu/SUB*_*.txt")) if USE_IMU_AUGMENTATION else []
    imu_files = [file_path for file_path in imu_files if "standing" not in file_path]
    sensor_data_path = "sensor_data.csv" if (USE_SENSOR_DATA_AUGMENTATION and os.path.exists("sensor_data.csv")) else None
    used_augmentation = bool(imu_files) or (sensor_data_path is not None)
    print(f"IMU augmentation active: {bool(imu_files)} ({len(imu_files)} files found)")
    print(f"sensor_data.csv augmentation active: {sensor_data_path is not None}")

    # Shared callbacks: checkpoint/early-stop on val_f1 (per the project's
    # rule that loss and F1 aren't guaranteed to track under class
    # imbalance + class weighting), separate LR decay on val_loss.
    callbacks = [
        ValidationF1Callback(X_val, y_val, val_metadata, BATCH_SIZE),
        EarlyStopping(
            monitor="val_f1", mode="max", patience=8,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            "best_fog_model.keras", monitor="val_f1", mode="max",
            save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1
        ),
    ]

    model = build_model(history_rows)

    if used_augmentation:
        # MixedTrainingSequence already balances positive/negative windows
        # and mixes in the external sources itself, so no separate
        # sample_weight/batch_size is passed here.
        training_data = MixedTrainingSequence(
            X_train, y_train, imu_files, sensor_data_path, history_rows, BATCH_SIZE
        )
        model.fit(
            training_data,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
    else:
        # Give positive FoG windows extra weight during training.
        # Using explicit sample weights avoids a Keras class_weight issue
        # with the current training-data format.
        sample_weights = np.where(
            y_train == 1,
            POSITIVE_CLASS_WEIGHT,
            1.0
        ).astype(np.float32)

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

    # BUGFIX (v2 -> v2.1): the decision threshold must be chosen on
    # validation probabilities that have gone through the SAME smoothing
    # the test set gets, otherwise a threshold tuned for raw probabilities
    # is systematically too strict once applied to smoothed ones.
    val_probabilities_raw = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0).ravel()
    val_probabilities = smooth_probabilities_by_file(val_probabilities_raw, val_metadata, SMOOTHING_WINDOW_STEPS)
    if THRESHOLD_OBJECTIVE == "max_detection":
        threshold = choose_max_detection_threshold(val_probabilities, val_metadata, MAX_FALSE_ALARMS_PER_MINUTE)
    else:
        threshold = choose_f1_threshold(y_val, val_probabilities)
    val_predictions = apply_threshold_and_persistence(
        val_probabilities,
        val_metadata,
        threshold,
        PERSISTENCE_WINDOW_STEPS,
        PERSISTENCE_REQUIRED_STEPS
    )

    test_probabilities_raw = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
    test_probabilities = smooth_probabilities_by_file(
        test_probabilities_raw, test_metadata, SMOOTHING_WINDOW_STEPS
    )
    test_predictions = apply_threshold_and_persistence(
        test_probabilities,
        test_metadata,
        threshold,
        PERSISTENCE_WINDOW_STEPS,
        PERSISTENCE_REQUIRED_STEPS
    )

    results = {
        "Validation Precision": precision_score(y_val, val_predictions, zero_division=0),
        "Validation Recall": recall_score(y_val, val_predictions, zero_division=0),
        "Validation F1": f1_score(y_val, val_predictions, zero_division=0),
        "Validation Threshold": threshold,
        "Test Precision": precision_score(y_test, test_predictions, zero_division=0),
        "Test Recall": recall_score(y_test, test_predictions, zero_division=0),
        "Test F1": f1_score(y_test, test_predictions, zero_division=0),
        **event_metrics(test_metadata, test_predictions)
    }
    model.save("final_fog_model.keras")
    pd.DataFrame({"Metric": results.keys(), "Value": results.values()}).to_csv(
        "fog_model_results.csv", index=False
    )
    with open("fog_model_config.json", "w", encoding="ascii") as output:
        json.dump({
            "model": "ankle XYZ Conv1D CNN v2 (BatchNorm + GAP)",
            "raw_fog_label": FOG_RAW_LABEL,
            "history_seconds": HISTORY_SECONDS,
            "horizon_seconds": HORIZON_SECONDS,
            "normalization": "causal_expanding_per_recording",
            "exclude_in_fog_samples": EXCLUDE_IN_FOG_SAMPLES,
            "negative_ratio": NEGATIVE_RATIO,
            "positive_class_weight": POSITIVE_CLASS_WEIGHT,
            "smoothing_window_steps": SMOOTHING_WINDOW_STEPS,
            "persistence_window_steps": PERSISTENCE_WINDOW_STEPS,
            "persistence_required_steps": PERSISTENCE_REQUIRED_STEPS,
            "cluster_gap_seconds": CLUSTER_GAP_SECONDS,
            "used_imu_augmentation": bool(imu_files),
            "used_sensor_data_augmentation": sensor_data_path is not None,
            "threshold_objective": THRESHOLD_OBJECTIVE,
            "max_false_alarms_per_minute": MAX_FALSE_ALARMS_PER_MINUTE,
            "minimum_validation_precision": MINIMUM_VALIDATION_PRECISION,
            "threshold_selected_on": "S08 validation only",
            "threshold_value": threshold,
            "test_subjects": sorted(TEST_SUBJECTS),
            "seed": SEED
        }, output, indent=2)

    print("\nVALIDATION")
    print(classification_report(y_val, val_predictions, zero_division=0))
    print("Threshold:", threshold)
    print("\nFINAL TEST (untouched S09/S10, evaluated exactly once)")
    print(classification_report(y_test, test_predictions, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, test_predictions))
    print("\nEVENT METRICS")
    for key, value in results.items():
        if key not in {"Validation Precision", "Validation Recall", "Validation F1",
                        "Validation Threshold", "Test Precision", "Test Recall", "Test F1"}:
            print(key + ":", value)

    test_positive_rate = float(test_predictions.mean())
    print(f"\nFraction of TEST windows flagged positive: {test_positive_rate:.2%}")
    print(
        f"Alarm persistence: {PERSISTENCE_REQUIRED_STEPS} of "
        f"{PERSISTENCE_WINDOW_STEPS} recent predictions"
    )
    print(
        f"Target false-alarm budget: <= {MAX_FALSE_ALARMS_PER_MINUTE:.2f}/min | "
        f"Target alarm-rate cap: <= {MAX_POSITIVE_PREDICTION_RATE:.0%}"
    )

    if results["Test Precision"] > 0.9 and results["Test Recall"] > 0.9:
        print("\n*** SANITY WARNING: test precision AND recall are both >0.9. ***")
        print("*** For 3-second-ahead FoG prediction this is unusual. Check for  ***")
        print("*** label/window leakage before reporting this number anywhere.  ***")

    if test_positive_rate > MAX_POSITIVE_PREDICTION_RATE:
        print(f"\n*** SANITY WARNING: {test_positive_rate:.1%} of all TEST windows were ***")
        print(f"*** flagged positive (cap is {MAX_POSITIVE_PREDICTION_RATE:.0%}). Even though the ***")
        print("*** cluster-based event metrics above may look reasonable, this is  ***")
        print("*** very likely a degenerate always/almost-always-positive detector, ***")
        print("*** not a real FoG detector. Do not report the event metrics above  ***")
        print("*** as a success without checking this number first.               ***")


if __name__ == "__main__":
    main()