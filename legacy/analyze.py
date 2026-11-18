import glob
import json

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

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D
)
from tensorflow.keras.models import Sequential


# ============================================================
# SETTINGS
# ============================================================

FOG_RAW_LABEL = 2

HISTORY_SECONDS = 3
HORIZON_SECONDS = 3

PREDICTION_INTERVAL_SECONDS = 0.25

# Training
NEGATIVE_RATIO = 3
POSITIVE_CLASS_WEIGHT = 1.5

EPOCHS = 15
BATCH_SIZE = 256

SEED = 42


# ============================================================
# SUBJECT SPLIT
# ============================================================

TRAIN_SUBJECTS = {
    "S01", "S02", "S03",
    "S04", "S05", "S06", "S07"
}

VALIDATION_SUBJECTS = {
    "S08"
}

TEST_SUBJECTS = {
    "S09", "S10"
}


# ============================================================
# COLUMNS
# ============================================================

COL_NAMES = [
    "Time",
    "Ankle_X",
    "Ankle_Y",
    "Ankle_Z",
    "Leg_X",
    "Leg_Y",
    "Leg_Z",
    "Trunk_X",
    "Trunk_Y",
    "Trunk_Z",
    "Label"
]


np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# READ RECORDING
# ============================================================

def read_recording(file_path):

    data = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COL_NAMES
    )

    # Keep the original row number so we can detect
    # gaps after removing label 0.
    data["Original_Index"] = np.arange(
        len(data)
    )

    # Label 0 is outside the experiment.
    data = data[
        data["Label"] != 0
    ].copy()

    data.reset_index(
        drop=True,
        inplace=True
    )

    ankle = data[
        [
            "Ankle_X",
            "Ankle_Y",
            "Ankle_Z"
        ]
    ].to_numpy(
        np.float32
    )

    labels = data[
        "Label"
    ].to_numpy(
        np.int8
    )

    times = data[
        "Time"
    ].to_numpy(
        np.float64
    )

    indices = data[
        "Original_Index"
    ].to_numpy(
        np.int64
    )

    return (
        ankle,
        labels,
        times,
        indices
    )


# ============================================================
# NORMALIZE EACH RECORDING
# ============================================================

def normalize_recording(ankle):

    mean = ankle.mean(
        axis=0,
        dtype=np.float64
    ).astype(
        np.float32
    )

    std = ankle.std(
        axis=0,
        dtype=np.float64
    ).astype(
        np.float32
    )

    std[
        std == 0
    ] = 1.0

    return (
        ankle - mean
    ) / std


# ============================================================
# FUTURE FOG TARGET
# ============================================================

def future_fog_target(
    labels,
    times
):

    # Find the beginning of every FoG episode.
    starts = (
        (labels == FOG_RAW_LABEL)
        &
        np.r_[
            True,
            labels[:-1] != FOG_RAW_LABEL
        ]
    )

    fog_start_times = times[
        starts
    ]

    target = np.zeros(
        len(labels),
        dtype=np.int8
    )

    if len(fog_start_times) == 0:
        return (
            target,
            fog_start_times
        )

    # Find the next FoG onset after each row.
    next_index = np.searchsorted(
        fog_start_times,
        times,
        side="right"
    )

    valid = (
        next_index
        <
        len(fog_start_times)
    )

    next_times = np.full(
        len(times),
        np.inf
    )

    next_times[valid] = (
        fog_start_times[
            next_index[valid]
        ]
    )

    # Positive if a FoG episode begins
    # within the next 3 seconds.
    target = (
        (labels != FOG_RAW_LABEL)
        &
        (next_times > times)
        &
        (
            next_times
            <=
            times
            +
            HORIZON_SECONDS * 1000
        )
    ).astype(
        np.int8
    )

    return (
        target,
        fog_start_times
    )


# ============================================================
# BUILD DATASET
# ============================================================

def build_split(
    files,
    history_rows,
    interval_rows
):

    sequences = []
    labels = []
    metadata = []

    for file_path in files:

        (
            ankle,
            raw_labels,
            times,
            original_indices
        ) = read_recording(
            file_path
        )

        target, fog_start_times = (
            future_fog_target(
                raw_labels,
                times
            )
        )

        ankle = normalize_recording(
            ankle
        )

        # Move through the recording every 0.25 seconds.
        for end_index in range(
            history_rows,
            len(ankle),
            interval_rows
        ):

            start_index = (
                end_index
                -
                history_rows
            )

            # Do not create a sequence that crosses
            # a gap where label 0 was removed.
            if not np.all(
                np.diff(
                    original_indices[
                        start_index:end_index
                    ]
                )
                ==
                1
            ):
                continue

            sequences.append(
                ankle[
                    start_index:end_index
                ]
            )

            labels.append(
                target[
                    end_index - 1
                ]
            )

            metadata.append({

                "file": file_path,

                "time": float(
                    times[
                        end_index - 1
                    ]
                ),

                "fog_starts": fog_start_times
            })

    if not sequences:

        return (
            np.empty(
                (
                    0,
                    history_rows,
                    3
                ),
                np.float32
            ),

            np.empty(
                0,
                np.int8
            ),

            []
        )

    return (
        np.asarray(
            sequences,
            np.float32
        ),

        np.asarray(
            labels,
            np.int8
        ),

        metadata
    )


# ============================================================
# SMOOTH MODEL PROBABILITIES
# ============================================================

def smooth_probabilities(
    metadata,
    probabilities
):

    smoothed = np.zeros_like(
        probabilities
    )

    current_file = None

    file_values = []
    file_indices = []

    def flush_file():

        if not file_indices:
            return

        values = np.asarray(
            file_values
        )

        for position, index in enumerate(
            file_indices
        ):

            # Look at the current prediction
            # and previous two predictions.
            start = max(
                0,
                position - 2
            )

            smoothed[index] = np.max(
                values[
                    start:position + 1
                ]
            )

    for index, item in enumerate(
        metadata
    ):

        if (
            current_file is not None
            and
            item["file"] != current_file
        ):

            flush_file()

            file_values = []
            file_indices = []

        current_file = item["file"]

        file_values.append(
            probabilities[index]
        )

        file_indices.append(
            index
        )

    flush_file()

    return smoothed


# ============================================================
# CHOOSE THRESHOLD
# ============================================================

def choose_threshold(
    y_true,
    probabilities
):

    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities
        )
    )

    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:

        predictions = (
            probabilities
            >= threshold
        ).astype(
            np.int8
        )

        current_precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        current_recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        current_f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        # We still care about F1,
        # but give recall some importance.
        score = (
            0.4 * current_f1
            +
            0.6 * current_recall
        )

        if score > best_f1:

            best_f1 = score

            best_threshold = float(
                threshold
            )

    return best_threshold


# ============================================================
# REQUIRE CONSECUTIVE PREDICTIONS
# ============================================================

def consecutive_predictions(
    metadata,
    probabilities,
    threshold,
    required
):

    predictions = np.zeros(
        len(probabilities),
        dtype=np.int8
    )

    current_file = None

    count = 0
    previous_time = None

    for i, item in enumerate(
        metadata
    ):

        current_file_name = item[
            "file"
        ]

        current_time = item[
            "time"
        ]

        # Reset when a new recording starts.
        if (
            current_file is None
            or
            current_file_name != current_file
        ):

            count = 0
            previous_time = None
            current_file = current_file_name

        is_positive = (
            probabilities[i]
            >= threshold
        )

        # Predictions must be approximately
        # 0.25 seconds apart.
        continuous = (
            previous_time is not None
            and
            abs(
                (
                    current_time
                    -
                    previous_time
                )
                -
                PREDICTION_INTERVAL_SECONDS
                * 1000
            )
            <= 10
        )

        if (
            is_positive
            and
            continuous
        ):

            count += 1

        elif is_positive:

            count = 1

        else:

            count = 0

        # Trigger only after enough consecutive
        # positive predictions.
        if count >= required:

            predictions[i] = 1

        previous_time = current_time

    return predictions


# ============================================================
# CLUSTER ALARMS
# ============================================================

def cluster_times(
    times,
    gap_ms
):

    clusters = []

    for time in sorted(times):

        if (
            not clusters
            or
            time
            -
            clusters[-1][-1]
            >
            gap_ms
        ):

            clusters.append(
                [time]
            )

        else:

            clusters[-1].append(
                time
            )

    return clusters


# ============================================================
# EVENT METRICS
# ============================================================

def event_metrics(
    metadata,
    predictions
):

    by_file = {}

    for item, prediction in zip(
        metadata,
        predictions
    ):

        values = by_file.setdefault(
            item["file"],
            {
                "fog_starts": item[
                    "fog_starts"
                ],
                "times": []
            }
        )

        if prediction == 1:

            values["times"].append(
                item["time"]
            )

    detected = 0
    total_events = 0

    warning_times = []

    false_alarm_clusters = 0

    experiment_minutes = 0.0

    for file_path, values in by_file.items():

        _, _, times, _ = read_recording(
            file_path
        )

        if len(times) > 1:

            experiment_minutes += (
                times[-1]
                -
                times[0]
            ) / 60000

        fog_starts = values[
            "fog_starts"
        ]

        total_events += len(
            fog_starts
        )

        positive_times = np.asarray(
            values["times"],
            dtype=np.float64
        )

        # ----------------------------------------------------
        # Detect FoG episodes.
        # ----------------------------------------------------

        for onset in fog_starts:

            warnings = positive_times[
                (positive_times < onset)
                &
                (
                    positive_times
                    >=
                    onset
                    -
                    HORIZON_SECONDS
                    * 1000
                )
            ]

            if len(warnings):

                detected += 1

                # Earliest warning is the useful one.
                warning_times.append(
                    (
                        onset
                        -
                        warnings[0]
                    ) / 1000
                )

        # ----------------------------------------------------
        # Group repeated alarms together.
        # ----------------------------------------------------

        clusters = cluster_times(
            positive_times,
            1000
        )

        for cluster in clusters:

            has_future_event = any(

                np.any(
                    (fog_starts > alarm)
                    &
                    (
                        fog_starts
                        <=
                        alarm
                        +
                        HORIZON_SECONDS
                        * 1000
                    )
                )

                for alarm in cluster
            )

            if not has_future_event:

                false_alarm_clusters += 1

    return {

        "FoG Episodes":
            total_events,

        "Episodes Detected Before Onset (%)":
            (
                100
                *
                detected
                /
                total_events
                if total_events
                else 0.0
            ),

        "False Alarms Per Minute":
            (
                false_alarm_clusters
                /
                experiment_minutes
                if experiment_minutes
                else 0.0
            ),

        "Average Warning Time (seconds)":
            (
                float(
                    np.mean(
                        warning_times
                    )
                )
                if warning_times
                else 0.0
            ),

        "Median Warning Time (seconds)":
            (
                float(
                    np.median(
                        warning_times
                    )
                )
                if warning_times
                else 0.0
            )
    }


# ============================================================
# CHOOSE BEST CONSECUTIVE REQUIREMENT
# ============================================================

def choose_persistence(
    metadata,
    y_true,
    probabilities,
    threshold
):

    print()
    print("=" * 55)
    print("TESTING WARNING PERSISTENCE")
    print("=" * 55)

    best_required = 1
    best_score = -999999

    for required in [1, 2, 3, 4, 5]:

        predictions = consecutive_predictions(
            metadata,
            probabilities,
            threshold,
            required
        )

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        events = event_metrics(
            metadata,
            predictions
        )

        detection = events[
            "Episodes Detected Before Onset (%)"
        ]

        false_alarms = events[
            "False Alarms Per Minute"
        ]

        # We want high episode detection,
        # but strongly punish excessive alarms.
        score = (
            detection
            -
            12
            *
            false_alarms
        )

        print()
        print(
            "Required consecutive predictions:",
            required
        )

        print(
            "Precision:",
            round(precision, 4)
        )

        print(
            "Recall:",
            round(recall, 4)
        )

        print(
            "F1:",
            round(f1, 4)
        )

        print(
            "Episode detection:",
            round(detection, 2),
            "%"
        )

        print(
            "False alarms/min:",
            round(false_alarms, 3)
        )

        print(
            "Score:",
            round(score, 3)
        )

        if score > best_score:

            best_score = score
            best_required = required

    print()
    print(
        "Chosen persistence:",
        best_required,
        "consecutive predictions"
    )

    return best_required


# ============================================================
# MAIN
# ============================================================

def main():

    files = sorted(
        glob.glob("S*.txt")
    )

    if not files:

        raise FileNotFoundError(
            "No Daphnet files matching S*.txt were found."
        )

    print()
    print("=" * 55)
    print("FILES")
    print("=" * 55)

    for file in files:
        print(file)

    # --------------------------------------------------------
    # Sample rate
    # --------------------------------------------------------

    all_differences = []

    for file_path in files:

        times = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            names=COL_NAMES
        )[
            "Time"
        ].to_numpy()

        differences = np.diff(
            times
        )

        differences = differences[
            differences > 0
        ]

        all_differences.extend(
            differences.tolist()
        )

    sample_rate = (
        1
        /
        (
            np.median(
                all_differences
            )
            /
            1000
        )
    )

    history_rows = int(
        HISTORY_SECONDS
        *
        sample_rate
    )

    interval_rows = max(
        1,
        int(
            PREDICTION_INTERVAL_SECONDS
            *
            sample_rate
        )
    )

    print()
    print("=" * 55)
    print("TIMING")
    print("=" * 55)

    print(
        "Sample rate:",
        sample_rate,
        "Hz"
    )

    print(
        "History rows:",
        history_rows
    )

    print(
        "Prediction interval:",
        interval_rows,
        "rows"
    )

    # --------------------------------------------------------
    # Split files
    # --------------------------------------------------------

    train_files = [
        f for f in files
        if f[:3] in TRAIN_SUBJECTS
    ]

    val_files = [
        f for f in files
        if f[:3] in VALIDATION_SUBJECTS
    ]

    test_files = [
        f for f in files
        if f[:3] in TEST_SUBJECTS
    ]

    # --------------------------------------------------------
    # Build datasets
    # --------------------------------------------------------

    print()
    print("=" * 55)
    print("BUILDING DATASETS")
    print("=" * 55)

    (
        X_train,
        y_train,
        train_meta
    ) = build_split(
        train_files,
        history_rows,
        interval_rows
    )

    (
        X_val,
        y_val,
        val_meta
    ) = build_split(
        val_files,
        history_rows,
        interval_rows
    )

    (
        X_test,
        y_test,
        test_meta
    ) = build_split(
        test_files,
        history_rows,
        interval_rows
    )

    print()
    print(
        "Training:",
        X_train.shape
    )

    print(
        "Validation:",
        X_val.shape
    )

    print(
        "Test:",
        X_test.shape
    )

    print()
    print(
        "Training positives:",
        np.sum(y_train == 1)
    )

    print(
        "Training negatives:",
        np.sum(y_train == 0)
    )

    # --------------------------------------------------------
    # Balance training data.
    # --------------------------------------------------------

    positive = np.flatnonzero(
        y_train == 1
    )

    negative = np.flatnonzero(
        y_train == 0
    )

    negative_count = min(
        len(negative),
        len(positive)
        *
        NEGATIVE_RATIO
    )

    selected_negative = np.random.choice(
        negative,
        negative_count,
        replace=False
    )

    selected = np.concatenate(
        [
            positive,
            selected_negative
        ]
    )

    np.random.shuffle(
        selected
    )

    X_train = X_train[
        selected
    ]

    y_train = y_train[
        selected
    ]

    print()
    print(
        "Balanced training set:",
        X_train.shape
    )

    print(
        "Positive:",
        np.sum(y_train == 1)
    )

    print(
        "Negative:",
        np.sum(y_train == 0)
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("=" * 55)
    print("BUILDING CNN")
    print("=" * 55)

    model = Sequential([

        Input(
            shape=(
                history_rows,
                3
            )
        ),

        Conv1D(
            32,
            5,
            activation="relu"
        ),

        MaxPooling1D(
            2
        ),

        Conv1D(
            64,
            5,
            activation="relu"
        ),

        MaxPooling1D(
            2
        ),

        Flatten(),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(
            0.3
        ),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print("=" * 55)
    print("TRAINING")
    print("=" * 55)

    model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight={
            0: 1.0,
            1: POSITIVE_CLASS_WEIGHT
        },

        callbacks=[

            EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True
            ),

            ModelCheckpoint(
                "best_fog_model.keras",
                monitor="val_loss",
                save_best_only=True
            )
        ],

        verbose=1
    )

    # ========================================================
    # VALIDATION PROBABILITIES
    # ========================================================

    val_probabilities = model.predict(
        X_val,
        batch_size=BATCH_SIZE,
        verbose=0
    ).ravel()

    val_probabilities = smooth_probabilities(
        val_meta,
        val_probabilities
    )

    # --------------------------------------------------------
    # Choose threshold on S08 only.
    # --------------------------------------------------------

    threshold = choose_threshold(
        y_val,
        val_probabilities
    )

    print()
    print(
        "Validation threshold:",
        threshold
    )

    # ========================================================
    # CHOOSE PERSISTENCE ON VALIDATION
    # ========================================================

    persistence = choose_persistence(

        val_meta,

        y_val,

        val_probabilities,

        threshold
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    val_predictions = consecutive_predictions(

        val_meta,

        val_probabilities,

        threshold,

        persistence
    )

    print()
    print("=" * 55)
    print("FINAL VALIDATION")
    print("=" * 55)

    print(
        classification_report(
            y_val,
            val_predictions,
            zero_division=0
        )
    )

    print(
        "Threshold:",
        threshold
    )

    print(
        "Persistence:",
        persistence
    )

    validation_events = event_metrics(
        val_meta,
        val_predictions
    )

    print()
    print(
        "Validation event metrics:"
    )

    for key, value in validation_events.items():

        print(
            key + ":",
            value
        )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print()
    print("=" * 55)
    print("FINAL TEST")
    print("=" * 55)

    print(
        "S09 and S10 are untouched until now."
    )

    test_probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0
    ).ravel()

    test_probabilities = smooth_probabilities(
        test_meta,
        test_probabilities
    )

    # IMPORTANT:
    # We use the threshold and persistence
    # selected from S08.
    test_predictions = consecutive_predictions(

        test_meta,

        test_probabilities,

        threshold,

        persistence
    )

    # --------------------------------------------------------
    # Point-wise results
    # --------------------------------------------------------

    test_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    print()

    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0
        )
    )

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            test_predictions
        )
    )

    print()
    print(
        "Test precision:",
        test_precision
    )

    print(
        "Test recall:",
        test_recall
    )

    print(
        "Test F1:",
        test_f1
    )

    # ========================================================
    # EVENT RESULTS
    # ========================================================

    test_events = event_metrics(
        test_meta,
        test_predictions
    )

    print()
    print("=" * 55)
    print("FINAL EVENT RESULTS")
    print("=" * 55)

    for key, value in test_events.items():

        print(
            key + ":",
            value
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model.save(
        "final_fog_model.keras"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {

        "Validation Precision":
            precision_score(
                y_val,
                val_predictions,
                zero_division=0
            ),

        "Validation Recall":
            recall_score(
                y_val,
                val_predictions,
                zero_division=0
            ),

        "Validation F1":
            f1_score(
                y_val,
                val_predictions,
                zero_division=0
            ),

        "Validation Threshold":
            threshold,

        "Consecutive Predictions Required":
            persistence,

        "Test Precision":
            test_precision,

        "Test Recall":
            test_recall,

        "Test F1":
            test_f1,

        **test_events
    }

    pd.DataFrame({
        "Metric": results.keys(),
        "Value": results.values()
    }).to_csv(
        "fog_model_results.csv",
        index=False
    )

    # ========================================================
    # SAVE CONFIG
    # ========================================================

    with open(
        "fog_model_config.json",
        "w",
        encoding="ascii"
    ) as output:

        json.dump({

            "raw_fog_label":
                FOG_RAW_LABEL,

            "history_seconds":
                HISTORY_SECONDS,

            "horizon_seconds":
                HORIZON_SECONDS,

            "prediction_interval_seconds":
                PREDICTION_INTERVAL_SECONDS,

            "negative_ratio":
                NEGATIVE_RATIO,

            "positive_class_weight":
                POSITIVE_CLASS_WEIGHT,

            "threshold_selected_on":
                "S08 validation only",

            "persistence_selected_on":
                "S08 validation only",

            "consecutive_predictions_required":
                persistence

        }, output, indent=2)

    print()
    print("=" * 55)
    print("DONE")
    print("=" * 55)

    print()
    print(
        "Saved:"
    )

    print(
        "final_fog_model.keras"
    )

    print(
        "fog_model_results.csv"
    )

    print(
        "fog_model_config.json"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()