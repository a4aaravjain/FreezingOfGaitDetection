"""Causal Daphnet data, targets, alarm replay, and evaluation (seconds, g).
Labels are used for supervision/scoring only, never to suppress online alarms.
"""
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix)

@dataclass(frozen=True)
class Config:
    sample_rate: int = 64
    history_seconds: float = 3.0
    horizon_seconds: float = 3.0
    interval_seconds: float = 0.25
    seed: int = 42
    epochs: int = 18
    batch_size: int = 256
    learning_rate: float = 0.0005
    positive_weight_cap: float = 10.0
    ema_alpha: float = 0.5
    persistence: int = 2
    hysteresis_ratio: float = 0.8
    cooldown_seconds: float = 3.0
    max_alarm_fraction: float = 0.15
    max_false_alarm_fraction: float = 0.10
    max_positive_fraction: float = 0.20
    max_false_alarms_per_minute: float = 1.0
    min_warning_precision: float = 0.20
    # Provisional engineering budget, not a clinically established tolerance.
    # Selection guard only: replay still reports the full alarm duration.
    max_alarm_duration_seconds: float = 5.0

    @property
    def history_rows(self):
        return round(self.sample_rate * self.history_seconds)

    @property
    def interval_rows(self):
        return round(self.sample_rate * self.interval_seconds)

    def validate(self):
        if self.sample_rate != 64 or self.history_rows < 2 or self.interval_rows < 1:
            raise ValueError("This adapter requires Daphnet 64 Hz; invalid window/stride.")
        if self.horizon_seconds <= 0 or not 0 < self.ema_alpha <= 1:
            raise ValueError("Invalid horizon or EMA.")
        if self.persistence < 1 or not 0 < self.hysteresis_ratio <= 1:
            raise ValueError("Invalid alarm settings.")
        if self.cooldown_seconds < 0 or self.positive_weight_cap <= 0 or self.max_alarm_duration_seconds <= 0:
            raise ValueError("Invalid cooldown/weight cap.")
        for value in asdict(self).values():
            if not np.isfinite(value):
                raise ValueError("Nonfinite configuration.")
        for name in ("max_alarm_fraction", "max_false_alarm_fraction", "max_positive_fraction", "min_warning_precision"):
            if not 0 <= getattr(self,name) <= 1:
                raise ValueError("Invalid policy budget.")
        for name in ("history_seconds", "interval_seconds"):
            n = getattr(self, name) * self.sample_rate
            if not np.isclose(n, round(n)):
                raise ValueError("History and interval must be integer sample counts.")

@dataclass
class Segment:
    key: str
    subject: str
    time: np.ndarray
    xyz_g: np.ndarray
    labels: np.ndarray
    onsets: np.ndarray
    left_censored_fog: bool

@dataclass
class Windows:
    x: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    segments: list

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

def verify_split(train, validation, evaluation=()):
    groups = [set(train), set(validation), set(evaluation)]
    if not groups[0] or not groups[1]:
        raise ValueError("Training and validation subjects cannot be empty.")
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Subject leakage: split subjects overlap.")

def read_recording(path, cfg=Config()):
    """Keep physical time; split on label 0 or missing samples, never concatenate."""
    cfg.validate()
    path = Path(path)
    a = pd.read_csv(path, sep=r"\s+", header=None).to_numpy(np.float64)
    if a.ndim != 2 or a.shape[1] != 11 or len(a) < 2:
        raise ValueError(f"{path}: expected 11 columns and at least two rows.")
    if not np.isfinite(a).all() or not np.isin(a[:, 10], [0, 1, 2]).all():
        raise ValueError(f"{path}: nonfinite values or unknown labels.")
    t, y = a[:, 0] / 1000, a[:, 10].astype(np.int8)
    dt = np.diff(t)
    if (dt <= 0).any():
        raise ValueError(f"{path}: timestamps must strictly increase; do not sort.")
    regular = dt < 1.5 / cfg.sample_rate
    if not regular.any() or abs(1 / dt[regular].mean() - cfg.sample_rate) > 0.5:
        raise ValueError(f"{path}: unexpected sampling rate.")
    if np.any(np.abs(dt[regular] - 1 / cfg.sample_rate) > .0011):
        raise ValueError(f"{path}: excessive timestamp jitter.")
    valid = y != 0
    breaks = np.r_[True, (~valid[:-1]) | (~valid[1:]) | (~regular)]
    starts = np.flatnonzero(valid & breaks)
    segments = []
    for start in starts:
        stop = start + 1
        while stop < len(y) and valid[stop] and not breaks[stop]:
            stop += 1
        ts, ys = t[start:stop], y[start:stop]
        # First row already freezing: actual onset is unknown, not a new event.
        onset = ts[1:][(ys[1:] == 2) & (ys[:-1] == 1)]
        segments.append(Segment(f"{path.name}:{start}", path.name[:3], ts,
                                (a[start:stop, 1:4] / 1000).astype(np.float32),
                                ys, onset, bool(ys[0] == 2)))
    return segments

def load_subjects(root, subjects, cfg=Config()):
    paths = sorted(p for p in Path(root).glob("S??R??.txt") if p.name[:3] in subjects)
    present = {p.name[:3] for p in paths}
    if present != set(subjects):
        raise ValueError(f"Missing subjects: {set(subjects) - present}")
    segments = [s for p in paths for s in read_recording(p, cfg)]
    if not segments:
        raise ValueError("No valid segments.")
    return segments, paths

def targets(segment, ends, cfg=Config()):
    """At t: label 1 iff a known onset occurs in (t, t+H].
    -1 = active FoG or incomplete future observation; omit from binary loss,
    but retain these predictions in device replay. Post-FoG is not removed.
    """
    t = segment.time[ends]
    y = np.zeros(len(ends), np.int8)
    j = np.searchsorted(segment.onsets, t, side="right")
    nxt = np.full(len(t), np.inf)
    valid = j < len(segment.onsets)
    nxt[valid] = segment.onsets[j[valid]]
    y[nxt <= t + cfg.horizon_seconds + 1e-9] = 1
    y[(segment.labels[ends] == 2) |
      (t + cfg.horizon_seconds > segment.time[-1] + 1e-9)] = -1
    return y

def build_windows(segments, cfg=Config()):
    xs, ys, rows = [], [], []
    for s in segments:
        ends = np.arange(cfg.history_rows - 1, len(s.time), cfg.interval_rows)
        if not len(ends):
            continue
        windows = np.lib.stride_tricks.sliding_window_view(
            s.xyz_g, cfg.history_rows, axis=0).transpose(0, 2, 1)[::cfg.interval_rows]
        xs.append(windows)
        ys.append(targets(s, ends, cfg))
        for end in ends:
            rows.append(dict(segment=s.key, subject=s.subject, time=float(s.time[end]),
                             end_row=int(end), active_fog=bool(s.labels[end] == 2),
                             history_contains_fog=bool(np.any(
                                 s.labels[end-cfg.history_rows+1:end+1] == 2))))
    if not xs:
        raise ValueError("No complete sensor windows.")
    x, y = np.concatenate(xs).copy(), np.concatenate(ys)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite features.")
    return Windows(x, y, pd.DataFrame(rows), segments)

def fit_normalizer(segments):
    # Each raw training row exactly once, before class handling/window overlap.
    x = np.concatenate([s.xyz_g for s in segments]).astype(np.float64)
    mean, std = x.mean(axis=0), x.std(axis=0)
    return dict(mean=mean.tolist(), std=np.maximum(std, 1e-3).tolist(),
                input_unit="g", raw_dataset_unit="mg",
                axes=["forward", "vertical", "lateral"], kind="fixed_training_zscore")

def normalize(x, parameters):
    return ((x - np.asarray(parameters["mean"], np.float32)) /
            np.asarray(parameters["std"], np.float32)).astype(np.float32)

class Alarm:
    """Same causal state machine for offline replay and future hardware."""
    def __init__(self, threshold, cfg=Config()):
        self.threshold, self.cfg = threshold, cfg
        self.reset()

    def reset(self):
        self.ema = None
        self.active = False
        self.count = 0
        self.next_allowed = -np.inf

    def step(self, time, probability):
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid model probability.")
        c = self.cfg
        self.ema = (probability if self.ema is None else
                    c.ema_alpha * probability + (1-c.ema_alpha) * self.ema)
        started = False
        if self.active:
            if self.ema < self.threshold * c.hysteresis_ratio:
                self.active = False
                self.next_allowed = time + c.cooldown_seconds
                self.count = 0
        else:
            self.count = self.count + 1 if self.ema >= self.threshold else 0
            if self.count >= c.persistence and time >= self.next_allowed:
                self.active, started = True, True
        return self.ema, self.active, started

def replay(data, probabilities, threshold, cfg=Config()):
    probabilities = np.asarray(probabilities).ravel()
    if len(probabilities) != len(data.y):
        raise ValueError("Probability/metadata length mismatch.")
    if len(data.meta) != len(data.y) or not data.meta.index.equals(pd.RangeIndex(len(data.y))):
        raise ValueError("Metadata must be aligned and consecutively indexed.")
    keys = data.meta.segment.to_numpy()
    runs = keys[np.r_[True,keys[1:] != keys[:-1]]]
    if len(runs) != len(set(runs)):
        raise ValueError("A segment reappeared out of order.")
    result = data.meta.copy()
    result["target"], result["probability"] = data.y, probabilities
    smooth, active, starts, durations = [], [], [], []
    by_key = {s.key: s for s in data.segments}
    for key, group in result.groupby("segment", sort=False):
        machine = Alarm(threshold, cfg)
        times = group.time.to_numpy()
        dt = np.diff(times)
        if (dt <= 0).any() or np.any(np.abs(dt-cfg.interval_seconds) > .002):
            raise ValueError("Noncontiguous predictions; split/reset before replay.")
        end = by_key[key].time[-1] + 1 / cfg.sample_rate
        holds = np.diff(np.r_[times, end])
        if np.any(holds <= 0) or np.any(holds > cfg.interval_seconds + .002):
            raise ValueError("Invalid hold duration.")
        for t, p, duration in zip(times, group.probability, holds):
            e, a, start = machine.step(t, float(p))
            smooth.append(e); active.append(a); starts.append(start)
            durations.append(float(duration))
    result["smoothed"], result["alarm_active"] = smooth, active
    result["alarm_start"], result["duration"] = starts, durations
    return result

def window_metrics(y, p, threshold):
    mask = y >= 0
    y, p = np.asarray(y)[mask], np.asarray(p)[mask]
    if not len(y):
        return {"n": 0, "precision": None, "recall": None, "f1": None,
                "pr_auc_average_precision": None, "confusion_matrix": [[0,0],[0,0]]}
    pred = p >= threshold
    return dict(n=len(y), positives=int(y.sum()),
                precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)),
                pr_auc_average_precision=float(average_precision_score(y,p)) if y.sum() else None,
                confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),
                positive_fraction=float(pred.mean()))

def event_metrics(data, replayed, cfg=Config()):
    """One alarm START can match one subsequent onset; an old continuous alarm
    never detects a later episode merely by remaining high.
    Burden counts every second, including active FoG and censored recording tails.
    """
    total = eligible = detected = false = true = censored = 0
    alarm_seconds = false_seconds = uncertain_seconds = duration = 0.
    warnings_s, alarm_lengths, event_rows = [], [], []
    groups = {k: g for k, g in replayed.groupby("segment", sort=False)}
    for s in data.segments:
        total += len(s.onsets)
        g = groups.get(s.key)
        matched = {}
        if g is None:
            for onset in s.onsets:
                event_rows.append(dict(segment=s.key,onset=float(onset),eligible=False,warning_seconds=None))
            continue
        times = g.time.to_numpy()
        onsets = s.onsets
        can_warn = [bool(np.any((times < o) & (times >= o-cfg.horizon_seconds) &
                               (~g.active_fog.to_numpy()))) for o in onsets]
        eligible += sum(can_warn)
        for t in g.loc[g.alarm_start, "time"]:
            j = np.searchsorted(onsets, t, side="right")
            # Require normal state at actual alarm start; no label-gated device.
            state = s.labels[np.searchsorted(s.time, t, side="right")-1]
            if j < len(onsets) and onsets[j]-t <= cfg.horizon_seconds+1e-9 and state == 1 and j not in matched:
                matched[j] = float(onsets[j]-t); true += 1
                warnings_s.append(float(onsets[j]-t))
            elif state == 1 and t + cfg.horizon_seconds > s.time[-1] + 1e-9:
                censored += 1
            else:
                false += 1
        for j,o in enumerate(onsets):
            event_rows.append(dict(segment=s.key,onset=float(o),eligible=can_warn[j],
                                   warning_seconds=matched.get(j)))
        detected += len(matched)
        run_length = 0.
        for row in g.itertuples():
            duration += row.duration
            if row.alarm_active:
                run_length += row.duration
                alarm_seconds += row.duration
                # Union of pre-onset intervals, clipped to known normal samples.
                # Integrate at source sample resolution, not overlapping windows.
                left = np.searchsorted(s.time, row.time, side="left")
                right = np.searchsorted(s.time, row.time+row.duration-1e-9, side="left")
                st = s.time[left:right]
                ends = np.minimum(np.r_[s.time[left+1:right], row.time+row.duration],
                                  row.time+row.duration)
                weights = np.maximum(0, ends-st)
                j = np.searchsorted(onsets, st, side="right")
                nxt = np.full(len(st),np.inf)
                ok = j < len(onsets)
                nxt[ok] = onsets[j[ok]]
                useful = (nxt-st <= cfg.horizon_seconds+1e-9) & (s.labels[left:right] == 1)
                unknown = (~useful) & (s.labels[left:right] == 1) & (st+cfg.horizon_seconds > s.time[-1]+1e-9)
                false_seconds += float(weights[(~useful) & (~unknown)].sum())
                uncertain_seconds += float(weights[unknown].sum())
            elif run_length:
                alarm_lengths.append(run_length); run_length = 0.
        if run_length:
            alarm_lengths.append(run_length)
    metrics = dict(total_fog_episodes=total, eligible_episodes=eligible,
                   episodes_predicted_before_onset=detected,
                   episode_sensitivity=detected/total if total else None,
                   eligible_episode_sensitivity=detected/eligible if eligible else None,
                   left_censored_fog_segments=sum(s.left_censored_fog for s in data.segments),
                   true_warning_events=true, total_false_alarms=false,
                   censored_alarm_starts=censored, unscorable_alarm_seconds=uncertain_seconds,
                   false_alarms_per_minute=false*60/duration if duration else None,
                   false_alarms_per_hour=false*3600/duration if duration else None,
                   monitored_seconds=duration, total_alarm_seconds=alarm_seconds,
                   valid_sensor_seconds=sum(float(s.time[-1]-s.time[0]+1/cfg.sample_rate) for s in data.segments),
                   alarm_active_percentage=100*alarm_seconds/duration if duration else 0.,
                   alarm_active_fraction=alarm_seconds/duration if duration else 0.,
                   false_alarm_seconds=false_seconds,
                   false_alarm_active_fraction=false_seconds/duration if duration else 0.,
                   average_alarm_duration_seconds=float(np.mean(alarm_lengths)) if alarm_lengths else 0.,
                   maximum_alarm_duration_seconds=max(alarm_lengths,default=0.),
                   warning_times_seconds=warnings_s,
                   average_warning_seconds=float(np.mean(warnings_s)) if warnings_s else None,
                   median_warning_seconds=float(np.median(warnings_s)) if warnings_s else None,
                   warning_precision=true/(true+false) if true+false else 0.)
    return metrics, event_rows

def evaluate(data, probabilities, threshold, cfg=Config()):
    r = replay(data, probabilities, threshold, cfg)
    device, events = event_metrics(data,r,cfg)
    raw = window_metrics(data.y,probabilities,threshold)
    device["smoothed_positive_fraction"] = float(np.average(
        r.smoothed >= threshold, weights=r.duration))
    for name, mask in [("history_without_fog", ~data.meta.history_contains_fog.to_numpy()),
                       ("history_with_fog", data.meta.history_contains_fog.to_numpy())]:
        raw[name] = window_metrics(data.y[mask],np.asarray(probabilities)[mask],threshold)
    return {"window":raw,"device":device}, r, events

def policy_rejection_reasons(metrics, cfg=Config()):
    """Explain ALL failed inner-validation constraints, without changing scores.
    A duration cap rejects a policy; it never truncates replay to hide burden.
    """
    d, w = metrics["device"], metrics["window"]
    checks = {
        "no_true_warning": d["true_warning_events"] > 0,
        "warning_precision_below_minimum": d["warning_precision"] >= cfg.min_warning_precision,
        "false_starts_over_budget": d["false_alarms_per_minute"] <= cfg.max_false_alarms_per_minute,
        "alarm_time_over_budget": d["alarm_active_fraction"] <= cfg.max_alarm_fraction,
        "false_alarm_time_over_budget": d["false_alarm_active_fraction"] <= cfg.max_false_alarm_fraction,
        "smoothed_positive_rate_over_budget": d["smoothed_positive_fraction"] <= cfg.max_positive_fraction,
        "raw_positive_rate_over_budget": w["positive_fraction"] <= cfg.max_positive_fraction,
        "continuous_alarm_too_long": d["maximum_alarm_duration_seconds"] > 0
            and d["maximum_alarm_duration_seconds"] <= cfg.max_alarm_duration_seconds + 1e-9,
    }
    # A no-alarm policy already fails no_true_warning; duration itself is fine.
    if d["maximum_alarm_duration_seconds"] == 0:
        checks["continuous_alarm_too_long"] = True
    return [name for name, passed in checks.items() if not passed]


def select_threshold(data, probabilities, cfg=Config(), role="validation"):
    if role != "validation":
        raise ValueError("Threshold optimization is allowed on validation only.")
    if set(data.y[data.y >= 0]) != {0,1}:
        raise ValueError("Threshold selection needs both validation classes.")
    candidates, best = [], None
    # Predeclared small grid; do not exhaustively overfit every probability.
    for threshold in np.linspace(.05,.95,37):
        metrics, _, _ = evaluate(data,probabilities,float(threshold),cfg)
        d = metrics["device"]
        reasons = policy_rejection_reasons(metrics,cfg)
        feasible = not reasons
        candidates.append(dict(threshold=float(threshold),feasible=feasible,
                               rejection_reasons=";".join(reasons),
                               raw_positive_fraction=metrics["window"]["positive_fraction"],
                               window_precision=metrics["window"]["precision"],
                               window_recall=metrics["window"]["recall"],
                               window_f1=metrics["window"]["f1"],**d))
        rank = (d["episodes_predicted_before_onset"],d["warning_precision"],
                -d["false_alarm_active_fraction"],float(threshold))
        if feasible and (best is None or rank > best[0]):
            best = (rank,float(threshold))
    if best is None:
        warnings.warn("NO FEASIBLE VALIDATION POLICY: alarms disabled; do not deploy.")
        return 1.1,candidates
    return best[1],candidates

class OnlinePredictor:
    """Input samples in g, forward/vertical/lateral, at 64 Hz.
    No labels. Missing/invalid samples reset buffer and alarm. Supply explicit
    reset at recording/device-session boundaries. Measure inference latency
    separately on the target hardware before reporting usable warning times.
    """
    def __init__(self, predict, parameters, threshold, cfg=Config()):
        self.predict, self.parameters, self.cfg = predict, parameters, cfg
        self.alarm = Alarm(threshold,cfg)
        self.reset()

    def reset(self):
        self.buffer = deque(maxlen=self.cfg.history_rows)
        self.count, self.last = 0, None
        self.alarm.reset()

    def step(self, time, xyz_g):
        x = np.asarray(xyz_g,np.float32)
        if x.shape != (3,) or not np.isfinite(x).all() or not np.isfinite(time):
            self.reset()
            raise ValueError("Invalid IMU sample; stream reset.")
        if self.last is not None:
            dt = time-self.last
            if dt <= 0 or abs(dt-1/self.cfg.sample_rate) > .0011:
                self.reset()
                if dt <= 0:
                    raise ValueError("Non-increasing timestamp; stream reset.")
        self.last = time
        self.buffer.append(x); self.count += 1
        if self.count < self.cfg.history_rows or (self.count-self.cfg.history_rows) % self.cfg.interval_rows:
            return None
        x = normalize(np.asarray(self.buffer)[None],self.parameters)
        p = float(np.asarray(self.predict(x)).ravel()[0])
        e,a,start = self.alarm.step(time,p)
        return dict(time=time,probability=p,smoothed=e,alarm_active=a,alarm_start=start)

