"""Past-only right-ankle complexity features; no annotations or participant inputs."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from gravity_gait import gravity_signals


def complexity(z):
    """Six descriptors per signal, for native 60 Hz trailing data.

    SampEn uses m=2, r=.2 SD, equal template counts, unordered pairs,
    and a documented .5 count correction to make sparse estimates finite.
    Only SampEn/permutation computations use 3-sample block means (20 Hz).
    """
    z = np.asarray(z, float)
    if z.ndim != 3 or z.shape[1] < 30 or z.shape[1] % 3 or not np.isfinite(z).all():
        raise ValueError('Expected finite batched native signals, length divisible by 3')
    short = z.reshape(len(z), -1, 3, z.shape[2]).mean(2)
    scale = short.std(1, keepdims=True)
    norm = (short-short.mean(1, keepdims=True))/np.maximum(scale, 1e-12)
    triples = sliding_window_view(norm, 3, axis=1)
    n = triples.shape[1]
    pair = np.triu(np.ones((n, n), bool), 1)[None, :, :, None]
    distance2 = np.zeros((len(z), n, n, z.shape[2]))
    for j in range(2):
        v = triples[..., j]
        distance2 = np.maximum(distance2, abs(v[:, :, None]-v[:, None, :]))
    b = ((distance2 <= .2) & pair).sum((1, 2))
    v = triples[..., 2]
    distance3 = np.maximum(distance2, abs(v[:, :, None]-v[:, None, :]))
    a = ((distance3 <= .2) & pair).sum((1, 2))
    sample = -np.log((a+.5)/(b+.5))
    orders = np.argsort(triples, axis=-1, kind='stable')
    code = orders[..., 0]*9+orders[..., 1]*3+orders[..., 2]
    perm = np.zeros_like(sample)
    for k in [5, 7, 11, 15, 19, 21]:
        p = (code == k).mean(1)
        perm -= p*np.log(np.maximum(p, 1e-15))/np.log(6)
    centered = z-z.mean(1, keepdims=True)
    power = abs(np.fft.rfft(centered*np.hanning(z.shape[1])[None, :, None], axis=1))**2
    power[:, 0] = 0
    total = power.sum(1, keepdims=True)
    p = power/np.maximum(total, 1e-20)
    spectral = -(p*np.log(np.maximum(p, 1e-15))).sum(1)/np.log(power.shape[1]-1)
    f = np.fft.rfftfreq(z.shape[1], 1/60)
    lo = power[:, (f >= .5) & (f < 3)].sum(1)
    hi = power[:, (f >= 3) & (f < 8)].sum(1)
    fi = np.log((hi+1e-12)/(lo+1e-12))
    ac = []
    for lag in range(3, min(21, norm.shape[1]//2)):
        left, right = norm[:, :-lag], norm[:, lag:]
        ac.append((left*right).sum(1)/np.maximum(np.sqrt((left*left).sum(1)*(right*right).sum(1)), 1e-12))
    regularity = np.max(ac, axis=0)
    constant = scale[:, 0] < 1e-10
    for value in [sample, perm, spectral, fi, regularity]:
        value[constant] = 0
    return np.concatenate([sample, perm, spectral, regularity, fi, np.log1p(z.std(1))], 1)


def regularity_features(raw, batch=64):
    """Full trailing 3 s, earlier/later 1.5 s and later-minus-earlier features."""
    raw = np.asarray(raw, np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (180, 6) or not np.isfinite(raw).all():
        raise ValueError('Expected finite native right-ankle windows (N,180,6)')
    rows = []
    for start in range(0, len(raw), batch):
        z = gravity_signals(raw[start:start+batch]).astype(float)
        full, early, late = complexity(z), complexity(z[:, :90]), complexity(z[:, 90:])
        rows.append(np.concatenate([full, early, late, late-early], 1))
    return np.concatenate(rows).astype(np.float32) if rows else np.empty((0, 96), np.float32)
