"""
tuning.py
=========
Direction-tuning generalization across tasks — the scale-free CI/CD readout.

The v1 readout scored context-invariance from whole-model per-task fit quality::

    CPI = (FDE_saccade - FDE_pursuit) / (FDE_saccade + FDE_pursuit)

which asks *in which task is this neuron better predicted?* In FEF that is
dominated by firing rate (2.1x fewer spikes in pursuit) and by response shape (a
saccade burst is intrinsically easier for a Poisson GLM than a smooth pursuit
ramp) rather than by coding. Measured consequences in v1: signed CPI +0.21 to
+0.45, a 1.48x sac-over-pur FDE ratio even among neurons with real signal in both
tasks, r(log rate, |CPI|) = -0.394, and a Figure-4 coupling slope that dissolved
under every control.

This module asks the mouse's actual question instead: **does the neuron encode
direction the same way in both tasks?** The Direction Generalization Index is the
correlation between the two tasks' mean-centred direction tuning curves. It is
invariant to any per-task affine rescaling of the curve, so the rate difference
and the baseline offset — the two dominant confounds — are removed by
construction rather than by post-hoc control.

The noise ceiling is not optional
---------------------------------
Raw DGI is bounded above by how reliably each task's tuning curve is measured,
and pursuit tuning is measured *less* reliably (fewer spikes) — which would
smuggle the SNR confound straight back in. So DGI is corrected for attenuation by
within-task split-half reliability.

Numerator and denominator are computed on **half-size tuning curves alike**
(cross-task correlations are taken between opposite halves, reliabilities between
same-task halves), so the two carry the same noise level and no Spearman-Brown
step-up is needed::

    DGI_corrected = mean_cross / sqrt( rel_saccade * rel_pursuit )

Neurons whose reliability in either task falls below ``reliability_floor`` are
returned as NaN: below that the correction divides by noise. That gate replaces
v1's ``credible`` gate and does the same job on a scale-free quantity.

Everything here operates on **tuning curves**, so it is independent of how they
were built — model-free epoch mean rates or GLM-predicted profiles both work (see
``docs/04_READOUTS.md`` §2.3). Running both is a design QC check.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "row_correlation",
    "circular_mean_vector",
    "preferred_direction",
    "pd_shift",
    "selectivity_index",
    "tuning_depth",
    "tuning_depth_ratio",
    "direction_generalization_index",
    "fisher_mean",
    "split_indices",
    "curves_from_indices",
    "tuning_curves_by_split",
    "dgi_with_noise_ceiling",
    "dgi_between_epochs",
    "DEFAULT_THETA",
]

#: The eight tested directions, in radians (0 deg = rightward).
DEFAULT_THETA = np.deg2rad(np.arange(0, 360, 45, dtype=float))

_EPS = 1e-12


def _as2d(x):
    """Coerce to (n_units, n_dir); a 1-D input is treated as a single unit."""
    a = np.asarray(x, dtype=float)
    return a[None, :] if a.ndim == 1 else a


def row_correlation(a, b):
    """
    Pearson correlation between corresponding rows of two (n_units, n_dir) arrays.

    Rows with zero variance in either array (a perfectly flat tuning curve) have
    an undefined correlation and are returned as NaN rather than 0 — a flat
    neuron is *unmeasured*, not *invariant*, and collapsing the two is exactly
    the v1 failure in which a maximally task-dependent unit was filed as
    context-invariant.
    """
    A, B = _as2d(a), _as2d(b)
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: {A.shape} vs {B.shape}")
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    na = np.sqrt((A * A).sum(axis=1))
    nb = np.sqrt((B * B).sum(axis=1))
    denom = na * nb
    out = np.full(A.shape[0], np.nan)
    ok = denom > _EPS
    out[ok] = (A[ok] * B[ok]).sum(axis=1) / denom[ok]
    return out


def circular_mean_vector(rates, theta=None, harmonic=1):
    """
    Resultant vector of a direction tuning curve.

    Returns ``(angle, length)``: the circular mean angle in radians and the
    normalised resultant length in [0, 1].

    ``harmonic=1`` treats the variable as a **direction** (0 and 180 deg are
    opposite) and is the right choice for movement direction. ``harmonic=2``
    treats it as an **axis/orientation** (0 and 180 deg equivalent) and matches
    the convention in Noneman et al. eq. 1-2, so use it when quoting numbers on
    their axis.

    Negative rates (possible for mean-centred or GLM-predicted curves) are
    floored at zero, since a resultant vector is only defined for non-negative
    weights.
    """
    R = np.clip(_as2d(rates), 0.0, None)
    th = DEFAULT_THETA if theta is None else np.asarray(theta, float)
    if R.shape[1] != th.size:
        raise ValueError(f"{R.shape[1]} directions but {th.size} angles")
    z = R @ np.exp(1j * harmonic * th)
    total = R.sum(axis=1)
    length = np.where(total > _EPS, np.abs(z) / np.maximum(total, _EPS), np.nan)
    angle = np.angle(z) / harmonic
    return angle, length


def preferred_direction(rates, theta=None, harmonic=1, degrees=True):
    """Preferred direction of each tuning curve, wrapped to [0, 360) deg."""
    ang, _ = circular_mean_vector(rates, theta, harmonic)
    period = 2.0 * np.pi / harmonic
    ang = np.mod(ang, period)
    return np.rad2deg(ang) if degrees else ang


def pd_shift(rates_a, rates_b, theta=None, harmonic=1, degrees=True):
    """
    Unsigned angular distance between the preferred directions of two tasks.

    Wrapped to [0, 180] deg for ``harmonic=1`` (or [0, 90] for ``harmonic=2``).
    Interpretable in degrees, and it separates a *rotated* neuron from an
    *untuned* one in a way the correlation alone cannot.
    """
    aa, _ = circular_mean_vector(rates_a, theta, harmonic)
    ab, _ = circular_mean_vector(rates_b, theta, harmonic)
    period = 2.0 * np.pi / harmonic
    d = np.abs(np.mod(aa - ab + period / 2.0, period) - period / 2.0)
    return np.rad2deg(d) if degrees else d


def selectivity_index(rates, theta=None, harmonic=2):
    """
    Direction selectivity index, Noneman et al. eq. 2::

        SI = | sum_n R_n exp(i k theta_n) | / sum_n |R_n|

    0 = uniform firing across directions, 1 = exclusively one direction.
    ``harmonic=2`` reproduces their published convention.
    """
    R = _as2d(rates)
    th = DEFAULT_THETA if theta is None else np.asarray(theta, float)
    if R.shape[1] != th.size:
        raise ValueError(f"{R.shape[1]} directions but {th.size} angles")
    num = np.abs(R @ np.exp(1j * harmonic * th))
    den = np.abs(R).sum(axis=1)
    return np.where(den > _EPS, num / np.maximum(den, _EPS), np.nan)


def tuning_depth(rates):
    """Peak-to-trough modulation of each tuning curve."""
    R = _as2d(rates)
    return R.max(axis=1) - R.min(axis=1)


def tuning_depth_ratio(rates_a, rates_b):
    """
    ``depth(b) / depth(a)`` — the per-task gain change that DGI is deliberately
    blind to.

    Report it alongside DGI so the gain difference is visible rather than hidden:
    a neuron can have DGI = 1 (identical tuning shape) and a depth ratio of 0.2
    (five times weaker modulation in pursuit), and those are different claims.
    """
    da, db = tuning_depth(rates_a), tuning_depth(rates_b)
    return np.where(da > _EPS, db / np.maximum(da, _EPS), np.nan)


def direction_generalization_index(rates_a, rates_b):
    """
    Raw DGI: correlation between two tasks' mean-centred tuning curves.

    +1 same preferred direction and shape (context-invariant); 0 tuning in one
    task only or unrelated tuning; -1 reversed preference. Uncorrected for the
    noise ceiling — use :func:`dgi_with_noise_ceiling` for the reportable value.
    """
    return row_correlation(rates_a, rates_b)


def fisher_mean(r, axis=0):
    """
    Average correlations via the Fisher z transform.

    Correlations are not additive; averaging them raw biases toward zero. Values
    are clipped just inside +/-1 so that a perfect correlation does not produce
    an infinite z.
    """
    r = np.asarray(r, float)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    with np.errstate(invalid="ignore"):
        return np.tanh(np.nanmean(z, axis=axis))


def split_indices(direction_idx, trial_mask, n_dir=8, rng=None):
    """
    Draw one within-direction half-split and return it as index arrays.

    Returned as a list of ``(idx_a, idx_b)`` pairs, one per direction; a
    direction with fewer than 2 usable trials contributes empty arrays.

    Exposed separately from :func:`tuning_curves_by_split` so that **two
    different epochs measured on the same trials** can be split identically —
    which is what makes a cross-*epoch* DGI well posed (see
    :func:`dgi_between_epochs`).
    """
    rng = np.random.default_rng() if rng is None else rng
    d = np.asarray(direction_idx, int)
    m = np.asarray(trial_mask, bool)
    out = []
    for k in range(n_dir):
        idx = np.flatnonzero(m & (d == k))
        if idx.size < 2:
            out.append((np.empty(0, int), np.empty(0, int)))
            continue
        perm = rng.permutation(idx)
        cut = perm.size // 2
        out.append((perm[:cut], perm[cut:]))
    return out


def curves_from_indices(counts, idx_pairs, half=0):
    """Tuning curve (n_units, n_dir) built from one half of a :func:`split_indices` draw."""
    Y = np.asarray(counts, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    out = np.full((Y.shape[1], len(idx_pairs)), np.nan)
    for k, pair in enumerate(idx_pairs):
        idx = pair[half]
        if idx.size:
            out[:, k] = Y[idx].mean(axis=0)
    return out


def tuning_curves_by_split(counts, direction_idx, trial_mask, n_dir=8, rng=None):
    """
    Split trials into two halves *within each direction* and return the two
    half-data tuning curves.

    Parameters
    ----------
    counts : (n_trials, n_units)
        Spike counts (or rates) in the epoch of interest.
    direction_idx : (n_trials,) int in [0, n_dir)
        Direction condition of each trial. For **pursuit this must be the
        ramp/movement direction, not the step direction** — on backward-step
        trials the target appears at ``angle + 180``.
    trial_mask : (n_trials,) bool
        Trials belonging to the task being summarised.
    rng : numpy Generator, optional

    Returns
    -------
    (half_a, half_b) : two (n_units, n_dir) arrays
        A direction with fewer than 2 usable trials yields NaN in both halves,
        which propagates to NaN in every downstream index rather than silently
        contributing a zero.

    Splitting *within* direction keeps both halves balanced across conditions, so
    a half cannot be biased by an uneven draw of directions.
    """
    pairs = split_indices(direction_idx, trial_mask, n_dir, rng)
    return curves_from_indices(counts, pairs, 0), curves_from_indices(counts, pairs, 1)


def dgi_with_noise_ceiling(counts, direction_idx, mask_a, mask_b,
                           n_dir=8, n_splits=50, reliability_floor=0.2,
                           rng=None):
    """
    Noise-corrected direction generalization index between two tasks.

    Over ``n_splits`` random within-direction half-splits of each task:

    * ``rel_a`` / ``rel_b`` -- same-task correlation between the two halves
      (how reliably the tuning curve is measured in that task);
    * ``cross``            -- the four opposite-half cross-task correlations,
      averaged.

    Because both are computed on **half-size** curves they carry the same noise,
    so the ratio needs no Spearman-Brown step-up::

        dgi_corrected = cross / sqrt(rel_a * rel_b)

    Correlations are averaged across splits in Fisher-z space. Units whose
    reliability in either task is below ``reliability_floor`` get NaN — the
    correction is meaningless when the denominator is noise, and admitting them
    is how a low-rate unit would be relabelled "context-dependent" purely for
    being poorly measured.

    Returns
    -------
    dict with arrays of length n_units:
        ``dgi_raw``           cross-task correlation, uncorrected
        ``dgi_corrected``     the reportable value (NaN below the floor)
        ``reliability_a/_b``  within-task split-half reliabilities
        ``usable``            bool, both reliabilities at or above the floor

    Notes
    -----
    ``dgi_corrected`` can exceed 1 when a reliability is near the floor; it is
    **not** clipped, so that such cases stay visible rather than piling up at the
    invariant pole. Filter on ``usable`` and inspect the tail.
    """
    rng = np.random.default_rng() if rng is None else rng
    ra, rb, cr = [], [], []
    for _ in range(int(n_splits)):
        a1, a2 = tuning_curves_by_split(counts, direction_idx, mask_a, n_dir, rng)
        b1, b2 = tuning_curves_by_split(counts, direction_idx, mask_b, n_dir, rng)
        ra.append(row_correlation(a1, a2))
        rb.append(row_correlation(b1, b2))
        # opposite halves only: a1-b1 shares no trials with a2-b2, so no half is
        # correlated with itself through a shared draw.
        cr.append(fisher_mean(np.vstack([
            row_correlation(a1, b1), row_correlation(a1, b2),
            row_correlation(a2, b1), row_correlation(a2, b2),
        ]), axis=0))

    rel_a = fisher_mean(np.vstack(ra), axis=0)
    rel_b = fisher_mean(np.vstack(rb), axis=0)
    cross = fisher_mean(np.vstack(cr), axis=0)

    usable = (rel_a >= reliability_floor) & (rel_b >= reliability_floor)
    with np.errstate(invalid="ignore"):
        denom = np.sqrt(np.clip(rel_a, 0.0, None) * np.clip(rel_b, 0.0, None))
        corrected = np.where(usable & (denom > _EPS), cross / denom, np.nan)

    return {
        "dgi_raw": cross,
        "dgi_corrected": corrected,
        "reliability_a": rel_a,
        "reliability_b": rel_b,
        "usable": usable,
    }


def dgi_between_epochs(counts_a, counts_b, direction_idx, trial_mask,
                       n_dir=8, n_splits=50, reliability_floor=0.2, rng=None):
    """
    Noise-corrected DGI between two **epochs measured on the same trials**.

    :func:`dgi_with_noise_ceiling` compares two *tasks*, which occupy disjoint
    trials, so independent splits of each are automatically disjoint. Two epochs
    of the same trial are not: the visual window [50, 150] ms after target onset
    and the saccade window [-50, +50] ms around saccade onset come from the same
    trial, and splitting them independently would put the same physical trial in
    both halves of a cross-epoch correlation. Trial-level noise shared between
    two windows of one trial would then inflate the cross term — biasing the
    result **toward** the invariance conclusion, which is the wrong direction to
    be wrong in.

    So this function draws **one** within-direction split and applies it to both
    epochs, and forms the cross term only from **opposite** halves::

        cross = mean( corr(a_half1, b_half2), corr(a_half2, b_half1) )

    Two cross terms rather than four, because ``a_half1`` and ``b_half1`` share
    trials by construction.

    Use this for the visual / saccade / pursuit (VSP) generalization matrix
    wherever the two epochs come from the same task block; use
    :func:`dgi_with_noise_ceiling` for saccade-vs-pursuit, whose trials differ.

    Returns the same dict as :func:`dgi_with_noise_ceiling`.
    """
    rng = np.random.default_rng() if rng is None else rng
    ra, rb, cr = [], [], []
    for _ in range(int(n_splits)):
        pairs = split_indices(direction_idx, trial_mask, n_dir, rng)
        a1 = curves_from_indices(counts_a, pairs, 0)
        a2 = curves_from_indices(counts_a, pairs, 1)
        b1 = curves_from_indices(counts_b, pairs, 0)
        b2 = curves_from_indices(counts_b, pairs, 1)
        ra.append(row_correlation(a1, a2))
        rb.append(row_correlation(b1, b2))
        cr.append(fisher_mean(np.vstack([
            row_correlation(a1, b2), row_correlation(a2, b1),
        ]), axis=0))

    rel_a = fisher_mean(np.vstack(ra), axis=0)
    rel_b = fisher_mean(np.vstack(rb), axis=0)
    cross = fisher_mean(np.vstack(cr), axis=0)
    usable = (rel_a >= reliability_floor) & (rel_b >= reliability_floor)
    with np.errstate(invalid="ignore"):
        denom = np.sqrt(np.clip(rel_a, 0.0, None) * np.clip(rel_b, 0.0, None))
        corrected = np.where(usable & (denom > _EPS), cross / denom, np.nan)

    return {
        "dgi_raw": cross,
        "dgi_corrected": corrected,
        "reliability_a": rel_a,
        "reliability_b": rel_b,
        "usable": usable,
    }
