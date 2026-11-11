"""
stacking.py
===========
Stage 1 of the WP6 fit pipeline: the FEF analog of the mouse
`glm_data_stacked.pkl` (built once by `run_sessions.py`, reused by every
model script in `models/`).

One PRODUCTION conversion per (session, probe) — real spikes, artifact gates
live, per-monkey window and kernel exceptions applied — staged through the
exact GLM path (`FEFPredictorBuilder.build` → `stack_design_matrices` →
`prepare_features_for_glm`) and cached to

    {RESULTS_DIR}/stacked/{session}_{probeLabel}.pkl

so the model scripts never touch the .mat files again. MEASURED on 0136a:
34-39 s per probe warm (~74-78 MB caches; an earlier 7.5 min/probe estimate
was cold-cache timing), so the full 48-probe pass is well under an hour,
one-time, resumable. .mat access must stay SERIAL (one HDF5 reader at a
time); the cached pickles are plain files and parallel-safe to read.

NOTE (measured 2026-08-11, do not "optimize" this away): the flat
`sorting.spike_times` vector CANNOT replace the per-trial spiketimes cells —
it is on the raw imec clock, and comparison against the refs shows a
per-trial offset drifting 0.45 → 3.7 ms across the session (imec-vs-nidq
drift that the per-trial refs already correct). Exactness is required; the
slow path is the ground truth.

Cache schema (pickle dict):
    X            float32 (n_frames, n_cols)   z-scored design (the fit's X)
    Y            int16   (n_frames, n_units)  raw spike counts
    metadata     dict of per-frame arrays (trial_id, context, choice, …)
    predictor_names  list[str]
    predictor_info   dict  (builder groups)
    predictor_info_split, feature_group_size  (prepare_features_for_glm)
    provenance   dict: session, monkey, probe, probe_label, bin_ms, win_pre,
                 win_post, builder windows/lags, dropped_groups,
                 converter_stats (incl. n_artifact / n_dead for S8),
                 n_units, created (iso date)
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from datetime import datetime

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     *(['..'] * 4)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core.session_pipeline import (stack_design_matrices,          # noqa: E402
                                   prepare_features_for_glm)

from .. import config as C                                         # noqa: E402
from ..design.predictors import FEFPredictorBuilder                # noqa: E402


def stacked_dir():
    return os.path.join(C.RESULTS_DIR, 'stacked')


def probe_key(session_key, probe_label):
    return f'{session_key}_{probe_label}'


def stack_path(session_key, probe_label):
    return os.path.join(stacked_dir(),
                        f'{probe_key(session_key, probe_label)}.pkl')


def session_mat_path(session_key, monkey):
    name = session_key if monkey == 'Y' else f'kendra_scrappy_{session_key}'
    return os.path.join(C.SESSIONS_DIR, f'{name}_g0-{C.EXPORT_HASH}.mat')


def all_probe_jobs():
    """[(session_key, monkey)] for the full 26-session inventory. Probe count
    is discovered from the file at conversion time (4 Y sessions are
    single-probe)."""
    return ([(s, 'S') for s in C.MONKEY_S_SESSIONS]
            + [(s, 'Y') for s in C.MONKEY_Y_SESSIONS])


def _check_raw_counts(Y):
    """Y must be raw spike counts — no calcium scaling/smoothing path."""
    if Y.size == 0:
        raise ValueError('Y is empty')
    if Y.min() < 0 or not np.allclose(Y, np.round(Y)):
        raise ValueError('Y is not raw integer spike counts')


def build_probe_stack(session_key, monkey, probe, cv=None, verbose=True):
    """
    Full production staging for one probe. Returns the cache dict.

    Pass an open converter via `cv` to reuse its table cache across the two
    probes of a session (halves the non-spike I/O).
    """
    from ..io import NHPSessionConverter   # local import: keep TF-free

    own = cv is None
    if own:
        cv = NHPSessionConverter(session_mat_path(session_key, monkey),
                                 verbose=verbose)
    try:
        t0 = time.time()
        glm = cv.build(probe, skip_spikes=False)          # PRODUCTION path
        t_convert = time.time() - t0

        builder = FEFPredictorBuilder(verbose=False)
        dms, names, pinfo = builder.build(glm, monkey=monkey)

        X, Y, meta, _ = stack_design_matrices(glm, dms, pinfo,
                                              neural_metric='spikes')
        _check_raw_counts(Y)
        import contextlib, io as _io
        with contextlib.redirect_stdout(_io.StringIO()):
            pinfo_split, fgs = prepare_features_for_glm(pinfo)
        if int(fgs.sum()) != X.shape[1]:
            raise ValueError('feature_group_size does not sum to n_cols')

        if Y.max() > np.iinfo(np.int16).max:
            raise ValueError(f'spike count {Y.max()} overflows int16')
        stack = dict(
            X=X.astype(np.float32),
            Y=Y.astype(np.int16),
            metadata=meta,
            predictor_names=names,
            predictor_info=pinfo,
            predictor_info_split=pinfo_split,
            feature_group_size=fgs,
            provenance=dict(
                session=session_key, monkey=monkey, probe=probe,
                probe_label=cv.probe_labels[probe],
                bin_ms=cv.bin_ms, win_pre=cv.win_pre, win_post=cv.win_post,
                move_win=builder.move_win, reward_win=builder.reward_win,
                n_eye_lags=builder.n_eye_lags, n_pos_lags=builder.n_pos_lags,
                direction_param=builder.direction_param,
                dropped_groups=dict(builder.dropped_groups),
                converter_stats={k: v for k, v in cv.stats.items()},
                n_units=Y.shape[1], n_trials=len(glm),
                n_frames=X.shape[0], n_cols=X.shape[1],
                t_convert_s=round(t_convert, 1),
                created=datetime.now().isoformat(timespec='seconds'),
            ),
        )
        return stack
    finally:
        if own:
            cv.close()


def write_probe_stack(session_key, monkey, probe, cv=None, verbose=True):
    """Build + atomically write one probe's cache. Returns its path."""
    stack = build_probe_stack(session_key, monkey, probe, cv=cv,
                              verbose=verbose)
    path = stack_path(session_key, stack['provenance']['probe_label'])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(stack, f, protocol=4)
    os.replace(tmp, path)
    return path


def load_probe_stack(path_or_key):
    """Load one cache (path, or '<session>_<label>' key). X is upcast to
    float64 for the TF fit; Y stays int (raw counts, no calcium scaling)."""
    path = (path_or_key if os.path.sep in path_or_key
            or path_or_key.endswith('.pkl')
            else os.path.join(stacked_dir(), f'{path_or_key}.pkl'))
    with open(path, 'rb') as f:
        stack = pickle.load(f)
    stack['X'] = np.asarray(stack['X'], np.float64)
    stack['Y'] = np.asarray(stack['Y'], np.float64)   # GLM_CV expects float
    return stack


def available_probe_keys():
    """Sorted keys of every cached probe (the model scripts' registry)."""
    d = stacked_dir()
    if not os.path.isdir(d):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d)
                  if f.endswith('.pkl'))
