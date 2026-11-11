"""
predictors.py
=============
`FEFPredictorBuilder` — the v2 oculomotor design matrix (WP4 port of v1's
`nhp/predictors.py`, 2026-08-11; blueprint in prompts/WP4_IMPLEMENTATION_PLAN.md).

Emits exactly the contract the mouse `generate_predictors` returns:

    (design_matrices, predictor_names, predictor_info)

where `design_matrices` is a list of per-trial (n_bins, n_features) arrays and
`predictor_info` maps group name -> {'start', 'end', 'names'} with contiguous,
ascending, non-overlapping column ranges covering every column. That contiguity
is load-bearing: `stack_design_matrices` builds `group_indices` by walking the
dict in order, `prepare_features_for_glm` derives `feature_group_size` from it,
and `GLM_CV`'s group lasso asserts the sizes sum to X.shape[1].

v2 CHANGES from the v1 builder (everything else is a faithful port):

1. DIRECTION REPARAMETERIZATION. v1's `direction` (48) + movement-locked-only
   `context_direction` (24) become
       direction_shared (48)  harmonics x (target + movement kernels), both tasks
       direction_diff   (48)  the SAME columns x the sum-coded task indicator
   so per-task tuning is recoverable in closed form (w_sac = w_shared + w_diff,
   w_pur = w_shared - w_diff; fef/design/reparam.py) and the target-locked
   half of the task difference — which v1 could not express — is modelled.
   `direction_param='per_task'` instead emits direction_sac / direction_pur
   (the WP7 comparison variant; exactly the shared_diff design times a sparse
   invertible transform).

2. DATA-DRIVEN DEGENERATE-GROUP DROPPING (PLAN A5). `jump` is dropped when the
   trial set carries fewer than 2 pursuit step-direction levels (Monkey Y: all
   backward -> the group would be an exact linear combination of `target` and
   `context`). `magnitude` is dropped when BOTH tasks are single-level (Y).
   Dropping is decided from the trials themselves, cross-checked against
   config.JUMP_DEGENERATE_MONKEYS when `monkey` is passed, and recorded in
   `self.dropped_groups`. Group ORDER of the remaining groups is unchanged, so
   Monkey-S column indexing is untouched; downstream code must key groups by
   NAME via predictor_info, never by position.

3. NEW OPTIONAL GROUPS (all off by default until WP5 rules):
   `catchup_amplitude` — each catch-up saccade's kernel weighted by its
   amplitude (converter's events['catchup_amps']).
   `pupil` — flag reserved; the v2 converter does not yet extract pupil, so
   requesting it raises rather than silently emitting nothing.

No intercept column is emitted (GLM_CV fits per-neuron w0). The coupling group
is deliberately absent here — it is added by the coupling stage (WP8).
"""

from __future__ import annotations

import os
import sys

import numpy as np

# repo root (…/GLM_encoding_model, which holds core/) is FIVE dirnames up:
# predictors.py -> design -> fef -> FEF_v2 -> NHP_FEF -> GLM_encoding_model
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     *(['..'] * 4)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core.predictor_builder import create_cosine_bumps   # noqa: E402

from .. import config as C   # noqa: E402


def _flag(cfg, name, override, default=False):
    """Resolve an optional config flag, letting a call-site override win."""
    return getattr(cfg, name, default) if override is None else override


def _kernel_bank(win_ms, n_bases):
    """Raised-cosine centers/widths spanning `win_ms` = (lo, hi) in ms."""
    lo, hi = win_ms
    centers = np.linspace(lo, hi, n_bases)
    spacing = centers[1] - centers[0] if n_bases > 1 else (hi - lo)
    widths = np.full(n_bases, spacing * 2.0)
    return centers, widths


class FEFPredictorBuilder:
    """
    Build oculomotor design matrices for a converted FEF session.

    Parameters mirror `fef.config`; override per-call for sensitivity analyses.
    After `build()`, `self.dropped_groups` maps group name -> reason for every
    group removed by the degeneracy rules (S8: report, never silently absorb).
    """

    def __init__(self, bin_ms=None,
                 n_target=None, n_move=None, n_go=None, n_reward=None,
                 target_win=None, move_win=None, go_win=None, reward_win=None,
                 harmonics=None, verbose=True,
                 include_sustained=None, n_sustained=None, sustained_win=None,
                 include_signed_velocity=None, include_signed_accel=None,
                 include_catchup=None, n_catchup=None, catchup_win=None,
                 include_eye_lags=None, n_eye_lags=None, eye_lag_win=None,
                 shared_only=None, include_go=None, include_delay=None,
                 include_target_velocity=None, include_retinal_slip=None,
                 include_drift=None, n_drift=None,
                 include_step_direction=None, include_jump=None,
                 include_stim_offset=None, n_stim_offset=None,
                 stim_offset_win=None, include_catchup_direction=None,
                 catchup_harmonics=None, include_magnitude=None,
                 direction_param=None, include_catchup_amplitude=None,
                 include_pupil=None, n_pos_lags=None):
        self.bin_ms = C.BIN_MS if bin_ms is None else float(bin_ms)
        self.n_target = C.N_TARGET_BASES if n_target is None else n_target
        self.n_move = C.N_MOVE_BASES if n_move is None else n_move
        self.n_go = C.N_GO_BASES if n_go is None else n_go
        self.n_reward = C.N_REWARD_BASES if n_reward is None else n_reward
        self.target_win = C.TARGET_WIN_MS if target_win is None else target_win
        self.move_win = C.MOVE_WIN_MS if move_win is None else move_win
        self.go_win = C.GO_WIN_MS if go_win is None else go_win
        self.reward_win = C.REWARD_WIN_MS if reward_win is None else reward_win
        # Per-monkey window overrides (build(monkey=...)) apply only when the
        # caller did not pass an explicit window.
        self._move_win_explicit = move_win is not None
        self._reward_win_explicit = reward_win is not None
        self.harmonics = C.DIRECTION_HARMONICS if harmonics is None else harmonics
        self.include_sustained = (C.INCLUDE_SUSTAINED if include_sustained is None
                                  else include_sustained)
        self.n_sustained = C.N_SUSTAINED_BASES if n_sustained is None else n_sustained
        self.sustained_win = (C.SUSTAINED_WIN_MS if sustained_win is None
                              else sustained_win)
        self.include_signed_velocity = (C.INCLUDE_SIGNED_VELOCITY
                                        if include_signed_velocity is None
                                        else include_signed_velocity)
        self.include_signed_accel = (C.INCLUDE_SIGNED_ACCEL
                                     if include_signed_accel is None
                                     else include_signed_accel)
        self.include_catchup = (C.INCLUDE_CATCHUP if include_catchup is None
                                else include_catchup)
        self.n_catchup = C.N_CATCHUP_BASES if n_catchup is None else n_catchup
        self.catchup_win = C.CATCHUP_WIN_MS if catchup_win is None else catchup_win
        self.shared_only = (C.SHARED_PREDICTORS_ONLY if shared_only is None
                            else shared_only)
        if self.shared_only and reward_win is None:
            self.reward_win = C.REWARD_WIN_SHARED_MS
        self.include_go = C.INCLUDE_GO if include_go is None else include_go
        self.include_delay = C.INCLUDE_DELAY if include_delay is None else include_delay
        self.include_target_velocity = (C.INCLUDE_TARGET_VELOCITY
                                        if include_target_velocity is None
                                        else include_target_velocity)
        self.include_retinal_slip = (C.INCLUDE_RETINAL_SLIP
                                     if include_retinal_slip is None
                                     else include_retinal_slip)
        self.include_drift = C.INCLUDE_DRIFT if include_drift is None else include_drift
        self.n_drift = C.N_DRIFT_BASES if n_drift is None else n_drift
        self.include_step_direction = _flag(C, 'INCLUDE_STEP_DIRECTION',
                                            include_step_direction)
        self.include_jump = _flag(C, 'INCLUDE_JUMP', include_jump)
        self.include_stim_offset = _flag(C, 'INCLUDE_STIM_OFFSET',
                                         include_stim_offset)
        self.n_stim_offset = (C.N_STIM_OFFSET_BASES if n_stim_offset is None
                              else n_stim_offset)
        self.stim_offset_win = (C.STIM_OFFSET_WIN_MS if stim_offset_win is None
                                else stim_offset_win)
        self.include_catchup_direction = _flag(C, 'INCLUDE_CATCHUP_DIRECTION',
                                               include_catchup_direction)
        self.catchup_harmonics = (C.CATCHUP_HARMONICS if catchup_harmonics is None
                                  else catchup_harmonics)
        self.include_magnitude = _flag(C, 'INCLUDE_MAGNITUDE', include_magnitude)
        self.include_catchup_amplitude = _flag(C, 'INCLUDE_CATCHUP_AMPLITUDE',
                                               include_catchup_amplitude)
        self.include_pupil = _flag(C, 'INCLUDE_PUPIL', include_pupil)
        self.include_eye_lags = (C.INCLUDE_EYE_LAGS if include_eye_lags is None
                                 else include_eye_lags)
        self.n_eye_lags = C.N_EYE_LAG_BASES if n_eye_lags is None else n_eye_lags
        self.eye_lag_win = C.EYE_LAG_WIN_MS if eye_lag_win is None else eye_lag_win
        # Position-only lag count (defaults to n_eye_lags). Eye POSITION is
        # slowly varying — on pursuit it is ~linear in time, so its lagged
        # copies are near-collinear; the velocity bank does not have this
        # problem and keeps its full resolution.
        self.n_pos_lags = self.n_eye_lags if n_pos_lags is None else n_pos_lags
        self._n_pos_lags_explicit = n_pos_lags is not None
        self._n_eye_lags_explicit = n_eye_lags is not None
        self.direction_param = (C.DIRECTION_PARAM if direction_param is None
                                else direction_param)
        if self.direction_param not in ('shared_diff', 'per_task'):
            raise ValueError(f'unknown direction_param {self.direction_param!r}')
        if self.include_pupil:
            raise NotImplementedError(
                'INCLUDE_PUPIL: the v2 converter does not extract pupil yet '
                '(WP5 decision; prompts/WP4_IMPLEMENTATION_PLAN.md §1). '
                'Refusing to build a design that silently lacks the group.')
        self.verbose = verbose
        self.dropped_groups = {}

    # -- primitives (ported unchanged from v1) --------------------------------
    def _event_kernels(self, n_bins, event_bin, win_ms, n_bases):
        """
        (n_bins, n_bases) raised-cosine kernels anchored on `event_bin`.

        All-zero when the event is absent (NaN) — e.g. the go cue on pursuit
        trials — so the group stays well defined for every trial.
        """
        out = np.zeros((n_bins, n_bases))
        if event_bin is None or not np.isfinite(event_bin):
            return out
        centers, widths = _kernel_bank(win_ms, n_bases)
        lag_ms = (np.arange(n_bins) - float(event_bin)) * self.bin_ms
        return create_cosine_bumps(lag_ms, centers, widths)

    def _multi_event_kernels(self, n_bins, event_bins, win_ms, n_bases,
                             weights=None):
        """
        (n_bins, n_bases) kernels for an arbitrary NUMBER of events in a trial:

            col_k(t) = sum_e w_e b_k(t - t_e)

        `weights` scales each event's contribution, defaulting to 1. It must
        align with `event_bins`; a length mismatch means the two were filtered
        inconsistently upstream, which would silently attach each weight to the
        wrong event, so it raises.
        """
        out = np.zeros((n_bins, n_bases))
        ev = np.asarray(event_bins, float).ravel()
        if weights is None:
            w = np.ones(ev.size)
        else:
            w = np.asarray(weights, float).ravel()
            if w.size != ev.size:
                raise ValueError(
                    f'{w.size} event weights for {ev.size} events — the two '
                    f'were filtered inconsistently')
        ok = np.isfinite(ev) & np.isfinite(w)
        ev, w = ev[ok], w[ok]
        if ev.size == 0:
            return out
        centers, widths = _kernel_bank(win_ms, n_bases)
        t = np.arange(n_bins)
        for e, wi in zip(ev, w):
            out += wi * create_cosine_bumps((t - e) * self.bin_ms,
                                            centers, widths)
        return out

    @staticmethod
    def _event_weights(dirs_deg, harmonic, fn):
        """cos/sin of `harmonic` x each event's direction."""
        d = (np.asarray(dirs_deg, float).ravel() if dirs_deg is not None
             else np.empty(0))
        return fn(harmonic * np.radians(d))

    def _lagged(self, x, win_ms, n_bases):
        """
        (n_bins, n_bases) temporal basis expansion of a CONTINUOUS regressor.
        `win_ms` may extend to negative lags (activity leads the eye).
        Edge-padded rather than zero-padded (kinematic signals have no natural
        zero; a zero pad would inject a spurious step at the trial boundary).
        """
        x = np.asarray(x, float)
        n = x.size
        lo, hi = win_ms
        d_lo = int(np.floor(lo / self.bin_ms))
        d_hi = int(np.ceil(hi / self.bin_ms))
        lags = np.arange(d_lo, d_hi + 1)
        centers, widths = _kernel_bank(win_ms, n_bases)
        B = create_cosine_bumps(lags * self.bin_ms, centers, widths)

        out = np.zeros((n, n_bases))
        for j, d in enumerate(lags):
            if not B[j].any():
                continue
            if d > 0:
                xs = np.concatenate([np.full(min(d, n), x[0]), x[:max(n - d, 0)]])
            elif d < 0:
                k = min(-d, n)
                xs = np.concatenate([x[k:], np.full(k, x[-1])])
            else:
                xs = x
            out += np.outer(xs, B[j])
        return out

    def _direction_basis(self, angle_deg):
        """[cos t, sin t, cos 2t, sin 2t, ...] for one trial's direction."""
        th = np.radians(angle_deg)
        vals = []
        for h in range(1, self.harmonics + 1):
            vals += [np.cos(h * th), np.sin(h * th)]
        return np.asarray(vals)

    @staticmethod
    def _zsafe(x):
        """Center/scale a within-trial continuous regressor, NaN-safe."""
        x = np.asarray(x, float)
        x = np.nan_to_num(x, nan=np.nanmean(x) if np.isfinite(x).any() else 0.0)
        return x

    # -- degeneracy scan (v2, PLAN A5) ----------------------------------------
    def _scan_degeneracies(self, trials, monkey=None):
        """
        Decide which optional groups are degenerate for THIS trial set.

        jump      : < 2 finite step-direction levels among pursuit trials.
                    With one level, jump = const x 1[pursuit] x K_targ — an
                    exact linear combination of `target` and `context`.
        magnitude : magnitude_code carries no variance in EITHER task (each
                    task single-level -> the within-task z-score guard zeroes
                    it -> 12 all-zero columns -> NaN under z-scoring).
                    If exactly ONE task is single-level the group silently
                    becomes task-specific, which changes its aliasing class —
                    warn loudly and record, but keep it.

        Cross-checked against config.JUMP_DEGENERATE_MONKEYS when `monkey` is
        given: a mismatch is new information about the data, so it raises
        rather than being absorbed.
        """
        jump_levels = set()
        mag_var = {C.CTX_SACCADE: 0.0, C.CTX_PURSUIT: 0.0}
        for t in trials:
            if int(t['context']) == C.CTX_PURSUIT:
                j = t.get('jump')
                if j is not None and np.isfinite(j):
                    jump_levels.add(float(j))
            m = t.get('magnitude_code')
            if m is not None and np.isfinite(m):
                mag_var[int(t['context'])] += abs(float(m))

        if self.include_jump and len(jump_levels) < 2:
            self.dropped_groups['jump'] = (
                f'pursuit has {len(jump_levels)} step-direction level(s) '
                f'{sorted(jump_levels)}; group would alias target x context')
        if monkey is not None:
            cfg_degenerate = monkey in C.JUMP_DEGENERATE_MONKEYS
            data_degenerate = len(jump_levels) < 2
            if cfg_degenerate != data_degenerate:
                raise RuntimeError(
                    f'jump degeneracy mismatch for monkey {monkey}: config '
                    f'says {cfg_degenerate}, data says {data_degenerate} '
                    f'(levels {sorted(jump_levels)}). New information — '
                    f'update fef/config.py deliberately, do not absorb.')

        if self.include_magnitude:
            dead = [k for k, v in mag_var.items() if v == 0.0]
            if len(dead) == 2:
                self.dropped_groups['magnitude'] = (
                    'magnitude_code has zero variance in both tasks '
                    '(single-level session); 12 all-zero columns')
            elif len(dead) == 1:
                # keep, but this changes the group's aliasing class
                self.dropped_groups['magnitude_TASK_SPECIFIC_WARNING'] = (
                    f'magnitude_code is single-level in '
                    f'{C.CONTEXT_LABEL[dead[0]]} only — group is now '
                    f'task-specific, check the context-aliasing QC')

    # -- per-trial assembly ---------------------------------------------------
    def _build_trial(self, trial):
        """Return list of (group_name, array, names) for one trial."""
        ev = trial['events']
        n = int(ev['trial_length'])
        ang = float(trial['angle'])
        ctx = int(trial['context'])
        # Sum coding: +1 saccade, -1 pursuit — keeps interaction columns
        # centered across tasks so context groups do not absorb the
        # task-averaged response the target/movement groups already carry.
        ctx_code = 1.0 if ctx == C.CTX_SACCADE else -1.0

        K_targ = self._event_kernels(n, ev.get('target_on'), self.target_win,
                                     self.n_target)
        K_move = self._event_kernels(n, ev.get('move_onset'), self.move_win,
                                     self.n_move)
        K_go = self._event_kernels(n, ev.get('go_cue'), self.go_win, self.n_go)
        K_rew = self._event_kernels(n, ev.get('reward'), self.reward_win,
                                    self.n_reward)

        parts = []
        parts.append(('target', K_targ,
                      [f'target_b{i}' for i in range(self.n_target)]))
        parts.append(('movement', K_move,
                      [f'move_b{i}' for i in range(self.n_move)]))
        if self.include_go:
            parts.append(('go', K_go, [f'go_b{i}' for i in range(self.n_go)]))
        parts.append(('reward', K_rew,
                      [f'reward_b{i}' for i in range(self.n_reward)]))

        # ---- direction bases -------------------------------------------------
        # TARGET-locked harmonics describe where the stimulus appeared;
        # MOVEMENT-locked ones where the eye went. They differ only on
        # backward-step pursuit trials (Rashbass step at angle+180), so the
        # target-locked set is re-signed by step_direction.
        dbasis = self._direction_basis(ang)
        stim_ang = ang
        if self.include_step_direction:
            sd = trial.get('step_direction')
            if sd is not None and np.isfinite(sd):
                stim_ang = float(sd)
        tbasis = (dbasis if stim_ang == ang else self._direction_basis(stim_ang))

        hnames = []
        for h in range(1, self.harmonics + 1):
            hnames += [f'cos{h}', f'sin{h}']

        # the shared direction columns: harmonics x (target, movement) kernels
        dir_cols, dir_names = [], []
        for K, tag, nb, basis in ((K_targ, 'targ', self.n_target, tbasis),
                                  (K_move, 'move', self.n_move, dbasis)):
            for d, hn in zip(basis, hnames):
                dir_cols.append(K * d)
                dir_names += [f'dir_{hn}_{tag}_b{i}' for i in range(nb)]
        D_shared = np.column_stack(dir_cols)

        if self.direction_param == 'shared_diff':
            # v2 parameterization: shared + task-signed copies of the SAME
            # columns. Closed-form per-task recovery lives in reparam.py.
            parts.append(('direction_shared', D_shared, dir_names))
            parts.append(('direction_diff', D_shared * ctx_code,
                          [nm.replace('dir_', 'dirdiff_', 1)
                           for nm in dir_names]))
        else:
            # per_task variant (WP7 comparison): each task its own copy.
            on_sac = 1.0 if ctx == C.CTX_SACCADE else 0.0
            parts.append(('direction_sac', D_shared * on_sac,
                          [nm.replace('dir_', 'dirsac_', 1)
                           for nm in dir_names]))
            parts.append(('direction_pur', D_shared * (1.0 - on_sac),
                          [nm.replace('dir_', 'dirpur_', 1)
                           for nm in dir_names]))

        # ---- context: task indicator, main effect + event-locked ------------
        ctx_cols = [np.full((n, 1), ctx_code), K_targ * ctx_code,
                    K_move * ctx_code]
        ctx_names = (['ctx_main']
                     + [f'ctx_targ_b{i}' for i in range(self.n_target)]
                     + [f'ctx_move_b{i}' for i in range(self.n_move)])
        parts.append(('context', np.column_stack(ctx_cols), ctx_names))

        # ---- sustained epoch (covers the pursuit ramp) ----------------------
        if self.include_sustained:
            K_sus = self._event_kernels(n, ev.get('target_on'),
                                        self.sustained_win, self.n_sustained)
            parts.append(('sustained', K_sus,
                          [f'sustained_b{i}' for i in range(self.n_sustained)]))
            sd_cols, sd_names = [], []
            for d, hn in zip(dbasis, hnames):
                sd_cols.append(K_sus * d)
                sd_names += [f'susdir_{hn}_b{i}'
                             for i in range(self.n_sustained)]
            parts.append(('sustained_direction',
                          np.column_stack(sd_cols), sd_names))

        # ---- catch-up saccades ----------------------------------------------
        if self.include_catchup:
            K_cu = self._multi_event_kernels(n, ev.get('catchup_saccades'),
                                             self.catchup_win, self.n_catchup)
            parts.append(('catchup_saccade', K_cu,
                          [f'catchup_b{i}' for i in range(self.n_catchup)]))
            if self.include_catchup_direction:
                cu_cols, cu_names = [], []
                for h in range(1, self.catchup_harmonics + 1):
                    for fn, tag in ((np.cos, 'cos'), (np.sin, 'sin')):
                        w = self._event_weights(ev.get('catchup_dirs'), h, fn)
                        cu_cols.append(self._multi_event_kernels(
                            n, ev.get('catchup_saccades'), self.catchup_win,
                            self.n_catchup, weights=w))
                        cu_names += [f'catchdir_{tag}{h}_b{i}'
                                     for i in range(self.n_catchup)]
                parts.append(('catchup_direction',
                              np.column_stack(cu_cols), cu_names))
            # NEW in v2: amplitude-weighted catch-up kernels (raw deg).
            if self.include_catchup_amplitude:
                amps = ev.get('catchup_amps')
                K_ca = self._multi_event_kernels(
                    n, ev.get('catchup_saccades'), self.catchup_win,
                    self.n_catchup, weights=amps)
                parts.append(('catchup_amplitude', K_ca,
                              [f'catchamp_b{i}'
                               for i in range(self.n_catchup)]))

        # ---- eye kinematics --------------------------------------------------
        mv = trial['movement']
        pos = [('eye_h', mv['eye_h']), ('eye_v', mv['eye_v'])]
        vel = [('eye_speed', mv['eye_speed']), ('eye_acc', mv['eye_acc'])]
        if self.include_signed_velocity and 'eye_vx' in mv:
            vel += [('eye_vx', mv['eye_vx']), ('eye_vy', mv['eye_vy'])]
        if self.include_signed_accel and 'eye_ax' in mv:
            vel += [('eye_ax', mv['eye_ax']), ('eye_ay', mv['eye_ay'])]

        eye_groups = [('eye_position', pos), ('eye_velocity', vel)]
        if self.include_target_velocity and 'target_vx' in mv:
            eye_groups.append(('target_velocity',
                               [('target_vx', mv['target_vx']),
                                ('target_vy', mv['target_vy'])]))
        if self.include_retinal_slip and 'slip_x' in mv:
            eye_groups.append(('retinal_slip',
                               [('slip_x', mv['slip_x']),
                                ('slip_y', mv['slip_y'])]))

        for gname, sigs in eye_groups:
            n_lags = (self.n_pos_lags if gname == 'eye_position'
                      else self.n_eye_lags)
            cols, names = [], []
            for nm, sig in sigs:
                s = self._zsafe(sig)
                if self.include_eye_lags:
                    cols.append(self._lagged(s, self.eye_lag_win, n_lags))
                    names += [f'{nm}_lag{i}' for i in range(n_lags)]
                else:
                    cols.append(s[:, None])
                    names.append(nm)
            parts.append((gname, np.column_stack(cols), names))

        # ---- stimulus offset -------------------------------------------------
        if self.include_stim_offset:
            K_off = self._event_kernels(n, ev.get('stim_offset'),
                                        self.stim_offset_win,
                                        self.n_stim_offset)
            parts.append(('stim_offset', K_off,
                          [f'stimoff_b{i}'
                           for i in range(self.n_stim_offset)]))

        # ---- jump: step direction x target-onset kernels --------------------
        if self.include_jump and 'jump' not in self.dropped_groups:
            j = trial.get('jump', np.nan)
            j_code = float(j) if j is not None and np.isfinite(j) else 0.0
            parts.append(('jump', K_targ * j_code,
                          [f'jump_targ_b{i}' for i in range(self.n_target)]))

        # ---- stimulus magnitude ---------------------------------------------
        if self.include_magnitude and 'magnitude' not in self.dropped_groups:
            mag = trial.get('magnitude_code', 0.0)
            mag = float(mag) if mag is not None and np.isfinite(mag) else 0.0
            parts.append(('magnitude',
                          np.column_stack([K_targ * mag, K_move * mag]),
                          [f'mag_targ_b{i}' for i in range(self.n_target)]
                          + [f'mag_move_b{i}' for i in range(self.n_move)]))

        # ---- delay: saccade memory period (target onset -> go cue) ----------
        if self.include_delay:
            delay = np.zeros((n, 1))
            t_on, g_on = ev.get('target_on'), ev.get('go_cue')
            if (ctx == C.CTX_SACCADE and np.isfinite(t_on)
                    and np.isfinite(g_on)):
                a, b = int(np.ceil(t_on)), int(np.floor(g_on))
                if b > a:
                    delay[max(a, 0):min(b, n), 0] = 1.0
            parts.append(('delay', delay, ['delay_boxcar']))

        # ---- drift: slow baseline over session time (shared) ----------------
        if self.include_drift and 'time_norm' in trial:
            tn = float(trial['time_norm'])
            vals = [np.cos((k + 1) * np.pi * tn) for k in range(self.n_drift)]
            drift = np.tile(np.asarray(vals), (n, 1))
            parts.append(('drift', drift,
                          [f'drift_b{i}' for i in range(self.n_drift)]))

        return parts

    # -- post-processing ------------------------------------------------------
    # Task-specific nuisance columns are centered within their own task's
    # frames so they cannot alias ctx_main (v1 measured r = 0.606 for the raw
    # delay boxcar). The map is name -> supporting task.
    _NUISANCE_SUPPORT = {
        'delay_boxcar': C.CTX_SACCADE,
    }
    _NUISANCE_GROUPS = ('delay',)

    def _center_nuisance_within_task(self, design_matrices, predictor_info,
                                     ctx_frames):
        """Center task-specific nuisance column(s) over their own task's
        frames, in place — making them orthogonal to that task's indicator."""
        big = np.vstack(design_matrices)
        for gname in self._NUISANCE_GROUPS:
            info = predictor_info.get(gname)
            if info is None:
                continue
            for off, name in enumerate(info['names']):
                task = self._NUISANCE_SUPPORT.get(name)
                if task is None:
                    continue
                col = info['start'] + off
                m = ctx_frames == task
                if not m.any():
                    continue
                mu = big[m, col].mean()
                if mu == 0.0:
                    continue
                row = 0
                for d in design_matrices:
                    nrow = d.shape[0]
                    sub = m[row:row + nrow]
                    d[sub, col] -= mu
                    row += nrow

    # -- public API -----------------------------------------------------------
    def build(self, glm_data_dict, monkey=None):
        """
        Build design matrices for every trial.

        Parameters
        ----------
        glm_data_dict : dict or list of trial dicts (converter output)
        monkey : 'S' | 'Y' | None
            When given, the data-driven degeneracy decisions are cross-checked
            against config expectations and a mismatch raises.

        Returns
        -------
        design_matrices : list of (n_bins, n_features) ndarray
        predictor_names : list of str
        predictor_info  : dict  group -> {'start', 'end', 'names'}
        """
        keys = (sorted(glm_data_dict.keys()) if isinstance(glm_data_dict, dict)
                else range(len(glm_data_dict)))
        trials = [glm_data_dict[k] for k in keys]

        # Per-monkey kernel windows (WP4e: Monkey Y's late saccades and fixed
        # ~1260 ms pursuit reward push the movement/reward kernel tails past
        # the 1350 ms window on one task only, which made the last bases
        # single-task and collinear with their context-signed copies — cond
        # 184-2289. Exceptions dict; S windows untouched.)
        if monkey is not None:
            if not self._move_win_explicit:
                self.move_win = C.MOVE_WIN_MS_BY_MONKEY.get(
                    monkey, C.MOVE_WIN_MS)
            if not self._reward_win_explicit:
                self.reward_win = C.REWARD_WIN_MS_BY_MONKEY.get(
                    monkey,
                    C.REWARD_WIN_SHARED_MS if self.shared_only
                    else C.REWARD_WIN_MS)
            if not self._n_eye_lags_explicit:
                self.n_eye_lags = C.EYE_LAG_BASES_BY_MONKEY.get(
                    monkey, C.N_EYE_LAG_BASES)
            if not self._n_pos_lags_explicit:
                self.n_pos_lags = C.POS_LAG_BASES_BY_MONKEY.get(
                    monkey, self.n_eye_lags)

        self.dropped_groups = {}
        self._scan_degeneracies(trials, monkey=monkey)

        design_matrices = []
        template = None
        contexts = []
        for i, trial in enumerate(trials):
            parts = self._build_trial(trial)
            if i == 0:
                template = parts
            design_matrices.append(np.column_stack([p[1] for p in parts]))
            contexts.append(np.full(design_matrices[-1].shape[0],
                                    trial['context']))

        predictor_names, predictor_info = [], {}
        cur = 0
        for group, arr, names in template:
            w = arr.shape[1]
            if len(names) != w:
                raise ValueError(
                    f'group {group}: {w} columns but {len(names)} names')
            predictor_info[group] = {'start': cur, 'end': cur + w,
                                     'names': names}
            predictor_names.extend(names)
            cur += w

        self._center_nuisance_within_task(design_matrices, predictor_info,
                                          np.concatenate(contexts))

        assert cur == design_matrices[0].shape[1], (
            f'predictor_info covers {cur} columns but design matrix has '
            f'{design_matrices[0].shape[1]}')
        widths = ([design_matrices[0].shape[1]]
                  + [d.shape[1] for d in design_matrices])
        assert len(set(widths)) == 1, 'design width differs across trials'

        if self.verbose:
            print(f'Trials: {len(design_matrices)} | Predictors: {cur} | '
                  f'direction_param: {self.direction_param}')
            for g, info in predictor_info.items():
                print(f'  {g:20s}: {info["end"] - info["start"]:3d}  '
                      f'(cols {info["start"]}-{info["end"] - 1})')
            for g, why in self.dropped_groups.items():
                print(f'  DROPPED {g}: {why}')
        return design_matrices, predictor_names, predictor_info
