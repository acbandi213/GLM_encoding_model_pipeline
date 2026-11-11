"""
convert.py
==========
`NHPSessionConverter` — turn one macaque FEF session into the `glm_data_dict`
schema the mouse GLM pipeline already consumes.

Both tasks go into a SINGLE dict per probe (saccade trials tagged context=1,
pursuit context=2). That is what lets one pooled Poisson GLM be fit per neuron
and then evaluated on held-out trials split by task, which is the CPI.

Schema produced per trial (mirrors the mouse `glm_data_dict`):

    {
      'neural':   {'spikes': (n_bins, n_units) int counts},
      'movement': {'eye_h','eye_v','eye_speed','eye_acc'},   each (n_bins,)
      'events':   {'trial_length','target_on','go_cue','move_onset','reward'},
                  all in BIN indices relative to the window start,
      'angle':    float  target direction, deg (0 = right),
      'context':  int    1 = saccade, 2 = pursuit,
      'outcome':  int    1 = correct (only correct trials are kept),
      'choice':   int    0 = rightward hemifield, 1 = leftward,
      'trial_type': int  direction index 0..7,
      'Visual_Stim': float  = angle,
      'speed':    float  pursuit target speed, NaN for saccade,
      'distance': float  saccade eccentricity, NaN for pursuit,
    }

`choice`, `trial_type` and `Visual_Stim` exist because `stack_design_matrices`
copies those keys into frame-level metadata, which downstream code uses for
train/test balancing and the trial-averaged R2 splits. Mapping direction onto
them keeps the shared pipeline working unmodified:
  * `choice`      -> hemifield, so the default L/R balancing has two populated
                     levels (without it every context is skipped as unbalanced);
  * `trial_type`  -> 8-way direction index, used to balance folds across
                     task x direction (see nhp.train);
  * `Visual_Stim` -> angle, so trial-averaged R2 is computed against direction
                     rather than a constant.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from .matio import NHPMatFile, open_session


def _sacc_onsets(mat, block, trial):
    """Onset times (ms from START_TRIAL) of every detected saccade in a trial."""
    s = mat.trial_saccades(block, trial)
    return s[:, 0] if s.size else np.empty(0)


def _sacc_vectors(mat, block, trial, eye_pos):
    """
    (onsets, directions_deg, amplitudes_deg) for every detected saccade.

    The `saccades` column stores [onset, offset] for each event, so the executed
    vector is just the eye displacement between those two samples. Only the
    onsets were being read; direction and amplitude were discarded, which is why
    `catchup_saccade` is currently direction-blind despite 4.35 catch-up
    saccades per pursuit trial in a direction-selective area.

    A detected saccade can carry an unusable offset (a non-finite value, or an
    index past the end of the eye trace — both occur, e.g. on 0137a). Its ONSET
    is still valid and the plain `catchup_saccade` kernels only need onsets, so
    the event is kept with a NaN direction rather than dropped: that preserves
    the existing group unchanged, keeps the three arrays index-aligned, and lets
    `_multi_event_kernels` skip the event in the direction-weighted group via
    its finite-weight mask.
    """
    s = mat.trial_saccades(block, trial)
    if s.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    n = eye_pos.shape[0]
    onsets, dirs, amps = [], [], []
    for on, off in s:
        if not np.isfinite(on):
            continue
        onsets.append(on)
        i = int(round(on))
        j = int(round(off)) if np.isfinite(off) else -1
        if 0 <= i < n and 0 <= j < n:
            d = eye_pos[j] - eye_pos[i]
            dirs.append(np.degrees(np.arctan2(d[1], d[0])))
            amps.append(float(np.hypot(d[0], d[1])))
        else:
            dirs.append(np.nan)
            amps.append(np.nan)
    return (np.asarray(onsets, float), np.asarray(dirs, float),
            np.asarray(amps, float))


def _scalar_col(tbl, name, n):
    """
    Read a table column as a float array of length n.

    Several event columns are stored as per-trial cells (e.g. TARG_ON holds a
    (2,1) array of [onset, offset]); take the first finite entry in that case.
    """
    v = tbl[name]
    if isinstance(v, list):
        out = np.full(n, np.nan)
        for i, x in enumerate(v):
            a = np.asarray(x, dtype=float).ravel()
            a = a[np.isfinite(a)]
            if a.size:
                out[i] = a[0]
        return out
    return np.asarray(v, dtype=float).ravel()


def _hemifield(angle_deg):
    """
    Binary right/left target hemifield, used as the `choice` analogue.

    Stated explicitly rather than as sign(cos(angle)) because the two vertical
    directions sit exactly on the boundary, where the answer would otherwise be
    decided by the floating-point sign of cos(pi/2) (~+6e-17) and cos(3pi/2)
    (~-1.8e-16). Splitting them one each keeps the two levels balanced 4-4 over
    the 8 directions, which matters if the default `choice` balancing is used.

    Returns 0 = right hemifield, 1 = left.
    """
    a = float(angle_deg) % 360.0
    if a == 90.0:            # straight up   -> right
        return 0
    if a == 270.0:           # straight down -> left
        return 1
    return 0 if (a < 90.0 or a > 270.0) else 1


def _bin_spikes(spike_ms, t0_ms, n_bins, bin_ms):
    """Bin one unit's spike times (ms from trial start) into the trial window."""
    if spike_ms.size == 0:
        return np.zeros(n_bins, dtype=np.int32)
    rel = (spike_ms - t0_ms) / bin_ms
    idx = np.floor(rel).astype(int)
    idx = idx[(idx >= 0) & (idx < n_bins)]
    return np.bincount(idx, minlength=n_bins).astype(np.int32)


def _resample_1ms(sig, t0_ms, n_bins, bin_ms):
    """
    Average a 1 ms-resolution signal within each bin of the trial window.

    Fast path (integer samples per bin, window fully inside the trial) reshapes
    and takes a row mean; otherwise falls back to per-bin slicing.
    """
    spb = int(round(bin_ms))
    a0 = int(round(t0_ms))
    if abs(bin_ms - spb) < 1e-9 and a0 >= 0 and a0 + n_bins * spb <= len(sig):
        return sig[a0:a0 + n_bins * spb].reshape(n_bins, spb).mean(axis=1)

    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        a = max(int(round(t0_ms + b * bin_ms)), 0)
        z = min(int(round(t0_ms + (b + 1) * bin_ms)), len(sig))
        if z > a:
            out[b] = sig[a:z].mean()
    return out


class NHPSessionConverter:
    """
    Convert one session `.mat` into per-probe `glm_data_dict`s.

    Parameters
    ----------
    path      : str   path to the session .mat
    bin_ms    : float spike-count bin width (default from config)
    win_pre   : float window start, ms relative to target onset (negative)
    win_post  : float window end,   ms relative to target onset
    """

    def __init__(self, path, bin_ms=None, win_pre=None, win_post=None,
                 verbose=True):
        self.path = str(path)
        self.bin_ms = C.BIN_MS if bin_ms is None else float(bin_ms)
        self.win_pre = C.WIN_PRE_MS if win_pre is None else float(win_pre)
        self.verbose = verbose

        self.mat: NHPMatFile = open_session(self.path)
        # WP4: the window is PER MONKEY. Y trials end 1488-1529 ms (pursuit)
        # and 1360-1420 ms (saccade) after target onset, so the S window of
        # 1500 ms silently dropped the whole Y pursuit block and ~40% of Y
        # saccade trials. See config.WIN_POST_MS_BY_MONKEY for the numbers.
        if win_post is None:
            mk = 'S' if self.mat.monkey == 'scrappy' else 'Y'
            self.win_post = float(C.WIN_POST_MS_BY_MONKEY.get(mk,
                                                              C.WIN_POST_MS))
        else:
            self.win_post = float(win_post)
        self.n_bins = int(round((self.win_post - self.win_pre) / self.bin_ms))
        self.stats = {}

    def close(self):
        self.mat.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- metadata ------------------------------------------------------------
    @property
    def sess_name(self):
        return self.mat.sess_name

    @property
    def monkey(self):
        return self.mat.monkey

    @property
    def probe_labels(self):
        return self.mat.probe_labels

    @property
    def sess_number(self):
        """'0136a' from 'kendra_scrappy_0136a'."""
        return self.sess_name.split('_')[-1]

    # -- task blocks ----------------------------------------------------------
    def task_blocks(self):
        """
        [(block_name, context_code)] for every task block in the file.

        Most sessions give [('mdir1', 1), ('purs1', 2)]. 0146a and 0147a record
        repeated blocks (mdir2, purs2) of the SAME task with the same xml and
        the same condition levels, interleaved in time; `C.POOL_TASK_BLOCKS`
        includes them. Ordered saccade-first to keep the trial numbering and
        the printed summaries comparable with single-block sessions.
        """
        present = self.mat.blocks()
        if not C.POOL_TASK_BLOCKS:
            return [(b, c) for b, c in ((C.SACCADE_BLOCK, C.CTX_SACCADE),
                                        (C.PURSUIT_BLOCK, C.CTX_PURSUIT))
                    if b in present]
        out = []
        for prefix, ctx in ((C.SACCADE_BLOCK_PREFIX, C.CTX_SACCADE),
                            (C.PURSUIT_BLOCK_PREFIX, C.CTX_PURSUIT)):
            out += [(b, ctx) for b in sorted(present) if b.startswith(prefix)]
        return out

    # -- per-block trial selection -------------------------------------------
    def _block_trials(self, block):
        """
        Return (keep_idx, fields) for one block after applying the trial filters.

        fields holds per-trial arrays already restricted to `keep_idx`.
        """
        tbl = self.mat.read_table(block)
        n = self.mat.n_trials(block)
        is_sacc = block.startswith(C.SACCADE_BLOCK_PREFIX)

        keep = np.array([r == C.RESULT_CORRECT for r in tbl['result']])
        n_correct = int(keep.sum())

        jump = (np.asarray(tbl['jump'], float) if not is_sacc
                else np.full(n, np.nan))
        n_jump = None
        if not is_sacc and C.PURSUIT_JUMP_KEEP is not None:
            keep &= (jump == C.PURSUIT_JUMP_KEEP)
            n_jump = int(keep.sum())

        targ_col = 'TARG_ON' if is_sacc else 'PURSUIT_TARG_ON'
        move_col = 'saccadeOnset' if is_sacc else 'pursuitOnset'
        # Each task's own stimulus-offset event: the memory-guided target flash
        # ending on saccade trials, the ramp ending on pursuit trials.
        off_col = 'TARG_OFF' if is_sacc else 'PURSUIT_TARG_OFF'

        target_on = _scalar_col(tbl, targ_col, n)
        go_cue = _scalar_col(tbl, 'FIX_OFF', n)
        move_on = _scalar_col(tbl, move_col, n)
        stim_off = _scalar_col(tbl, off_col, n)
        reward = _scalar_col(tbl, 'REWARD', n)
        end_tr = _scalar_col(tbl, 'END_TRIAL', n)
        angle = np.asarray(tbl['angle'], float)

        # Require the full analysis window to exist inside the trial and every
        # anchoring event to be finite.
        t0 = target_on + self.win_pre
        t1 = target_on + self.win_post
        ok_win = np.isfinite(target_on) & np.isfinite(move_on) & (t0 >= 0) & (t1 <= end_tr)
        keep &= ok_win
        n_win = int(keep.sum())

        # min reps per direction
        for a in np.unique(angle[keep]):
            m = keep & (angle == a)
            if m.sum() < C.MIN_REPS_PER_CONDITION:
                keep &= ~(angle == a)

        idx = np.where(keep)[0]
        eye_pos = [np.asarray(tbl['eyePos'][i], float) for i in idx]
        sacc = [_sacc_vectors(self.mat, block, int(i), ep)
                for i, ep in zip(idx, eye_pos)]
        fields = dict(
            angle=angle[idx],
            jump=jump[idx],
            target_on=target_on[idx],
            go_cue=go_cue[idx],
            move_on=move_on[idx],
            stim_off=stim_off[idx],
            reward=reward[idx],
            end_trial=end_tr[idx],
            speed=(np.asarray(tbl['speed'], float)[idx] if not is_sacc
                   else np.full(len(idx), np.nan)),
            distance=(np.asarray(tbl['distance'], float)[idx] if is_sacc
                      else np.full(len(idx), np.nan)),
            eyePos=eye_pos,
            eyeVel=[tbl['eyeVel'][i] for i in idx],
            eyeAcc=[tbl['eyeAcc'][i] for i in idx],
            # Onsets of every detected saccade (ms from START_TRIAL), plus the
            # executed direction and amplitude of each. On pursuit trials most
            # of these are catch-up saccades during the ramp.
            sacc_onsets=[s[0] for s in sacc],
            sacc_dirs=[s[1] for s in sacc],
            sacc_amps=[s[2] for s in sacc],
            # Step-ramp pursuit target speed (deg/s), NaN on saccade — used to
            # reconstruct the target-velocity vector and retinal slip.
            speed_ramp=(_scalar_col(tbl, 'speed', n)[idx] if not is_sacc
                        else np.full(len(idx), np.nan)),
            # Absolute session time of each trial (s, imec clock) for the drift
            # regressor. mdir1 and purs1 are recorded in separate, time-segregated
            # blocks, so slow recording drift is confounded with task.
            abs_time=_scalar_col(tbl, 'imec_sec', n)[idx],
        )
        self.stats[block] = dict(n_total=n, n_correct=n_correct,
                                 n_after_jump=n_jump, n_after_window=n_win,
                                 n_kept=len(idx))
        if self.verbose:
            msg = f'  {block}: {n} trials -> {n_correct} correct'
            if n_jump is not None:
                msg += f' -> {n_jump} jump=={C.PURSUIT_JUMP_KEEP}'
            msg += f' -> {n_win} window-fits -> {len(idx)} kept'
            print(msg)
        return idx, fields

    # -- main entry point -----------------------------------------------------
    def build(self, probe: int, skip_spikes: bool = False) -> dict:
        """
        Build the `glm_data_dict` for one probe (both tasks pooled).

        Returns dict {trial_index: trial_dict} with contiguous integer keys.

        ``skip_spikes=True`` (WP4) leaves `neural.spikes` all-zero and skips
        the per-trial HDF5 spike dereferences — the expensive part over SMB.
        For design-matrix QC only: with all-zero population rates BOTH artifact
        gates in `_reject_artifacts` are no-ops (sd = 0 disables the high-rate
        gate; alive baseline = 0 disables the dead gate), so the trial set is
        the full correct-trial set, NOT the production trial set. Never fit on
        a skip_spikes conversion.
        """
        if self.verbose:
            print(f'Converting {self.sess_name} probe {probe} '
                  f'({self.probe_labels[probe]}) ...')

        n_units = self.mat.n_units(probe)
        glm = {}
        k = 0
        pop_rate = []      # mean population rate per trial, for artifact rejection

        for block, ctx in self.task_blocks():
            idx, F = self._block_trials(block)
            is_sacc = block.startswith(C.SACCADE_BLOCK_PREFIX)

            for j, tr in enumerate(idx):
                t0 = F['target_on'][j] + self.win_pre
                nb = self.n_bins

                # ---- spikes ------------------------------------------------
                counts = np.zeros((nb, n_units), dtype=np.int32)
                if not skip_spikes:
                    spk = self.mat.trial_spiketimes(block, probe, int(tr))
                    for u in range(n_units):
                        counts[:, u] = _bin_spikes(spk[u], t0, nb, self.bin_ms)

                # ---- eye kinematics ---------------------------------------
                ep = np.asarray(F['eyePos'][j], float)
                ev = np.asarray(F['eyeVel'][j], float)
                ea = np.asarray(F['eyeAcc'][j], float)
                eye_h = _resample_1ms(ep[:, C.EYE_H], t0, nb, self.bin_ms)
                eye_v = _resample_1ms(ep[:, C.EYE_V], t0, nb, self.bin_ms)
                spd = np.hypot(ev[:, C.EYE_H], ev[:, C.EYE_V])
                acc = np.hypot(ea[:, C.EYE_H], ea[:, C.EYE_V])
                eye_sp = _resample_1ms(spd, t0, nb, self.bin_ms)
                eye_ac = _resample_1ms(acc, t0, nb, self.bin_ms)
                # SIGNED velocity components. eye_speed/eye_acc above are
                # unsigned magnitudes and so carry no direction information —
                # which is precisely what a smooth-pursuit response is tuned to.
                eye_vx = _resample_1ms(ev[:, C.EYE_H], t0, nb, self.bin_ms)
                eye_vy = _resample_1ms(ev[:, C.EYE_V], t0, nb, self.bin_ms)
                # SIGNED acceleration components. Pursuit initiation is an
                # acceleration transient with a direction; the unsigned eye_acc
                # above cannot carry it. Kept available for predictor testing.
                eye_ax = _resample_1ms(ea[:, C.EYE_H], t0, nb, self.bin_ms)
                eye_ay = _resample_1ms(ea[:, C.EYE_V], t0, nb, self.bin_ms)

                # ---- events, converted to bin indices ----------------------
                def to_bin(t_ms):
                    if not np.isfinite(t_ms):
                        return np.nan
                    return (t_ms - t0) / self.bin_ms

                ang = float(F['angle'][j])
                dir_idx = int(round(ang / 45.0)) % 8

                # ---- target velocity + retinal slip (pursuit sensory drive) --
                # Step-ramp target moves at constant `speed` in direction `angle`
                # from target onset onward (validated: eye velocity tracks it).
                # Retinal slip = target_vel - eye_vel is the canonical FEF pursuit
                # drive. Both are zero on saccade trials (no smooth target).
                tvx = np.zeros(nb)
                tvy = np.zeros(nb)
                slip_x = np.zeros(nb)
                slip_y = np.zeros(nb)
                if not is_sacc and np.isfinite(F['speed_ramp'][j]):
                    spd_r = float(F['speed_ramp'][j])
                    on = to_bin(F['target_on'][j])
                    live = np.arange(nb) >= (on if np.isfinite(on) else 0)
                    tvx[live] = spd_r * np.cos(np.radians(ang))
                    tvy[live] = spd_r * np.sin(np.radians(ang))
                    # slip only where the target is moving; elsewhere zero (not
                    # -eye_vel, which would smuggle eye velocity into saccade-like
                    # pre-ramp frames).
                    slip_x[live] = tvx[live] - eye_vx[live]
                    slip_y[live] = tvy[live] - eye_vy[live]
                events = {
                    'trial_length': nb,
                    'target_on': to_bin(F['target_on'][j]),
                    'move_onset': to_bin(F['move_on'][j]),
                    'reward': to_bin(F['reward'][j]),
                    # Each task's own stimulus offset: the memory-guided flash
                    # ending (saccade, ~117 ms after onset) or the ramp ending
                    # (pursuit, ~1017 ms after onset). Both fall inside the
                    # analysis window and neither was modelled.
                    'stim_offset': to_bin(F['stim_off'][j]),
                    # The go cue is only a distinct event in the memory-guided
                    # saccade task. In step-ramp pursuit FIX_OFF coincides with
                    # target onset (both medians 1019 ms), so a pursuit go
                    # kernel would be collinear with the target kernel.
                    'go_cue': to_bin(F['go_cue'][j]) if is_sacc else np.nan,
                }

                # Catch-up saccades: every detected saccade inside the window
                # EXCEPT the primary movement, which the `movement` kernel
                # already covers. Excluding it keeps the new group from being
                # collinear with `movement`. Stored as bin indices.
                so = np.asarray(F['sacc_onsets'][j], float)
                sd = np.asarray(F['sacc_dirs'][j], float)
                sa = np.asarray(F['sacc_amps'][j], float)
                if so.size:
                    m_on = F['move_on'][j]
                    if np.isfinite(m_on):
                        m = np.abs(so - m_on) > C.CATCHUP_EXCLUDE_MS
                        so, sd, sa = so[m], sd[m], sa[m]
                    sb = (so - t0) / self.bin_ms
                    m = (sb >= 0) & (sb < nb)
                    sb, sd, sa = sb[m], sd[m], sa[m]
                else:
                    sb = sd = sa = np.empty(0)
                events['catchup_saccades'] = sb
                # Executed direction (deg) and amplitude (deg) of each catch-up
                # saccade, kept aligned with `catchup_saccades`.
                events['catchup_dirs'] = sd
                events['catchup_amps'] = sa

                glm[k] = {
                    'neural': {'spikes': counts},
                    'movement': {'eye_h': eye_h, 'eye_v': eye_v,
                                 'eye_speed': eye_sp, 'eye_acc': eye_ac,
                                 'eye_vx': eye_vx, 'eye_vy': eye_vy,
                                 'eye_ax': eye_ax, 'eye_ay': eye_ay,
                                 'target_vx': tvx, 'target_vy': tvy,
                                 'slip_x': slip_x, 'slip_y': slip_y},
                    'events': events,
                    'angle': ang,
                    'abs_time': float(F['abs_time'][j]),
                    'context': int(ctx),
                    'outcome': 1,
                    'choice': _hemifield(ang),
                    'trial_type': dir_idx,
                    'Visual_Stim': ang,
                    'speed': float(F['speed'][j]),
                    'distance': float(F['distance'][j]),
                    # Step direction, +1 forward / -1 backward, NaN on saccade.
                    'jump': float(F['jump'][j]),
                    # The direction the target APPEARS at, as opposed to `angle`,
                    # the direction it then moves in. They differ only on
                    # backward-step pursuit trials, where the Rashbass step puts
                    # the target at angle+180 before it ramps through fixation.
                    'step_direction': (ang if is_sacc or not np.isfinite(F['jump'][j])
                                       else (ang if F['jump'][j] > 0
                                             else (ang + 180.0) % 360.0)),
                    'block': block,
                    'src_trial': int(tr),
                }
                pop_rate.append(counts.mean())
                k += 1

        glm = self._reject_artifacts(glm, np.asarray(pop_rate))
        # Normalized session time in [0, 1] for the drift basis: (t - min)/(range)
        # over the trials that survived, so the basis spans exactly the recording.
        if glm:
            at = np.array([t['abs_time'] for t in glm.values()], float)
            lo, span = at.min(), max(at.max() - at.min(), 1e-9)
            for t in glm.values():
                t['time_norm'] = float((t['abs_time'] - lo) / span)
            self._add_magnitude_code(glm)
        if self.verbose:
            ctxs = np.array([t['context'] for t in glm.values()])
            print(f'  -> {len(glm)} trials total '
                  f'({(ctxs == C.CTX_SACCADE).sum()} saccade, '
                  f'{(ctxs == C.CTX_PURSUIT).sum()} pursuit) x '
                  f'{self.n_bins} bins x {n_units} units')
        return glm

    @staticmethod
    def _add_magnitude_code(glm):
        """
        Add `magnitude_code`: each task's own stimulus magnitude, z-scored
        WITHIN task.

        The saccade magnitude is target eccentricity (`distance`, 10/20 deg) and
        the pursuit magnitude is target speed (`speed`, 10/20 deg/s) — the
        missing half of "target location", which no predictor group reads today.

        Standardising within task is what keeps the resulting column from
        aliasing the task indicator: the two variables are on different scales
        and in different units, so a raw column would be a task contrast rather
        than a magnitude contrast. Both happen to have two balanced levels, so
        this lands on +-1 in each task and the group is a clean magnitude
        contrast on both sides.
        """
        by_ctx = {}
        for t in glm.values():
            raw = t['distance'] if t['context'] == C.CTX_SACCADE else t['speed']
            t['_mag_raw'] = float(raw)
            by_ctx.setdefault(t['context'], []).append(float(raw))

        stats = {}
        for ctx, vals in by_ctx.items():
            v = np.asarray(vals, float)
            v = v[np.isfinite(v)]
            mu = v.mean() if v.size else 0.0
            sd = v.std() if v.size else 0.0
            stats[ctx] = (mu, sd if sd > 0 else 1.0)

        for t in glm.values():
            mu, sd = stats[t['context']]
            raw = t.pop('_mag_raw')
            t['magnitude_code'] = 0.0 if not np.isfinite(raw) else (raw - mu) / sd

    def _reject_artifacts(self, glm, pop_rate):
        """
        Drop trials that are recording artifacts, then renumber 0..n-1.

        Two independent gates on the population rate:
          * HIGH-rate artifacts: within each task, rate > ARTIFACT_SD from the
            task mean (the paper's rejection). Computed within task because the
            two tasks differ in mean rate, so a pooled threshold would mostly
            measure that task difference.
          * DEAD/LOW-rate trials (stability gate, DROP_DEAD_TRIALS): rate below
            DEAD_RATE_FRAC of a robust "alive" baseline. Removes the pursuit-block
            rundown tail that would otherwise deflate FDE_pursuit. Task-BLIND: it
            asks "was the recording alive?", not "which task/tuning is this".
        """
        if len(pop_rate) == 0:
            return glm
        keys = sorted(glm.keys())
        ctx = np.array([glm[k]['context'] for k in keys])

        bad_hi = np.zeros(len(pop_rate), dtype=bool)
        for c in np.unique(ctx):
            m = ctx == c
            mu, sd = pop_rate[m].mean(), pop_rate[m].std()
            if sd > 0:
                bad_hi[m] = np.abs(pop_rate[m] - mu) > C.ARTIFACT_SD * sd

        bad_dead = np.zeros(len(pop_rate), dtype=bool)
        if C.DROP_DEAD_TRIALS and len(pop_rate) > 4:
            # alive baseline = median of the liveliest half (robust to the dead
            # tail and to the between-task rate difference).
            alive = np.median(np.sort(pop_rate)[len(pop_rate) // 2:])
            bad_dead = pop_rate < C.DEAD_RATE_FRAC * alive

        bad = bad_hi | bad_dead
        self.stats['n_artifact'] = int(bad_hi.sum())
        self.stats['n_dead'] = int(bad_dead.sum())
        if self.verbose:
            n_dead_sac = int((bad_dead & (ctx == C.CTX_SACCADE)).sum())
            n_dead_pur = int((bad_dead & (ctx == C.CTX_PURSUIT)).sum())
            print(f'  artifact rejection: {bad_hi.sum()} high-rate; '
                  f'stability gate: {bad_dead.sum()} dead '
                  f'({n_dead_sac} sac, {n_dead_pur} pur)')
        if not bad.any():
            return glm
        kept = {}
        for new_i, old_i in enumerate([k for k in keys if not bad[k]]):
            kept[new_i] = glm[old_i]
        return kept

    # -- positive-control tuning stats ---------------------------------------
    def direction_tuning(self, glm, context=None, around=None, halfwidth_bins=10):
        """
        Per-unit preferred direction and selectivity index from binned counts.

        Used as the converter's positive control: FEF must show clear 8-way
        direction tuning. If it does not, binning or alignment is wrong.

        around : str or None
            Event name ('move_onset', 'target_on', ...) to lock the analysis
            window to, +-`halfwidth_bins` around it. None averages the WHOLE
            trial window.

            Use an event for any check meant to be sensitive to ALIGNMENT.
            Whole-trial averaging is nearly alignment-invariant: shifting the
            window slides which bins are averaged but barely moves each unit's
            trial-mean rate, so a whole-trial selectivity index stays high even
            if every event time were wrong. Only an event-locked window can
            detect that, because a misaligned anchor averages the response
            together with baseline and flattens the tuning curve.

        Returns (pref_dir_deg, selectivity_index), each (n_units,).
        SI = 1 - (mean over directions) / (max over directions), 0 = untuned.
        """
        trials = [t for t in glm.values()
                  if context is None or t['context'] == context]
        if not trials:
            raise ValueError(f'no trials for context={context}')
        by_dir = {d: [] for d in range(8)}
        for t in trials:
            S = t['neural']['spikes']
            if around is not None:
                e = t['events'].get(around)
                if e is None or not np.isfinite(e):
                    continue
                lo = max(int(round(e)) - halfwidth_bins, 0)
                hi = min(int(round(e)) + halfwidth_bins, S.shape[0])
                if hi <= lo:
                    continue
                S = S[lo:hi]
            by_dir[t['trial_type']].append(S.mean(axis=0))
        if not any(by_dir.values()):
            raise ValueError(f'no trials had a finite "{around}" event')
        dirs = sorted(d for d in by_dir if by_dir[d])
        R = np.stack([np.mean(by_dir[d], axis=0) for d in dirs])   # (n_dir, n_units)
        angles = np.radians(np.array(dirs) * 45.0)
        vx = (R * np.cos(angles)[:, None]).sum(0)
        vy = (R * np.sin(angles)[:, None]).sum(0)
        pref = np.degrees(np.arctan2(vy, vx)) % 360
        mx = R.max(0)
        si = np.where(mx > 0, 1.0 - R.mean(0) / np.where(mx > 0, mx, 1), 0.0)
        return pref, si
