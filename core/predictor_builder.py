"""
predictor_builder.py
====================
Cosine-bump basis functions and GLMPredictorBuilder class.

Used by session_pipeline.py to generate per-trial design matrices
from raw behavioural / movement data stored in glm_data_dict.
"""

import numpy as np
import matplotlib.pyplot as plt


def create_cosine_bumps(x, centers, widths):
    """
    Create raised cosine bumps (matching Harvey Lab tutorial).

    Parameters
    ----------
    x : ndarray, shape (n_samples,)
        Positions/times to evaluate bumps at
    centers : ndarray, shape (n_bumps,)
        Center positions of each bump
    widths : ndarray, shape (n_bumps,)
        Width of each bump

    Returns
    -------
    bases : ndarray, shape (n_samples, n_bumps)
        Evaluated basis functions
    """
    assert centers.shape == widths.shape, "Centers and widths must have same shape"
    x_reshape = np.asarray(x).reshape(-1)
    bases = np.zeros((x_reshape.shape[0], centers.shape[0]))

    for idx, cent in enumerate(centers):
        w = widths[idx]
        bases[:, idx] = (np.cos(2 * np.pi * (x_reshape - cent) / w) * 0.5 + 0.5) * \
                        (np.abs(x_reshape - cent) < w / 2)

    return bases


class GLMPredictorBuilder:
    """
    GLM predictor builder.

    Position-dependent (0-500):
    - Position-only: 10 bumps, restricted to y ∈ [0, pos_only_y_max] (default 350 cm).
      Bumps are placed only within the proximal zone so that the distal maze
      (y > 350 cm) is covered exclusively by choice_spatial, reducing collinearity.
    - Visual context × Pos: 20 (context mean × 10 + stim contrast × 10)
      * visual_context_pos: active on ALL visual trials (sum coding)
      * visual_stim_pos: +pos on 0deg, -pos on 90deg (contrast coding)
    - Audio context × Pos: 20 (context mean × 10 + stim contrast × 10, starts at y=50)
      * audio_context_pos: active on ALL audio trials (sum coding)
      * audio_stim_pos: +pos for left(-90), -pos for right(90) (contrast coding)
    - Choice place fields: 60 (2 directions × 30, truncated at turn)
    - Velocity × Pos: 20 (4 directions × 5)

    Congruent trials (context 0): both visual_context_pos AND audio_context_pos
    are simultaneously active, allowing the GLM to separate shared context signals.

    Temporally-dependent:
    - Turn onset: 8 (2 directions × 4 forward only, choice-specific)
    - Reward/ITI: 8 (4 back + 4 forward)

    Optional extended groups (disabled by default for Model 1 compatibility):
    - Peri-stimulus temporal (include_stim_temporal):
        4 groups × n_stim_bases cols each, anchored to cue onset.
        audio_context_temporal, audio_stim_temporal,
        visual_context_temporal, visual_stim_temporal
    - Pre-turn anticipatory (include_pre_turn):
        Anti-causal mirror of turn_onset; bases span [-pre_turn_window, 0].
        2 choices × n_pre_turn_bases = 2*n_pre_turn_bases cols.
    - Context × choice interaction (include_context_choice):
        Choice place fields conditioned on visual or audio context.
        visual_choice and audio_choice, each 2 × n_place_fields cols.

    Combined groups (off by default; recommended for Model 3+ runs):
    - include_combined_stimulus=True:
        Merges audio context×pos + audio stim×pos + audio temporal into a
        single ``audio`` group (2*n_pos_bases + 2*n_stim_bases cols).
        Merges visual equivalents into ``visual_all``.
        When active, include_stim_temporal is ignored.
    - include_combined_choice=True:
        Merges choice_spatial + pre_turn + turn_onset into a single ``choice``
        group (2*(n_place_fields + n_pre_turn_bases + n_turn_bases) cols).
        When active, include_spatial_choice / include_pre_turn /
        include_turn_onset flags are ignored.

    Total (Model 1 defaults): 147 predictors (146 + constant)
    """

    def __init__(
        self,
        sampling_rate=15.6,
        # Position parameters
        n_pos_bases=10,
        stem_y_max=500.0,
        pos_bump_width_ratio=4.0,
        audio_start_position=50.0,
        # position_only zone: bumps are placed only up to this y value so that
        # the distal maze is covered exclusively by choice_spatial predictors.
        pos_only_y_max=350.0,
        # Velocity
        n_vel_bases=5,
        # Spatial choice
        n_place_fields=30,
        # Temporal parameters
        n_turn_bases=4,
        turn_window=1.0,
        n_reward_bases=8,
        reward_window=2.0,
        temporal_bump_width_ratio=4.0,
        # Event derivation
        outcome_offset_from_iti=0,
        turn_y_thresh=505.0,
        turn_x_vel_thresh=20.0,
        visual_y_thresh=500.0,
        # Modular inclusion flags
        include_position_only=True,
        include_sound_position=True,
        include_visual_position=True,
        include_velocity=True,
        include_spatial_choice=True,
        include_turn_onset=True,
        include_reward=True,
        # ── Extended groups (off by default for Model 1 compatibility) ────
        # Peri-stimulus temporal bases
        include_stim_temporal=False,
        n_stim_bases=6,
        stim_window=1.0,
        # Pre-turn anticipatory bases
        include_pre_turn=False,
        n_pre_turn_bases=4,
        pre_turn_window=1.0,
        # Context × choice interaction place fields
        include_context_choice=False,
        # ── Combined groups (off by default; Model 3+ combined structure) ─
        # include_combined_stimulus : merges audio & visual positional + temporal
        #   each into one group named ``audio`` / ``visual_all``.
        # include_combined_choice   : merges choice_spatial, pre_turn, turn_onset
        #   into one group named ``choice``.
        include_combined_stimulus=False,
        include_combined_choice=False,
    ):
        self.sampling_rate = sampling_rate

        self.n_pos_bases = n_pos_bases
        self.stem_y_max = stem_y_max
        self.pos_bump_width_ratio = pos_bump_width_ratio
        self.audio_start_position = audio_start_position
        self.pos_only_y_max = pos_only_y_max

        self.n_vel_bases = n_vel_bases
        self.n_place_fields = n_place_fields

        self.n_turn_bases = n_turn_bases
        self.turn_window = turn_window
        self.n_reward_bases = n_reward_bases
        self.reward_window = reward_window
        self.temporal_bump_width_ratio = temporal_bump_width_ratio

        self.outcome_offset_from_iti = outcome_offset_from_iti
        self.turn_y_thresh = turn_y_thresh
        self.turn_x_vel_thresh = turn_x_vel_thresh
        self.visual_y_thresh = visual_y_thresh

        self.include_position_only = include_position_only
        self.include_sound_position = include_sound_position
        self.include_visual_position = include_visual_position
        self.include_velocity = include_velocity
        self.include_spatial_choice = include_spatial_choice
        self.include_turn_onset = include_turn_onset
        self.include_reward = include_reward

        self.include_stim_temporal = include_stim_temporal
        self.n_stim_bases = n_stim_bases
        self.stim_window = stim_window

        self.include_pre_turn = include_pre_turn
        self.n_pre_turn_bases = n_pre_turn_bases
        self.pre_turn_window = pre_turn_window

        self.include_context_choice = include_context_choice

        self.include_combined_stimulus = include_combined_stimulus
        self.include_combined_choice   = include_combined_choice

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _derive_events(self, trial, movement):
        """Derive missing event timings from trial and movement data."""
        events = trial.get("events", {}).copy()

        if "choice_frame" not in events:
            iti_start = events.get("iti_start")
            if iti_start is not None and iti_start > 0:
                events["choice_frame"] = iti_start - 1

        if "reward_frame" not in events:
            iti_start = events.get("iti_start")
            if iti_start is not None:
                events["reward_frame"] = iti_start + self.outcome_offset_from_iti

        if "turn_onset" not in events:
            y_pos = np.asarray(movement.get("y_position", [])).flatten()
            x_vel = np.asarray(movement.get("x_velocity", [])).flatten()
            choice = trial.get("choice", 0)

            turn_frame = None
            for i in range(len(y_pos)):
                if y_pos[i] > self.turn_y_thresh:
                    if i > 0:
                        if choice == 0 and x_vel[i] < -self.turn_x_vel_thresh:
                            turn_frame = i
                            break
                        elif choice == 1 and x_vel[i] > self.turn_x_vel_thresh:
                            turn_frame = i
                            break

            if turn_frame is not None:
                events["turn_onset"] = turn_frame

        return events

    def _get_y500_frame(self, movement):
        """Get frame where y_position crosses 500."""
        y_pos = np.asarray(movement.get("y_position", [])).flatten()
        for i, y in enumerate(y_pos):
            if y > self.visual_y_thresh:
                return i
        return len(y_pos)

    def _get_audio_start_frame(self, movement):
        """Get frame where y_position crosses audio_start_position (50)."""
        y_pos = np.asarray(movement.get("y_position", [])).flatten()
        for i, y in enumerate(y_pos):
            if y >= self.audio_start_position:
                return i
        return 0

    def _build_position_bases(self, movement, n_frames):
        """
        Build position basis functions (raised cosine along stem 0–500 cm).

        Two sets of bumps are returned in a single call:
          - Full-stem bases (0–stem_y_max): used by context and velocity predictors.
          - Proximal-only bases (0–pos_only_y_max): used by position_only predictor.
            Bumps are placed only within the proximal zone so that the distal maze
            (y > pos_only_y_max) is covered exclusively by choice_spatial.

        Returns
        -------
        bases_full     : ndarray (n_frames, n_pos_bases)  — full-stem bases
        bases_proximal : ndarray (n_frames, n_pos_bases)  — proximal-only bases
        names          : list of str
        """
        y_pos = np.asarray(movement["y_position"]).flatten()
        if y_pos.size < n_frames:
            y_pos = np.pad(y_pos, (0, n_frames - y_pos.size), constant_values=np.nan)

        x = y_pos[:n_frames].astype(float)
        n_b = self.n_pos_bases
        stem = self.stem_y_max

        if n_b == 1:
            centers_full = np.array([stem / 2])
            widths_full  = np.array([stem])
        else:
            centers_full = np.linspace(0, stem, n_b)
            spacing_full = stem / (n_b - 1)
            widths_full  = np.full(n_b, spacing_full * self.pos_bump_width_ratio)

        bases_full = create_cosine_bumps(x, centers_full, widths_full)
        bases_full = np.nan_to_num(bases_full, nan=0.0, posinf=0.0, neginf=0.0)

        # Proximal-only bases: bump centers span 0–pos_only_y_max
        prox = self.pos_only_y_max
        if n_b == 1:
            centers_prox = np.array([prox / 2])
            widths_prox  = np.array([prox])
        else:
            centers_prox = np.linspace(0, prox, n_b)
            spacing_prox = prox / (n_b - 1)
            widths_prox  = np.full(n_b, spacing_prox * self.pos_bump_width_ratio)

        bases_proximal = create_cosine_bumps(x, centers_prox, widths_prox)
        bases_proximal = np.nan_to_num(bases_proximal, nan=0.0, posinf=0.0, neginf=0.0)
        # Zero out any activation beyond the proximal zone
        bases_proximal[x > prox, :] = 0.0

        names = [f"pos_bump_{i+1}" for i in range(n_b)]
        return bases_full, bases_proximal, names

    def _build_velocity_bases(self, movement, n_frames):
        """Build velocity basis functions (5 bumps along stem 0-500)."""
        y_pos = np.asarray(movement["y_position"]).flatten()
        if y_pos.size < n_frames:
            y_pos = np.pad(y_pos, (0, n_frames - y_pos.size), constant_values=np.nan)

        x = y_pos[:n_frames].astype(float)
        n_b = self.n_vel_bases
        stem = self.stem_y_max

        if n_b == 1:
            centers = np.array([stem / 2])
            widths = np.array([stem])
        else:
            centers = np.linspace(0, stem, n_b)
            spacing = stem / (n_b - 1)
            widths = np.full(n_b, spacing * self.pos_bump_width_ratio)

        bases = create_cosine_bumps(x, centers, widths)
        bases = np.nan_to_num(bases, nan=0.0, posinf=0.0, neginf=0.0)

        return bases

    # ------------------------------------------------------------------
    # Per-trial predictor builders
    # ------------------------------------------------------------------

    def _build_position_only_trial(self, pos_bases_proximal):
        """
        Position-only (trialPhase) predictors.
        Uses proximal-only bases (0–pos_only_y_max) to avoid collinearity with
        choice_spatial, which covers the full 0–500 cm range.
        """
        names = [f"trialPhase_pos{i+1}" for i in range(pos_bases_proximal.shape[1])]
        return pos_bases_proximal.copy(), names

    def _build_sound_position_trial(self, n_frames, audio_start_frame, audio_stim,
                                     stim_end_frame, pos_bases):
        """
        Audio context × position predictors (starts at y=50, ends at y=500).

        Columns [0:n_b]   audio_context_pos: active on ALL audio trials (context mean).
        Columns [n_b:2*n_b] audio_stim_pos: +pos for left(-90), -pos for right(90) (contrast).
        """
        n_b = pos_bases.shape[1]
        pred = np.zeros((n_frames, 2 * n_b))
        names = [f"audio_context_pos{i+1}" for i in range(n_b)] + \
                [f"audio_stim_pos{i+1}" for i in range(n_b)]

        if np.isnan(audio_stim):
            return pred, names

        end_f = min(stim_end_frame, n_frames)
        if audio_start_frame >= end_f:
            return pred, names

        boxcar = np.zeros(n_frames)
        boxcar[audio_start_frame:end_f] = 1.0

        pred[:, :n_b] = boxcar[:, None] * pos_bases
        stim_sign = +1.0 if int(audio_stim) == 0 else -1.0
        pred[:, n_b:] = boxcar[:, None] * pos_bases * stim_sign

        return pred, names

    def _build_visual_position_trial(self, n_frames, stim_onset, visual_end_frame,
                                      visual_stim, pos_bases):
        """
        Visual context × position predictors (from sound_onset to y=500).

        Columns [0:n_b]   visual_context_pos: active on ALL visual trials (context mean).
        Columns [n_b:2*n_b] visual_stim_pos: +pos for 0deg, -pos for 90deg (contrast).
        """
        n_b = pos_bases.shape[1]
        pred = np.zeros((n_frames, 2 * n_b))
        names = [f"visual_context_pos{i+1}" for i in range(n_b)] + \
                [f"visual_stim_pos{i+1}" for i in range(n_b)]

        if np.isnan(visual_stim):
            return pred, names

        start_f = stim_onset
        end_f = min(visual_end_frame, n_frames)

        if end_f <= start_f:
            return pred, names

        boxcar = np.zeros(n_frames)
        boxcar[start_f:end_f] = 1.0

        pred[:, :n_b] = boxcar[:, None] * pos_bases
        stim_sign = +1.0 if (float(visual_stim) == -90.0 or float(visual_stim) == 0.0) else -1.0
        pred[:, n_b:] = boxcar[:, None] * pos_bases * stim_sign

        return pred, names

    def _build_velocity_trial(self, movement, n_frames, vel_bases):
        """
        Velocity × position predictors.
        4 directions (forward, reverse, leftward, rightward) × 5 position bumps = 20.
        """
        n_b = self.n_vel_bases

        y_vel = np.asarray(movement.get("y_velocity", [])).flatten()
        x_vel = np.asarray(movement.get("x_velocity", [])).flatten()

        if y_vel.size < n_frames:
            y_vel = np.pad(y_vel, (0, n_frames - y_vel.size), constant_values=0.0)
        if x_vel.size < n_frames:
            x_vel = np.pad(x_vel, (0, n_frames - x_vel.size), constant_values=0.0)

        y_vel = y_vel[:n_frames]
        x_vel = x_vel[:n_frames]

        forward_vel  = np.maximum(y_vel,  0)
        reverse_vel  = np.maximum(-y_vel, 0)
        rightward_vel = np.maximum(x_vel,  0)
        leftward_vel  = np.maximum(-x_vel, 0)

        pred = np.zeros((n_frames, 4 * n_b))
        pred[:, 0*n_b:1*n_b] = forward_vel[:, None]   * vel_bases
        pred[:, 1*n_b:2*n_b] = reverse_vel[:, None]   * vel_bases
        pred[:, 2*n_b:3*n_b] = rightward_vel[:, None] * vel_bases
        pred[:, 3*n_b:4*n_b] = leftward_vel[:, None]  * vel_bases

        names = [f"vel_forward_pos{i+1}"  for i in range(n_b)] + \
                [f"vel_reverse_pos{i+1}"  for i in range(n_b)] + \
                [f"vel_right_pos{i+1}"    for i in range(n_b)] + \
                [f"vel_left_pos{i+1}"     for i in range(n_b)]

        return pred, names

    def _build_spatial_choice_trial(self, movement, n_frames, choice, turn_frame):
        """
        Spatial choice place fields (0-500, TRUNCATED at turn_onset).
        Only active BEFORE the turn is executed.
        """
        n_pf = self.n_place_fields
        stem = self.stem_y_max

        if n_pf == 1:
            centers = np.array([stem / 2])
            widths = np.array([stem])
        else:
            centers = np.linspace(0, stem, n_pf)
            spacing = stem / (n_pf - 1)
            widths = np.full(n_pf, spacing * self.pos_bump_width_ratio)

        y_pos = np.asarray(movement["y_position"]).flatten()
        if y_pos.size < n_frames:
            y_pos = np.pad(y_pos, (0, n_frames - y_pos.size), constant_values=np.nan)

        x = y_pos[:n_frames].astype(float)

        place_bases = create_cosine_bumps(x, centers, widths)
        place_bases = np.nan_to_num(place_bases, nan=0.0)

        if turn_frame < n_frames:
            place_bases[turn_frame:, :] = 0.0

        pred = np.zeros((n_frames, 2 * n_pf))
        names = [f"place_left_{i+1}"  for i in range(n_pf)] + \
                [f"place_right_{i+1}" for i in range(n_pf)]

        if choice == 0:
            pred[:, :n_pf] = place_bases
        else:
            pred[:, n_pf:] = place_bases

        return pred, names

    def _build_turn_onset_trial(self, n_frames, turn_frame, choice):
        """
        Turn onset predictors (4 forward bumps only, choice-specific).
        Left choice trials get left_turn predictors, right choice get right_turn.
        """
        n_b = self.n_turn_bases
        window_frames = int(self.turn_window * self.sampling_rate)

        if n_b == 1:
            centers = np.array([window_frames / 2])
            widths = np.array([window_frames])
        else:
            centers = np.linspace(0, window_frames, n_b)
            spacing = window_frames / (n_b - 1)
            widths = np.full(n_b, spacing * self.temporal_bump_width_ratio)

        time_from_turn = np.arange(n_frames, dtype=float) - turn_frame
        time_from_turn[time_from_turn < 0] = np.nan

        temporal_bases = create_cosine_bumps(time_from_turn, centers, widths)
        temporal_bases = np.nan_to_num(temporal_bases, nan=0.0)

        pred = np.zeros((n_frames, 2 * n_b))
        names = [f"turn_left_t{i+1}"  for i in range(n_b)] + \
                [f"turn_right_t{i+1}" for i in range(n_b)]

        if choice == 0:
            pred[:, :n_b] = temporal_bases
        else:
            pred[:, n_b:] = temporal_bases

        return pred, names

    def _build_reward_trial(self, n_frames, reward_frame, outcome):
        """
        Outcome/ITI predictors — forward-only, outcome-specific.

        outcome : 1 = correct (rewarded), 0 = error (unrewarded)

        Returns n_reward_bases * 2 columns:
            columns 0..n_b-1   : correct bases
            columns n_b..2n_b-1: error bases
        """
        n_b = self.n_reward_bases
        window_frames = int(self.reward_window * self.sampling_rate)

        centers = np.linspace(0, window_frames, n_b)
        spacing = window_frames / max(n_b - 1, 1)
        widths = np.full(n_b, spacing * self.temporal_bump_width_ratio)

        time_from_reward = np.arange(n_frames, dtype=float) - reward_frame
        temporal_bases = create_cosine_bumps(time_from_reward, centers, widths)
        temporal_bases = np.nan_to_num(temporal_bases, nan=0.0)

        pred = np.zeros((n_frames, 2 * n_b))
        names = ([f"outcome_correct_t{i+1}" for i in range(n_b)] +
                 [f"outcome_error_t{i+1}"   for i in range(n_b)])

        if outcome == 1:
            pred[:, :n_b] = temporal_bases
        else:
            pred[:, n_b:] = temporal_bases

        return pred, names

    # ------------------------------------------------------------------
    # Extended predictor builders (Model 3+)
    # ------------------------------------------------------------------

    def _build_stim_temporal_trial(self, n_frames, audio_start_frame, stim_onset,
                                    audio_stim, visual_stim):
        """
        Peri-stimulus temporal basis functions anchored to cue onset.

        Produces four predictor groups, each n_stim_bases columns:

          audio_context_temporal  — sum coding, all audio trials
          audio_stim_temporal     — contrast coding (+/- by stim identity)
          visual_context_temporal — sum coding, all visual trials
          visual_stim_temporal    — contrast coding (+/- by stim identity)

        Bases are causal (0 → stim_window seconds post-cue) and evaluated
        on the frame-time axis relative to the cue onset frame.
        On trials where a modality is absent the corresponding columns are
        zero-filled.

        Returns
        -------
        audio_ctx  : ndarray (n_frames, n_stim_bases)
        names_ac   : list of str
        audio_stm  : ndarray (n_frames, n_stim_bases)
        names_as   : list of str
        visual_ctx : ndarray (n_frames, n_stim_bases)
        names_vc   : list of str
        visual_stm : ndarray (n_frames, n_stim_bases)
        names_vs   : list of str
        """
        n_b = self.n_stim_bases
        win = int(self.stim_window * self.sampling_rate)

        centers = np.linspace(0, win, n_b)
        spacing = win / max(n_b - 1, 1)
        widths  = np.full(n_b, spacing * self.temporal_bump_width_ratio)

        # ── Audio temporal ─────────────────────────────────────────────
        audio_ctx = np.zeros((n_frames, n_b))
        audio_stm = np.zeros((n_frames, n_b))
        if not np.isnan(audio_stim):
            t_audio = np.arange(n_frames, dtype=float) - audio_start_frame
            t_audio[t_audio < 0] = np.nan
            bases      = create_cosine_bumps(t_audio, centers, widths)
            bases      = np.nan_to_num(bases, nan=0.0)
            audio_ctx  = bases
            stim_sign  = +1.0 if int(audio_stim) == 0 else -1.0
            audio_stm  = bases * stim_sign

        # ── Visual temporal ────────────────────────────────────────────
        visual_ctx = np.zeros((n_frames, n_b))
        visual_stm = np.zeros((n_frames, n_b))
        if not np.isnan(visual_stim):
            t_vis  = np.arange(n_frames, dtype=float) - stim_onset
            t_vis[t_vis < 0] = np.nan
            bases      = create_cosine_bumps(t_vis, centers, widths)
            bases      = np.nan_to_num(bases, nan=0.0)
            visual_ctx = bases
            stim_sign  = +1.0 if (float(visual_stim) in (-90.0, 0.0)) else -1.0
            visual_stm = bases * stim_sign

        names_ac = [f"audio_ctx_t{i+1}"   for i in range(n_b)]
        names_as = [f"audio_stim_t{i+1}"  for i in range(n_b)]
        names_vc = [f"visual_ctx_t{i+1}"  for i in range(n_b)]
        names_vs = [f"visual_stim_t{i+1}" for i in range(n_b)]

        return (audio_ctx, names_ac, audio_stm, names_as,
                visual_ctx, names_vc, visual_stm, names_vs)

    def _build_pre_turn_trial(self, n_frames, turn_frame, choice):
        """
        Anti-causal pre-turn predictors — active BEFORE the turn.

        Mirror image of _build_turn_onset_trial: bases span
        [-pre_turn_window, 0] relative to turn_frame, capturing
        anticipatory / deliberation activity in the approach phase.

        Choice-specific (left / right): 2 × n_pre_turn_bases columns.

        Returns
        -------
        pred  : ndarray (n_frames, 2 * n_pre_turn_bases)
        names : list of str
        """
        n_b = self.n_pre_turn_bases
        win = int(self.pre_turn_window * self.sampling_rate)

        # Centers span [−win, 0] so bumps cover the pre-turn interval
        centers = np.linspace(-win, 0, n_b)
        spacing = win / max(n_b - 1, 1)
        widths  = np.full(n_b, spacing * self.temporal_bump_width_ratio)

        time_from_turn = np.arange(n_frames, dtype=float) - turn_frame
        # Keep only pre-turn frames; set post-turn to NaN so bumps are zero
        t_pre = time_from_turn.copy()
        t_pre[t_pre >= 0] = np.nan

        bases = create_cosine_bumps(t_pre, centers, widths)
        bases = np.nan_to_num(bases, nan=0.0)

        pred  = np.zeros((n_frames, 2 * n_b))
        names = ([f"pre_turn_left_t{i+1}"  for i in range(n_b)] +
                 [f"pre_turn_right_t{i+1}" for i in range(n_b)])

        if choice == 0:
            pred[:, :n_b] = bases
        else:
            pred[:, n_b:] = bases

        return pred, names

    def _build_context_choice_trial(self, movement, n_frames, context,
                                     choice, turn_frame):
        """
        Context × choice interaction place fields.

        Captures context-specific deviations in choice-selective spatial
        activity, using indicator coding:

          visual_choice : 2 × n_place_fields cols — active on visual (ctx=1) trials
          audio_choice  : 2 × n_place_fields cols — active on audio  (ctx=2) trials

        The CG context (ctx=0) effect is the baseline captured by the
        existing choice_spatial group — no interaction column is needed.

        Place fields are built identically to choice_spatial (same y_pos
        data, same n_place_fields resolution) and truncated at turn_onset.

        Parameters
        ----------
        movement : dict — trial movement data (needs 'y_position')
        context  : int  — 0=CG, 1=visual, 2=audio

        Returns
        -------
        vis_pred   : ndarray (n_frames, 2 * n_place_fields)
        names_v    : list of str
        audio_pred : ndarray (n_frames, 2 * n_place_fields)
        names_a    : list of str
        """
        n_pf = self.n_place_fields
        stem = self.stem_y_max

        centers = np.linspace(0, stem, n_pf)
        spacing = stem / max(n_pf - 1, 1)
        widths  = np.full(n_pf, spacing * self.pos_bump_width_ratio)

        y_pos = np.asarray(movement["y_position"]).flatten()
        if y_pos.size < n_frames:
            y_pos = np.pad(y_pos, (0, n_frames - y_pos.size), constant_values=np.nan)
        x = y_pos[:n_frames].astype(float)

        place_bases = create_cosine_bumps(x, centers, widths)
        place_bases = np.nan_to_num(place_bases, nan=0.0)
        if turn_frame < n_frames:
            place_bases[turn_frame:, :] = 0.0

        vis_pred   = np.zeros((n_frames, 2 * n_pf))
        audio_pred = np.zeros((n_frames, 2 * n_pf))
        names_v = ([f"vis_choice_left_{i+1}"    for i in range(n_pf)] +
                   [f"vis_choice_right_{i+1}"   for i in range(n_pf)])
        names_a = ([f"audio_choice_left_{i+1}"  for i in range(n_pf)] +
                   [f"audio_choice_right_{i+1}" for i in range(n_pf)])

        if context == 1:       # visual trial
            if choice == 0: vis_pred[:, :n_pf] = place_bases
            else:           vis_pred[:, n_pf:]  = place_bases
        elif context == 2:     # audio trial
            if choice == 0: audio_pred[:, :n_pf] = place_bases
            else:           audio_pred[:, n_pf:]  = place_bases

        return vis_pred, names_v, audio_pred, names_a

    # ------------------------------------------------------------------
    # Combined group builders (Model 3+ combined structure)
    # ------------------------------------------------------------------

    def _build_combined_audio_trial(self, n_frames, audio_start_frame, stim_onset,
                                     audio_stim, y500_frame, pos_bases_full):
        """
        Combined ``audio`` group: positional (context×pos + stim×pos) concatenated
        with peri-cue temporal bases.

        Columns: [audio_context_pos × n_pos_bases | audio_stim_pos × n_pos_bases
                  | audio_ctx_t × n_stim_bases    | audio_stim_t × n_stim_bases]

        Returns
        -------
        pred  : ndarray (n_frames, 2*n_pos_bases + 2*n_stim_bases)
        names : list of str
        """
        pos_pred, pos_names = self._build_sound_position_trial(
            n_frames, audio_start_frame, audio_stim, y500_frame, pos_bases_full
        )

        n_b = self.n_stim_bases
        win = int(self.stim_window * self.sampling_rate)
        centers = np.linspace(0, win, n_b)
        spacing = win / max(n_b - 1, 1)
        widths  = np.full(n_b, spacing * self.temporal_bump_width_ratio)

        audio_ctx_t = np.zeros((n_frames, n_b))
        audio_stm_t = np.zeros((n_frames, n_b))
        if not np.isnan(audio_stim):
            t_audio = np.arange(n_frames, dtype=float) - audio_start_frame
            t_audio[t_audio < 0] = np.nan
            bases      = create_cosine_bumps(t_audio, centers, widths)
            bases      = np.nan_to_num(bases, nan=0.0)
            audio_ctx_t = bases
            stim_sign   = +1.0 if int(audio_stim) == 0 else -1.0
            audio_stm_t = bases * stim_sign

        temp_names = ([f"audio_ctx_t{i+1}"  for i in range(n_b)] +
                      [f"audio_stim_t{i+1}" for i in range(n_b)])
        temp_pred  = np.hstack([audio_ctx_t, audio_stm_t])
        return np.hstack([pos_pred, temp_pred]), pos_names + temp_names

    def _build_combined_visual_trial(self, n_frames, stim_onset, y500_frame,
                                      visual_stim, pos_bases_full):
        """
        Combined ``visual_all`` group: positional (context×pos + stim×pos) concatenated
        with peri-cue temporal bases.

        Columns: [visual_context_pos × n_pos_bases | visual_stim_pos × n_pos_bases
                  | visual_ctx_t × n_stim_bases    | visual_stim_t × n_stim_bases]

        Returns
        -------
        pred  : ndarray (n_frames, 2*n_pos_bases + 2*n_stim_bases)
        names : list of str
        """
        pos_pred, pos_names = self._build_visual_position_trial(
            n_frames, stim_onset, y500_frame, visual_stim, pos_bases_full
        )

        n_b = self.n_stim_bases
        win = int(self.stim_window * self.sampling_rate)
        centers = np.linspace(0, win, n_b)
        spacing = win / max(n_b - 1, 1)
        widths  = np.full(n_b, spacing * self.temporal_bump_width_ratio)

        visual_ctx_t = np.zeros((n_frames, n_b))
        visual_stm_t = np.zeros((n_frames, n_b))
        if not np.isnan(visual_stim):
            t_vis  = np.arange(n_frames, dtype=float) - stim_onset
            t_vis[t_vis < 0] = np.nan
            bases      = create_cosine_bumps(t_vis, centers, widths)
            bases      = np.nan_to_num(bases, nan=0.0)
            visual_ctx_t = bases
            stim_sign    = +1.0 if (float(visual_stim) in (-90.0, 0.0)) else -1.0
            visual_stm_t = bases * stim_sign

        temp_names = ([f"visual_ctx_t{i+1}"  for i in range(n_b)] +
                      [f"visual_stim_t{i+1}" for i in range(n_b)])
        temp_pred  = np.hstack([visual_ctx_t, visual_stm_t])
        return np.hstack([pos_pred, temp_pred]), pos_names + temp_names

    def _build_combined_choice_trial(self, movement, n_frames, choice, turn_frame):
        """
        Combined ``choice`` group: choice place fields + pre-turn anticipatory +
        turn-onset temporal, all concatenated into a single predictor group.

        Column order:
          choice_spatial (2 × n_place_fields)
          pre_turn       (2 × n_pre_turn_bases)
          turn_onset     (2 × n_turn_bases)

        Returns
        -------
        pred  : ndarray (n_frames, 2*(n_place_fields + n_pre_turn_bases + n_turn_bases))
        names : list of str
        """
        spatial_pred, spatial_names = self._build_spatial_choice_trial(
            movement, n_frames, choice, turn_frame
        )

        if turn_frame < n_frames:
            turn_pred,     turn_names     = self._build_turn_onset_trial(n_frames, turn_frame, choice)
            pre_turn_pred, pre_turn_names = self._build_pre_turn_trial(n_frames, turn_frame, choice)
        else:
            n_t = self.n_turn_bases
            n_p = self.n_pre_turn_bases
            turn_pred = np.zeros((n_frames, 2 * n_t))
            turn_names = ([f"turn_left_t{i+1}"      for i in range(n_t)] +
                          [f"turn_right_t{i+1}"     for i in range(n_t)])
            pre_turn_pred = np.zeros((n_frames, 2 * n_p))
            pre_turn_names = ([f"pre_turn_left_t{i+1}"  for i in range(n_p)] +
                              [f"pre_turn_right_t{i+1}" for i in range(n_p)])

        combined = np.hstack([spatial_pred, pre_turn_pred, turn_pred])
        names    = spatial_names + pre_turn_names + turn_names
        return combined, names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_design_matrices(self, glm_data_dict):
        """
        Build design matrices for all trials.

        Parameters
        ----------
        glm_data_dict : dict or list
            Dictionary/list indexed by trial number

        Returns
        -------
        design_matrices : list of ndarray
        predictor_names : list of str
        predictor_info : dict
        """
        if isinstance(glm_data_dict, dict):
            trial_indices = sorted(glm_data_dict.keys())
        else:
            trial_indices = range(len(glm_data_dict))

        design_matrices = []
        all_group_parts = []

        for idx, trial_idx in enumerate(trial_indices):
            trial_data = glm_data_dict[trial_idx]

            movement = trial_data.get("movement", {})
            events_raw = trial_data.get("events", {})

            if "trial_length" in events_raw:
                n_frames = events_raw["trial_length"]
            else:
                neural_dict = trial_data.get("neural", {})
                if neural_dict:
                    first_metric = next(iter(neural_dict.values()))
                    n_frames = first_metric.shape[0]
                else:
                    raise ValueError(f"Trial {trial_idx} missing trial_length")

            trial = {
                "trial_length": n_frames,
                "events": events_raw,
                "audio_stim":  trial_data.get("Audio_Stim", np.nan),
                "visual_stim": trial_data.get("Visual_Stim", np.nan),
                "choice":  trial_data.get("choice", 0),
                "outcome": trial_data.get("outcome", 0),
                "context": trial_data.get("context", -1),
            }

            events = self._derive_events(trial, movement)

            stim_onset        = events.get("sound_onset", 0)
            audio_start_frame = self._get_audio_start_frame(movement)
            turn_frame        = events.get("turn_onset", n_frames - 1)
            reward_frame      = events.get("reward_frame", n_frames - 1)
            y500_frame        = self._get_y500_frame(movement)

            pos_bases_full, pos_bases_proximal, _ = self._build_position_bases(movement, n_frames)
            vel_bases = self._build_velocity_bases(movement, n_frames) if self.include_velocity else None

            trial_parts = []

            if self.include_position_only:
                # Use proximal-only bases to decouple from choice_spatial
                pred, names = self._build_position_only_trial(pos_bases_proximal)
                trial_parts.append(("position_only", pred, names))

            # ── Stimulus groups ───────────────────────────────────────────
            if self.include_combined_stimulus:
                # Merge positional + temporal into single audio / visual_all groups.
                # include_stim_temporal is ignored when combined mode is active.
                if self.include_sound_position:
                    pred, names = self._build_combined_audio_trial(
                        n_frames, audio_start_frame, stim_onset,
                        trial["audio_stim"], y500_frame, pos_bases_full
                    )
                    trial_parts.append(("audio", pred, names))
                if self.include_visual_position:
                    pred, names = self._build_combined_visual_trial(
                        n_frames, stim_onset, y500_frame,
                        trial["visual_stim"], pos_bases_full
                    )
                    trial_parts.append(("visual_all", pred, names))
            else:
                if self.include_sound_position:
                    pred, names = self._build_sound_position_trial(
                        n_frames, audio_start_frame, trial["audio_stim"], y500_frame, pos_bases_full
                    )
                    trial_parts.append(("sound", pred, names))
                if self.include_visual_position:
                    pred, names = self._build_visual_position_trial(
                        n_frames, stim_onset, y500_frame, trial["visual_stim"], pos_bases_full
                    )
                    trial_parts.append(("visual", pred, names))
                if self.include_stim_temporal:
                    (ac, nac, as_, nas,
                     vc, nvc, vs, nvs) = self._build_stim_temporal_trial(
                        n_frames, audio_start_frame, stim_onset,
                        trial["audio_stim"], trial["visual_stim"]
                    )
                    trial_parts.append(("audio_context_temporal",  ac,  nac))
                    trial_parts.append(("audio_stim_temporal",     as_, nas))
                    trial_parts.append(("visual_context_temporal", vc,  nvc))
                    trial_parts.append(("visual_stim_temporal",    vs,  nvs))

            if self.include_velocity and vel_bases is not None:
                pred, names = self._build_velocity_trial(movement, n_frames, vel_bases)
                trial_parts.append(("velocity", pred, names))

            # ── Choice groups ─────────────────────────────────────────────
            if self.include_combined_choice:
                # Merge spatial + pre-turn + turn-onset into single ``choice`` group.
                # include_spatial_choice / include_pre_turn / include_turn_onset ignored.
                pred, names = self._build_combined_choice_trial(
                    movement, n_frames, trial["choice"], turn_frame
                )
                trial_parts.append(("choice", pred, names))
            else:
                if self.include_spatial_choice:
                    pred, names = self._build_spatial_choice_trial(
                        movement, n_frames, trial["choice"], turn_frame
                    )
                    trial_parts.append(("choice_spatial", pred, names))
                if self.include_turn_onset and turn_frame < n_frames:
                    pred, names = self._build_turn_onset_trial(n_frames, turn_frame, trial["choice"])
                    trial_parts.append(("turn_onset", pred, names))
                if self.include_pre_turn and turn_frame < n_frames:
                    pred, names = self._build_pre_turn_trial(n_frames, turn_frame, trial["choice"])
                    trial_parts.append(("pre_turn", pred, names))

            # ── Context × choice interaction (always separate) ────────────
            if self.include_context_choice:
                vp, nv, ap, na = self._build_context_choice_trial(
                    movement, n_frames, trial["context"], trial["choice"], turn_frame
                )
                trial_parts.append(("visual_choice", vp, nv))
                trial_parts.append(("audio_choice",  ap, na))

            if self.include_reward and reward_frame < n_frames:
                pred, names = self._build_reward_trial(n_frames, reward_frame, trial["outcome"])
                trial_parts.append(("outcome", pred, names))

            if idx == 0:
                all_group_parts = trial_parts

            trial_preds = [part[1] for part in trial_parts]
            design_trial = np.column_stack(trial_preds) if trial_preds else np.zeros((n_frames, 0))
            design_matrices.append(design_trial)

        predictor_names = []
        predictor_info = {}
        current_idx = 0

        for group_name, pred, names in all_group_parts:
            n_pred = pred.shape[1]
            predictor_info[group_name] = {
                "start": current_idx,
                "end":   current_idx + n_pred,
                "names": names,
            }
            predictor_names.extend(names)
            current_idx += n_pred

        return design_matrices, predictor_names, predictor_info

    def plot_predictors(self, glm_data_dict, design_matrices, predictor_info,
                        start_trial=0, n_trials=5, figsize=(24, 22),
                        neural_metric='deconv', n_example_neurons=15,
                        example_neuron_ids=None):
        """
        Plot all predictors for consecutive trials in frame-time view.

        Parameters
        ----------
        start_trial : int
        n_trials : int
        neural_metric : str
        n_example_neurons : int
        example_neuron_ids : array-like or None
        """
        trial_indices = list(range(start_trial, min(start_trial + n_trials, len(design_matrices))))
        n_trials = len(trial_indices)

        n_rows = 10 if self.include_velocity else 9
        fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=False)

        first_neural = glm_data_dict[trial_indices[0]].get('neural', {}).get(neural_metric)
        if first_neural is not None:
            n_neurons_total = first_neural.shape[1]
            if example_neuron_ids is None:
                rng = np.random.default_rng(seed=42)
                example_neuron_ids = np.sort(
                    rng.choice(n_neurons_total,
                               size=min(n_example_neurons, n_neurons_total),
                               replace=False)
                )

        frame_offset = 0
        trial_boundaries = [0]

        for trial_idx in trial_indices:
            trial_data    = glm_data_dict[trial_idx]
            design_trial  = design_matrices[trial_idx]
            movement      = trial_data['movement']
            events        = trial_data['events']

            n_frames   = design_trial.shape[0]
            frame_axis = np.arange(frame_offset, frame_offset + n_frames)
            frame_offset += n_frames
            trial_boundaries.append(frame_offset)

            y_pos      = np.asarray(movement['y_position']).flatten()[:n_frames]
            audio_stim = trial_data.get('Audio_Stim', np.nan)
            visual_stim = trial_data.get('Visual_Stim', np.nan)
            choice     = trial_data.get('choice', 0)
            outcome    = trial_data.get('outcome', 0)

            audio_start_frame = self._get_audio_start_frame(movement)
            y500_frame        = self._get_y500_frame(movement)
            choice_frame      = events.get('iti_start', n_frames) - 1
            reward_frame      = choice_frame + self.outcome_offset_from_iti

            x_vel = np.asarray(movement.get('x_velocity', [])).flatten()
            turn_frame = None
            for i in range(len(y_pos)):
                if y_pos[i] > self.turn_y_thresh:
                    if i > 0 and abs(x_vel[i]) > self.turn_x_vel_thresh:
                        turn_frame = i
                        break
            if turn_frame is None:
                turn_frame = choice_frame

            audio_start_global = frame_axis[0] + audio_start_frame
            y500_global  = frame_axis[0] + y500_frame if y500_frame < n_frames else frame_axis[-1]
            choice_global = frame_axis[0] + choice_frame
            reward_global = frame_axis[0] + reward_frame
            turn_global   = frame_axis[0] + turn_frame

            row_idx = 0

            # Row: Y-position
            ax = axes[row_idx]
            ax.plot(frame_axis, y_pos, 'k-', linewidth=1.5, alpha=0.7)
            ax.axhline(self.audio_start_position, color='orange', linestyle=':', alpha=0.3, linewidth=1)
            ax.axhline(self.stem_y_max, color='red', linestyle=':', alpha=0.3, linewidth=1)
            row_idx += 1

            # Row: Audio boxcar (raw)
            ax = axes[row_idx]
            if not np.isnan(audio_stim):
                boxcar = np.zeros(n_frames)
                if audio_start_frame < y500_frame:
                    boxcar[audio_start_frame:min(y500_frame, n_frames)] = 1.0 if int(audio_stim) == 0 else -1.0
                color = 'blue' if int(audio_stim) == 0 else 'red'
                ax.fill_between(frame_axis, 0, boxcar, color=color, alpha=0.3, step='mid')
                ax.plot(frame_axis, boxcar, color=color, linewidth=2, alpha=0.8)
            row_idx += 1

            # Row: Audio context × position
            ax = axes[row_idx]
            if 'sound' in predictor_info:
                info = predictor_info['sound']
                start, end = info['start'], info['end']
                preds = design_trial[:, start:end]
                n_b = preds.shape[1] // 2
                if not np.isnan(audio_stim):
                    for i in range(n_b):
                        ax.plot(frame_axis, preds[:, i],
                                color=plt.cm.Greys(0.35 + 0.45 * i / max(n_b - 1, 1)),
                                alpha=0.7, linewidth=1.2)
                    stim_color = plt.cm.Blues if int(audio_stim) == 0 else plt.cm.Reds
                    for i in range(n_b):
                        ax.plot(frame_axis, preds[:, n_b + i],
                                color=stim_color(0.3 + 0.6 * i / max(n_b - 1, 1)),
                                alpha=0.6, linewidth=1.0, linestyle='--')
            row_idx += 1

            # Row: Visual boxcar (raw)
            ax = axes[row_idx]
            if not np.isnan(visual_stim):
                boxcar = np.zeros(n_frames)
                boxcar[:min(y500_frame, n_frames)] = 1.0 if float(visual_stim) in [-90.0, 0.0] else -1.0
                color = 'blue' if float(visual_stim) in [-90.0, 0.0] else 'red'
                ax.fill_between(frame_axis, 0, boxcar, color=color, alpha=0.3, step='mid')
                ax.plot(frame_axis, boxcar, color=color, linewidth=2, alpha=0.8)
            row_idx += 1

            # Row: Visual context × position
            ax = axes[row_idx]
            if 'visual' in predictor_info:
                info = predictor_info['visual']
                start, end = info['start'], info['end']
                preds = design_trial[:, start:end]
                n_b = preds.shape[1] // 2
                if not np.isnan(visual_stim):
                    for i in range(n_b):
                        ax.plot(frame_axis, preds[:, i],
                                color=plt.cm.Greys(0.35 + 0.45 * i / max(n_b - 1, 1)),
                                alpha=0.7, linewidth=1.2)
                    stim_color = plt.cm.Blues if float(visual_stim) in [-90.0, 0.0] else plt.cm.Reds
                    for i in range(n_b):
                        ax.plot(frame_axis, preds[:, n_b + i],
                                color=stim_color(0.3 + 0.6 * i / max(n_b - 1, 1)),
                                alpha=0.6, linewidth=1.0, linestyle='--')
            row_idx += 1

            # Row: Velocity × position
            if self.include_velocity:
                ax = axes[row_idx]
                if 'velocity' in predictor_info:
                    info = predictor_info['velocity']
                    start, end = info['start'], info['end']
                    preds = design_trial[:, start:end]
                    n_b = preds.shape[1] // 4
                    for i in range(n_b):
                        ax.plot(frame_axis, preds[:, 0*n_b + i], color=plt.cm.Blues(0.4 + 0.5*i/(n_b-1)), alpha=0.5, linewidth=0.8)
                        ax.plot(frame_axis, preds[:, 1*n_b + i], color=plt.cm.Reds(0.4 + 0.5*i/(n_b-1)), alpha=0.5, linewidth=0.8)
                        ax.plot(frame_axis, preds[:, 2*n_b + i], color=plt.cm.Purples(0.4 + 0.5*i/(n_b-1)), alpha=0.5, linewidth=0.8)
                        ax.plot(frame_axis, preds[:, 3*n_b + i], color=plt.cm.Greens(0.4 + 0.5*i/(n_b-1)), alpha=0.5, linewidth=0.8)
                row_idx += 1

            # Row: Choice indicator (raw)
            ax = axes[row_idx]
            choice_line = np.ones(n_frames) * (1.0 if choice == 0 else -1.0)
            color = 'green' if choice == 0 else 'purple'
            ax.fill_between(frame_axis, 0, choice_line, color=color, alpha=0.3, step='mid')
            ax.plot(frame_axis, choice_line, color=color, linewidth=2, alpha=0.8)
            ax.axvline(choice_global, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
            row_idx += 1

            # Row: Choice place fields
            ax = axes[row_idx]
            if 'choice_spatial' in predictor_info:
                info = predictor_info['choice_spatial']
                start, end = info['start'], info['end']
                preds = design_trial[:, start:end]
                n_pf = preds.shape[1] // 2
                if choice == 0:
                    for i in range(n_pf):
                        ax.plot(frame_axis, preds[:, i], color=plt.cm.Greens(0.3 + 0.6*i/(n_pf-1)), alpha=0.4, linewidth=0.6)
                else:
                    for i in range(n_pf):
                        ax.plot(frame_axis, preds[:, n_pf + i], color=plt.cm.Purples(0.3 + 0.6*i/(n_pf-1)), alpha=0.4, linewidth=0.6)
                ax.axvline(turn_global, color='orange', linestyle=':', alpha=0.5, linewidth=2)
            row_idx += 1

            # Row: Turn & Reward events
            ax = axes[row_idx]
            if 'turn_onset' in predictor_info:
                info = predictor_info['turn_onset']
                start, end = info['start'], info['end']
                preds = design_trial[:, start:end]
                n_b = preds.shape[1] // 2
                if choice == 0:
                    for i in range(n_b):
                        ax.plot(frame_axis, preds[:, i], color=plt.cm.Greens(0.4 + 0.5*i/(n_b-1)), alpha=0.6, linewidth=1.0)
                else:
                    for i in range(n_b):
                        ax.plot(frame_axis, preds[:, n_b + i], color=plt.cm.Purples(0.4 + 0.5*i/(n_b-1)), alpha=0.6, linewidth=1.0)
                ax.axvline(turn_global, color='orange', linestyle='--', alpha=0.7, linewidth=2)
            if 'outcome' in predictor_info:
                info = predictor_info['outcome']
                start, end = info['start'], info['end']
                preds = design_trial[:, start:end]
                n_b = preds.shape[1] // 2
                for i in range(n_b):
                    ax.plot(frame_axis, preds[:, i], color=plt.cm.Greens(0.4 + 0.5*i/(max(n_b-1, 1))), alpha=0.7, linewidth=0.9, linestyle='--')
                for i in range(n_b):
                    ax.plot(frame_axis, preds[:, n_b + i], color=plt.cm.Reds(0.4 + 0.5*i/(max(n_b-1, 1))), alpha=0.7, linewidth=0.9, linestyle=':')
                ax.axvline(reward_global, color='green', linestyle='--', alpha=0.7, linewidth=2)
            row_idx += 1

            # Row: Neural activity
            ax = axes[row_idx]
            neural_data = trial_data.get('neural', {}).get(neural_metric)
            if neural_data is not None and example_neuron_ids is not None:
                traces = np.asarray(neural_data)[:n_frames, :][:, example_neuron_ids]
                n_shown = traces.shape[1]
                trace_std  = traces.std(axis=0)
                trace_mean = traces.mean(axis=0)
                traces_z = (traces - trace_mean) / (trace_std + 1e-8)
                spacing = 3.0
                for j in range(n_shown):
                    ax.plot(frame_axis, traces_z[:, j] + j * spacing, linewidth=0.8, alpha=0.85, color='black')
                ax.set_ylim(-spacing, n_shown * spacing)

        # Trial boundaries
        for boundary in trial_boundaries[1:-1]:
            for ax in axes:
                ax.axvline(boundary, color='black', linestyle='-', alpha=0.3, linewidth=2, zorder=100)

        # Axis labels
        row_idx = 0
        axes[row_idx].set_ylabel('Y Position', fontsize=11, fontweight='bold')
        axes[row_idx].set_title(
            f'Predictors: {n_trials} Consecutive Trials (T{trial_indices[0]}-T{trial_indices[-1]})',
            fontsize=13, fontweight='bold'
        )
        row_idx += 1
        axes[row_idx].set_ylabel('Audio\n(raw)', fontsize=11, fontweight='bold')
        axes[row_idx].set_ylim(-1.2, 1.2)
        axes[row_idx].axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        row_idx += 1
        axes[row_idx].set_ylabel('Audio Ctx × Pos\n(gray=ctx, dash=stim)', fontsize=11, fontweight='bold')
        row_idx += 1
        axes[row_idx].set_ylabel('Visual\n(raw)', fontsize=11, fontweight='bold')
        axes[row_idx].set_ylim(-1.2, 1.2)
        axes[row_idx].axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        row_idx += 1
        axes[row_idx].set_ylabel('Visual Ctx × Pos\n(gray=ctx, dash=stim)', fontsize=11, fontweight='bold')
        row_idx += 1
        if self.include_velocity:
            axes[row_idx].set_ylabel('Velocity × Pos\n(expanded)', fontsize=11, fontweight='bold')
            axes[row_idx].text(0.98, 0.95, 'Blue=Fwd, Red=Rev, Purple=Right, Green=Left',
                               transform=axes[row_idx].transAxes, fontsize=8, ha='right',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            row_idx += 1
        axes[row_idx].set_ylabel('Choice\n(raw)', fontsize=11, fontweight='bold')
        axes[row_idx].set_ylim(-1.2, 1.2)
        axes[row_idx].axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        row_idx += 1
        axes[row_idx].set_ylabel('Choice Fields\n(→turn)', fontsize=11, fontweight='bold')
        row_idx += 1
        axes[row_idx].set_ylabel('Turn + Outcome\n(temporal)', fontsize=11, fontweight='bold')
        axes[row_idx].text(0.98, 0.95, 'Solid=Turn, Dash=Correct, Dot=Error',
                           transform=axes[row_idx].transAxes, fontsize=8, ha='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        row_idx += 1
        n_shown_label = len(example_neuron_ids) if example_neuron_ids is not None else 0
        axes[row_idx].set_ylabel(f'Neural\n({neural_metric}, n={n_shown_label})', fontsize=11, fontweight='bold')
        axes[row_idx].set_yticks([])
        axes[row_idx].set_xlabel('Frame (concatenated)', fontsize=12)

        for i, trial_idx in enumerate(trial_indices):
            mid_frame = (trial_boundaries[i] + trial_boundaries[i + 1]) / 2
            axes[0].text(mid_frame, 1.05, f'T{trial_idx}',
                         transform=axes[0].get_xaxis_transform(),
                         ha='center', fontsize=10, fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        plt.tight_layout()
        plt.show()
