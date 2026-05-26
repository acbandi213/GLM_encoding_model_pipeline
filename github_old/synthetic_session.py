"""
synthetic_session.py
====================
Generate a fully synthetic glm_data_dict with tightly controlled trial
structure for simulation and validation of the GLM pipeline.

No real behavioural data is required.  All trials are generated from scratch
with identical kinematics, so:
  - Every trial has the same y-position ramp (0 → 504 cm over n_pre_turn frames)
  - Turn onset is at exactly frame n_pre_turn in every trial
  - L/R choice balance is exact (50/50)
  - Outcome ratio is exact (correct_rate correct, rest error)
  - Stimulus identity is 50/50 within each context, independent of choice
  - ITI / reward onset is at a fixed frame (iti_frame) in every trial

This eliminates all kinematic and behavioural confounds that corrupt
simulations built from real sessions:
  - No trial-length variability → clean turn-aligned averages
  - No L/R kinematic asymmetry → symmetric ground-truth responses
  - No outcome-timing variability → outcome predictors cleanly post-turn

Usage
-----
    from synthetic_session import generate_synthetic_session
    from session_pipeline import (
        generate_predictors, stack_design_matrices, prepare_features_for_glm,
    )

    glm_data_dict = generate_synthetic_session(n_trials_per_context=100, seed=42)
    design_matrices, pred_names, pred_info_raw = generate_predictors(
        glm_data_dict, include_velocity=False
    )
    X, Y_placeholder, metadata, group_indices = stack_design_matrices(
        glm_data_dict, design_matrices, pred_info_raw, neural_metric='deconv'
    )
    pred_info, feature_group_size = prepare_features_for_glm(pred_info_raw)

Trial structure
---------------
Each trial has exactly n_pre_turn + n_post_turn = 200 frames:

  Frames 0 … n_pre_turn-1  (default 0–124):  approach phase
      y_position: linear ramp 0 → 504 cm   (just below 505 cm turn threshold)
      y_velocity: constant forward = 504 / (n_pre_turn-1) cm/frame
      x_velocity: 0 (straight approach)

  Frame n_pre_turn (default 125):  turn onset
      y_position: 510 cm  (above 505 cm threshold → turn detected here)
      x_velocity: −30 cm/frame (left) or +30 cm/frame (right)

  Frames n_pre_turn … n_pre_turn+n_post_turn-1 (default 125–199):  post-turn
      y_position: 510 cm
      x_velocity: ±30 cm/frame (choice direction)

  iti_frame (default 160):  ITI / reward onset
      Outcome temporal bases start here (35 frames after turn onset).

Derived event timings (via GLMPredictorBuilder._derive_events):
  sound_onset   = 0        → visual context predictor active from frame 0
  audio_start   ≈ frame 13 → where y_position ≥ 50 cm
  y500_frame    = 124      → where y_position > 500 cm; context predictors end here
  turn_onset    = 125      → explicitly stored in events dict
  choice_frame  = 139      → iti_start − 1
  reward_frame  = 140      → iti_start + outcome_offset_from_iti (= 0)
                             outcome bases active from +15 to +74 frames post-turn
"""

import numpy as np

# ---------------------------------------------------------------------------
# Kinematic constants (match predictor_builder defaults)
# ---------------------------------------------------------------------------
_N_PRE_TURN  = 125      # frames before turn onset (turn at exactly this frame)
_N_POST_TURN =  75      # frames after turn onset
_N_FRAMES    = _N_PRE_TURN + _N_POST_TURN   # 200 frames per trial

_Y_APPROACH_MAX = 504.0  # cm at frame n_pre_turn-1 (< 505 turn threshold, > 500)
_Y_POST_TURN    = 510.0  # cm after turn onset (> 505 threshold)
_X_VEL_TURN     =  30.0  # cm/frame lateral velocity after turn onset

_ITI_FRAME      = 140    # ITI onset (15 frames after turn, ~1.0 s at 15.6 Hz)
_SOUND_ONSET    =   0    # frame of visual stimulus onset


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_kinematics(choice, n_frames=_N_FRAMES, n_pre_turn=_N_PRE_TURN,
                     y_approach_max=_Y_APPROACH_MAX, y_post=_Y_POST_TURN,
                     x_vel_mag=_X_VEL_TURN):
    """
    Build synthetic movement arrays for a single trial.

    Parameters
    ----------
    choice : int  0 = left, 1 = right

    Returns
    -------
    dict with keys 'y_position', 'y_velocity', 'x_velocity'
    """
    n_post = n_frames - n_pre_turn

    # y_position: linear ramp 0 → y_approach_max over the approach phase,
    # then constant y_post (> 505 cm turn threshold) after turn onset.
    y_pos = np.empty(n_frames)
    y_pos[:n_pre_turn] = np.linspace(0.0, y_approach_max, n_pre_turn)
    y_pos[n_pre_turn:] = y_post

    # y_velocity: constant forward speed during approach, zero after turn.
    y_vel = np.zeros(n_frames)
    if n_pre_turn > 1:
        y_vel[:n_pre_turn] = y_approach_max / (n_pre_turn - 1)

    # x_velocity: zero during approach, ±x_vel_mag after turn.
    x_vel = np.zeros(n_frames)
    x_vel[n_pre_turn:] = -x_vel_mag if choice == 0 else +x_vel_mag

    return {'y_position': y_pos, 'y_velocity': y_vel, 'x_velocity': x_vel}


def _make_trial_assignments(n_trials, correct_rate, rng):
    """
    Return balanced choice, outcome, audio_stim, and visual_stim arrays.

    Rules
    -----
    - choices   : exact 50/50 left (0) / right (1)
    - outcomes  : round(correct_rate * n_trials) correct (1), rest error (0)
    - audio_stims  : 50/50  0.0 (left cue) / 90.0 (right cue)
    - visual_stims : 50/50 −90.0 (left pref) / 90.0 (right pref)
    All four arrays are independently shuffled.
    """
    half = n_trials // 2

    choices = np.array([0] * half + [1] * (n_trials - half), dtype=int)
    rng.shuffle(choices)

    n_correct = round(n_trials * correct_rate)
    outcomes  = np.array([1] * n_correct + [0] * (n_trials - n_correct), dtype=int)
    rng.shuffle(outcomes)

    audio_stims  = np.array([0.0]  * half + [90.0]  * (n_trials - half))
    visual_stims = np.array([-90.0] * half + [90.0] * (n_trials - half))
    rng.shuffle(audio_stims)
    rng.shuffle(visual_stims)

    return choices, outcomes, audio_stims, visual_stims


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_synthetic_session(
    n_trials_per_context=100,
    correct_rate=0.7,
    n_pre_turn=_N_PRE_TURN,
    n_post_turn=_N_POST_TURN,
    iti_frame=_ITI_FRAME,
    seed=42,
):
    """
    Generate a glm_data_dict with fully synthetic, balanced trials.

    Returns a dict compatible with:
      - GLMPredictorBuilder.build_design_matrices()
      - session_pipeline.stack_design_matrices()

    Parameters
    ----------
    n_trials_per_context : int   — trials per context (total = 3 ×)
    correct_rate         : float — fraction correct (default 0.70)
    n_pre_turn           : int   — frames before turn onset (default 125)
    n_post_turn          : int   — frames after turn onset  (default  75)
    iti_frame            : int   — frame of ITI/reward onset (default 140 = 15 frames
                               after turn onset, so the full outcome temporal profile
                               fits within the +75 post-turn plotting window)
    seed                 : int   — RNG seed for reproducibility

    Returns
    -------
    glm_data_dict : dict  trial_idx (int) -> trial_data (dict)

    Context coding
    --------------
    0 = CG     : both Audio_Stim and Visual_Stim present (congruent)
    1 = visual : only Visual_Stim present  (Audio_Stim = NaN)
    2 = audio  : only Audio_Stim  present  (Visual_Stim = NaN)

    Stimulus coding
    ---------------
    Audio_Stim  :  0.0 = left cue  (-90°),  90.0 = right cue (+90°)
    Visual_Stim : -90.0 = left pref,         90.0 = right pref
    (Matches the stim_sign convention in predictor_builder._build_*_trial)
    """
    assert iti_frame > n_pre_turn, \
        f'iti_frame ({iti_frame}) must be after turn onset ({n_pre_turn})'

    rng      = np.random.default_rng(seed)
    n_frames = n_pre_turn + n_post_turn

    glm_data_dict = {}
    trial_idx = 0

    for context in [0, 1, 2]:
        choices, outcomes, audio_stims, visual_stims = _make_trial_assignments(
            n_trials_per_context, correct_rate, rng
        )

        for i in range(n_trials_per_context):
            choice  = int(choices[i])
            outcome = int(outcomes[i])

            # Stimulus assignment per context
            if context == 0:       # CG: both modalities present
                a_stim = float(audio_stims[i])
                v_stim = float(visual_stims[i])
            elif context == 1:     # visual: no audio cue
                a_stim = np.nan
                v_stim = float(visual_stims[i])
            else:                  # audio: no visual cue
                a_stim = float(audio_stims[i])
                v_stim = np.nan

            movement = _make_kinematics(choice, n_frames, n_pre_turn)

            glm_data_dict[trial_idx] = {
                # Placeholder neural data (1 neuron, all zeros).
                # Replaced by SimulatedNeurons.build_all_neurons() output.
                'neural': {'deconv': np.zeros((n_frames, 1))},
                'movement': movement,
                'events': {
                    'trial_length': n_frames,
                    'sound_onset':  _SOUND_ONSET,
                    # Provide turn_onset explicitly so stack_design_matrices
                    # and GLMPredictorBuilder use exactly n_pre_turn — no
                    # kinematic detection variability.
                    'turn_onset':   n_pre_turn,
                    'iti_start':    iti_frame,
                },
                'Audio_Stim':  a_stim,
                'Visual_Stim': v_stim,
                'choice':      choice,
                'outcome':     outcome,
                'context':     context,
                'trial_type':  context * 4 + choice * 2 + outcome,
            }
            trial_idx += 1

    _print_summary(glm_data_dict, n_trials_per_context, n_frames, n_pre_turn,
                   iti_frame, correct_rate)
    return glm_data_dict


def _print_summary(glm_data_dict, n_per_ctx, n_frames, n_pre_turn,
                   iti_frame, correct_rate):
    """Print a formatted breakdown of the generated trial structure."""
    ctx_map = {0: 'CG', 1: 'visual', 2: 'audio'}
    ch_map  = {0: 'L',  1: 'R'}
    oc_map  = {0: 'error', 1: 'correct'}

    n_total = len(glm_data_dict)
    print(f'Generated {n_total} synthetic trials '
          f'({n_per_ctx} per context × 3 contexts)')
    print(f'  Frames / trial : {n_frames}  '
          f'(turn at frame {n_pre_turn},  ITI at frame {iti_frame})')
    print(f'  Total frames   : {n_total * n_frames:,}')
    print(f'  Correct rate   : {correct_rate:.0%}')
    print()
    print(f'  {"Context":8s}  {"Ch":2s}  {"Outcome":7s}   N')
    print('  ' + '-' * 30)
    for ctx in [0, 1, 2]:
        for ch in [0, 1]:
            for oc in [0, 1]:
                n = sum(
                    1 for d in glm_data_dict.values()
                    if d['context'] == ctx and d['choice'] == ch
                    and d['outcome'] == oc
                )
                print(f'  {ctx_map[ctx]:8s}  {ch_map[ch]}   '
                      f'{oc_map[oc]:7s}   {n}')
    print()
