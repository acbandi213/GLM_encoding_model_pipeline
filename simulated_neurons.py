"""
simulated_neurons.py
====================
SimulatedNeurons class — generate synthetic neural data from known ground-truth
weights for validating GLM fitting and analysis pipelines.

Each simulated neuron has non-zero weights in exactly ONE predictor group,
giving an unambiguous ground truth for the predictor-contribution analysis.

Usage
-----
    from session_pipeline import (
        extract_data, generate_predictors, stack_design_matrices,
        prepare_features_for_glm,
    )
    from simulated_neurons import SimulatedNeurons

    glm_data_dict = extract_data(mouse_ID, date)
    design_matrices, pred_names, pred_info_raw = generate_predictors(
        glm_data_dict, include_velocity=False
    )
    X, Y_real, metadata, group_indices = stack_design_matrices(
        glm_data_dict, design_matrices, pred_info_raw
    )
    pred_info, feature_group_size = prepare_features_for_glm(pred_info_raw)

    sim = SimulatedNeurons(pred_info, metadata, X, baseline_rate=0.004, seed=42)
    sim_data = sim.build_all_neurons()

    # Drop into fit_glm_single_session in place of real Y
    fit_glm_single_session(X, sim_data['Y_sim'] * 10, metadata,
                           feature_group_size, pred_info,
                           mouse_ID='simulated', date='v1')

Neuron catalogue  (one group per neuron — clean ground truth)
-------------------------------------------------------------
    N01  position_only      : Gaussian place field at mid-proximal zone (y ≈ 175 cm)
                              Active on ALL trial types (no context gating)
    N02  audio_context      : Gaussian spatial bump (~60 frames, peak ≈ turn-align −64)
                              on audio + CG trials; fires for ALL audio stim identities
    N03  audio_stim_L       : contrast-coded audio_stim, positive weights → selective
                              for Audio_Stim=0 (left-side tone); near-silent for R tone
    N04  audio_stim_R       : contrast-coded audio_stim, negative weights → selective
                              for Audio_Stim=90 (right-side tone); near-silent for L tone
    N05  visual_context     : identical shape to N02 on visual + CG trials;
                              fires for ALL visual stim identities
    N06  visual_stim_0      : contrast-coded visual_stim, positive weights → selective
                              for Visual_Stim=0°; near-silent for 90°
    N07  visual_stim_90     : contrast-coded visual_stim, negative weights → selective
                              for Visual_Stim=90°; near-silent for 0°
    N08  choice_left        : left-choice selective, distally-peaked place fields
    N09  choice_right       : right-choice selective, distally-peaked place fields
    N10  turn_left          : fires at left turn onset (temporal basis)
    N11  turn_right         : fires at right turn onset (temporal basis)
    N12  outcome_correct    : fires after reward (correct trials)
    N13  outcome_error      : fires after error (unrewarded trials)
    N14  null               : no tuning, pure Poisson noise

Total: 14 neurons  |  130 features (no velocity)
"""

import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns


# ---------------------------------------------------------------------------
# Predictor layout constants (must match session_pipeline / predictor_builder)
# Velocity is excluded from simulations (include_velocity=False).
# ---------------------------------------------------------------------------
N_POS = 10   # position_only, audio_context, visual_context, audio_stim, visual_stim
N_PF  = 30   # choice_spatial half (left or right place fields)
N_TB  =  4   # turn_onset half (left or right)
N_RB  =  6   # outcome half (correct or error)

# Context codes (matching session_pipeline)
CTX_CG     = 0
CTX_VISUAL = 1
CTX_AUDIO  = 2

# Choice codes
CHOICE_LEFT  = 0
CHOICE_RIGHT = 1

# Outcome codes
OUTCOME_ERROR   = 0
OUTCOME_CORRECT = 1


class SimulatedNeurons:
    """
    Generate synthetic Poisson-spiking neurons from known ground-truth GLM weights.

    Each neuron has weights in exactly ONE predictor group so that the
    predictor-contribution analysis has an unambiguous ground truth to validate against.

    Parameters
    ----------
    pred_info      : dict  — split predictor info from prepare_features_for_glm()
                             keys: position_only, audio_context, audio_stim,
                                   visual_context, visual_stim,
                                   choice_spatial, turn_onset, outcome
                             (velocity is NOT required)
    metadata       : dict  — frame-level metadata from stack_design_matrices()
                             keys: context, choice, outcome, audio_stim, visual_stim,
                                   trial_id, frame_in_trial
    X              : ndarray (n_frames, n_features) — z-scored design matrix
    baseline_rate  : float  — mean firing rate in events/frame (default 0.004 ≈ 0.06 Hz)
    seed           : int    — RNG seed for reproducibility
    """

    def __init__(self, pred_info, metadata, X, baseline_rate=0.004, seed=42):
        self.pred_info     = pred_info
        self.metadata      = metadata
        self.X             = X
        self.baseline_rate = baseline_rate
        self.w0_scalar     = np.log(baseline_rate)
        self.rng           = np.random.default_rng(seed)
        self.n_frames      = X.shape[0]
        self.n_features    = X.shape[1]

        # Populated by build_all_neurons()
        self.Y_sim         = None
        self.W_true        = None
        self.neuron_names  = None
        self.neuron_type   = None
        self.neuron_groups = None

        self._validate_pred_info()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_pred_info(self):
        required = ['position_only', 'audio_context', 'audio_stim',
                    'visual_context', 'visual_stim',
                    'choice_spatial', 'turn_onset', 'outcome']
        missing = [g for g in required if g not in self.pred_info]
        if missing:
            raise ValueError(f'pred_info is missing groups: {missing}')

    # ------------------------------------------------------------------
    # Weight construction helpers
    # ------------------------------------------------------------------

    def _make_weights(self, specs):
        """
        Build a weight vector (n_features,) from a human-readable spec dict.

        specs : dict  group_name -> weight_pattern
            weight_pattern:
                float / int         : fill ALL columns in group with this value
                ndarray             : must match group width exactly
                ('left',    float)  : fill first half  of choice_spatial / turn_onset
                ('right',   float)  : fill second half of choice_spatial / turn_onset
                ('correct', float)  : fill first half  of outcome
                ('error',   float)  : fill second half of outcome
        """
        w = np.zeros(self.n_features)

        for group, pattern in specs.items():
            if group not in self.pred_info:
                raise KeyError(f'Group "{group}" not in pred_info')
            s = self.pred_info[group]['start']
            e = self.pred_info[group]['end']
            n = e - s
            half = n // 2

            if isinstance(pattern, (int, float)):
                w[s:e] = float(pattern)

            elif isinstance(pattern, np.ndarray):
                if pattern.shape[0] != n:
                    raise ValueError(
                        f'Weight array for "{group}" has length {pattern.shape[0]}, '
                        f'expected {n}')
                w[s:e] = pattern

            elif isinstance(pattern, tuple):
                tag, val = pattern
                val = float(val)
                if tag == 'left':
                    w[s : s + half] = val
                elif tag == 'right':
                    w[s + half : e] = val
                elif tag == 'correct':
                    w[s : s + half] = val
                elif tag == 'error':
                    w[s + half : e] = val
                else:
                    raise ValueError(f'Unknown tag "{tag}" for group "{group}"')
            else:
                raise TypeError(
                    f'Unknown pattern type for "{group}": {type(pattern)}')

        return w

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def _simulate_additive(self, w_true, w0=None):
        """
        Generate Y from Poisson(exp(X @ w + w0)).

        Parameters
        ----------
        w_true : ndarray (n_features,)
        w0     : float or None  (defaults to self.w0_scalar)

        Returns
        -------
        Y : ndarray (n_frames,)  — integer spike counts
        """
        if w0 is None:
            w0 = self.w0_scalar
        log_rate = self.X @ w_true + w0
        log_rate = np.clip(log_rate, -10.0, 5.0)
        rate = np.exp(log_rate)
        return self.rng.poisson(rate).astype(float)

    # ------------------------------------------------------------------
    # Condition mask helpers
    # ------------------------------------------------------------------

    def _mask(self, context=None, choice=None, outcome=None):
        """Return boolean frame mask for a given condition combination."""
        m = np.ones(self.n_frames, dtype=bool)
        if context is not None:
            m &= (self.metadata['context'] == context)
        if choice is not None:
            m &= (self.metadata['choice'] == choice)
        if outcome is not None:
            m &= (self.metadata['outcome'] == outcome)
        return m

    # ------------------------------------------------------------------
    # Weight profile helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _distal_gradient(n_fields, peak_frac=0.85, sigma_frac=0.25, scale=1.0):
        """
        Gaussian weight profile across n_fields place fields, peaked at the
        distal end of the maze (near turn onset).

        Parameters
        ----------
        n_fields   : int   — number of place fields (e.g. 30 for choice_spatial half)
        peak_frac  : float — fractional position of peak (0=proximal, 1=distal)
        sigma_frac : float — width of Gaussian as fraction of track length
        scale      : float — overall weight magnitude

        Returns
        -------
        weights : ndarray (n_fields,)
        """
        x = np.linspace(0.0, 1.0, n_fields)
        w = np.exp(-0.5 * ((x - peak_frac) / sigma_frac) ** 2)
        w = w / w.max() * scale
        return w

    @staticmethod
    def _smooth(trace, sigma):
        """
        Apply Gaussian smoothing to a 1-D trace, ignoring NaNs.
        """
        if sigma <= 0:
            return trace
        from scipy.ndimage import gaussian_filter1d
        nan_mask = np.isnan(trace)
        filled   = trace.copy()
        filled[nan_mask] = 0.0
        weights  = (~nan_mask).astype(float)
        smoothed_vals    = gaussian_filter1d(filled,   sigma=sigma)
        smoothed_weights = gaussian_filter1d(weights,  sigma=sigma)
        with np.errstate(invalid='ignore'):
            out = smoothed_vals / smoothed_weights
        out[nan_mask] = np.nan
        return out

    # ------------------------------------------------------------------
    # Build all neurons
    # ------------------------------------------------------------------

    def build_all_neurons(self):
        """
        Build all 14 simulated neurons (N01–N14).

        Each neuron has non-zero weights in exactly ONE predictor group.
        This provides a clean, unambiguous ground truth for the
        predictor-contribution analysis.

        Neuron catalogue
        ----------------
            N01  position_only      : Gaussian place field, mid-proximal zone.
                                      position_only is active on ALL trials (no context
                                      gating) — the mouse always moves through the maze.
            N02  audio_context      : Gaussian bump (~60 frames, peak at turn-aligned
                                      ≈−64 frames) on audio + CG trials; fires for ALL
                                      audio stim identities (context selectivity only)
            N03  audio_stim_L       : audio_stim group, positive weights → selective for
                                      Audio_Stim=0 (L tone); near-silent for R tone
            N04  audio_stim_R       : audio_stim group, negative weights → selective for
                                      Audio_Stim=90 (R tone); near-silent for L tone
            N05  visual_context     : identical shape to N02 on visual + CG trials; fires
                                      for ALL visual stim identities
            N06  visual_stim_0      : visual_stim group, positive weights → selective for
                                      Visual_Stim=0°; near-silent for 90°
            N07  visual_stim_90     : visual_stim group, negative weights → selective for
                                      Visual_Stim=90°; near-silent for 0°
            N08  choice_left        : left-choice selective, distally-peaked (pre-turn)
            N09  choice_right       : right-choice selective, distally-peaked (pre-turn)
            N10  turn_left          : fires at left turn onset (temporal, forward only)
            N11  turn_right         : fires at right turn onset
            N12  outcome_correct    : fires from +15 frames post-turn (correct trials);
                                      ~60-frame bump within the +75 post-turn window
            N13  outcome_error      : same timing as N12 but on error trials
            N14  null               : pure Poisson noise, no weights

        Returns
        -------
        dict with keys:
            Y_sim         : ndarray (n_frames, 14)
            W_true        : ndarray (n_features, 14)
            neuron_names  : list of str
            neuron_type   : list of str  (all 'additive')
            neuron_groups : dict  group_label -> list of neuron names
        """
        S_STRONG = 0.8   # nominal weight scale; all vectors are rescaled by TARGET below

        # ── Position-only profile: Gaussian centred at mid-proximal zone ──────
        # N_POS bumps span 0–350 cm; peak_frac=0.50 → y ≈ 175 cm
        _pos_mid = self._distal_gradient(N_POS, peak_frac=0.50, sigma_frac=0.20,
                                         scale=S_STRONG)

        # ── Sensory (context / stim) Gaussian profiles ─────────────────────────
        # 10 bumps span y ≈ 0–504 cm.  peak_frac=0.50 → peak at y ≈ 252 cm
        # (turn-aligned ≈ −64 frames).  sigma_frac=0.20 → FWHM ≈ 58 frames,
        # creating a clean ~60-frame bump visible in the trial-average.
        # All four sensory neurons use the same SHAPE; absolute scale is set by
        # the TARGET normalization below, so S_STRONG here just controls shape.
        _ctx_bump  = self._distal_gradient(N_POS, peak_frac=0.50, sigma_frac=0.20,
                                           scale=S_STRONG)
        _stim_bump = self._distal_gradient(N_POS, peak_frac=0.50, sigma_frac=0.20,
                                           scale=S_STRONG)

        # ── Choice spatial profiles: distally-peaked (tight Gaussian near turn) ─
        _left_distal = np.concatenate([
            self._distal_gradient(N_PF, peak_frac=0.97, sigma_frac=0.12, scale=S_STRONG),
            np.zeros(N_PF),
        ])   # (60,)  left side only

        _right_distal = np.concatenate([
            np.zeros(N_PF),
            self._distal_gradient(N_PF, peak_frac=0.97, sigma_frac=0.12, scale=S_STRONG),
        ])   # (60,)  right side only

        # ── Neuron specs: one group per neuron ────────────────────────────────
        additive_specs = {

            # position_only is purely spatial (y_position), active on ALL trials.
            'N01_position_only': {
                'position_only': _pos_mid,
            },

            # audio_context: Gaussian bump peaked at y≈252 cm (turn-aligned ≈−64 frames).
            # Active on audio + CG trials (y≥50 cm gate); fires for ALL stim identities.
            'N02_audio_context': {
                'audio_context': _ctx_bump,
            },

            # audio_stim_L/R: contrast-coded (±1) predictor.
            # Positive weights → near-silent for −1 stim sign (R tone) at TARGET=4.
            # Negative weights → near-silent for +1 stim sign (L tone).
            # Both are normalized to max|X@w| = TARGET so peak rates match other neurons.
            'N03_audio_stim_L': {
                'audio_stim': _stim_bump,          # positive → selective for L tone
            },
            'N04_audio_stim_R': {
                'audio_stim': -_stim_bump,         # negative → selective for R tone
            },

            # visual_context: identical shape to audio_context on visual + CG trials.
            'N05_visual_context': {
                'visual_context': _ctx_bump,
            },

            # visual_stim_0/90: same logic as audio_stim_L/R.
            'N06_visual_stim_0': {
                'visual_stim': _stim_bump,         # positive → selective for 0°
            },
            'N07_visual_stim_90': {
                'visual_stim': -_stim_bump,        # negative → selective for 90°
            },

            # choice_spatial: distally-peaked place fields, zeroed after turn onset.
            'N08_choice_left': {
                'choice_spatial': _left_distal,
            },
            'N09_choice_right': {
                'choice_spatial': _right_distal,
            },

            # turn_onset: forward temporal bases after turn onset frame.
            'N10_turn_left': {
                'turn_onset': ('left', S_STRONG),
            },
            'N11_turn_right': {
                'turn_onset': ('right', S_STRONG),
            },

            # outcome: forward temporal bases after reward/ITI onset (+15 frames).
            'N12_outcome_correct': {
                'outcome': ('correct', S_STRONG),
            },
            'N13_outcome_error': {
                'outcome': ('error', S_STRONG),
            },

            # Null neuron — no weights, pure noise baseline.
            'N14_null': {},
        }

        # ── Build Y and W_true ─────────────────────────────────────────────────
        # Normalize each weight vector so that max|X @ w| = TARGET.
        # This equalises peak activity across groups regardless of how sparse
        # or dense the corresponding predictor columns are after z-scoring
        # (sparse predictors such as choice_spatial have much larger z-scored
        # values than dense ones such as audio_context, which would otherwise
        # produce vastly different peak rates before normalization).
        TARGET    = 6.0   # target peak log-rate contribution above w0
                          # → peak rate ≈ e^2.5 × baseline ≈ 12 × baseline
        all_names = list(additive_specs.keys())
        W_cols    = []
        Y_cols    = []

        for name in all_names:
            w_raw = self._make_weights(additive_specs[name])
            if w_raw.any():
                peak = np.max(np.abs(self.X @ w_raw))
                w    = w_raw * (TARGET / peak) if peak > 0 else w_raw
            else:
                w = w_raw   # null neuron: no weights, no rescaling
            W_cols.append(w)
            Y_cols.append(self._simulate_additive(w))

        Y_sim  = np.column_stack(Y_cols)
        W_true = np.column_stack(W_cols)

        neuron_type = ['additive'] * len(all_names)

        neuron_groups = {
            'position': ['N01_position_only'],
            'audio':    ['N02_audio_context', 'N03_audio_stim_L', 'N04_audio_stim_R'],
            'visual':   ['N05_visual_context', 'N06_visual_stim_0', 'N07_visual_stim_90'],
            'choice':   ['N08_choice_left', 'N09_choice_right'],
            'turn':     ['N10_turn_left', 'N11_turn_right'],
            'outcome':  ['N12_outcome_correct', 'N13_outcome_error'],
            'null':     ['N14_null'],
        }

        self.Y_sim         = Y_sim
        self.W_true        = W_true
        self.neuron_names  = all_names
        self.neuron_type   = neuron_type
        self.neuron_groups = neuron_groups

        n = len(all_names)
        print('SimulatedNeurons.build_all_neurons() complete')
        print(f'  Y_sim  : {Y_sim.shape}  (frames × neurons)')
        print(f'  W_true : {W_true.shape}  (features × neurons)')
        print(f'  {n} neurons — one per predictor group/sub-group')
        print(f'  Mean firing rate: {Y_sim.mean():.4f} events/frame')
        print(f'  Peak log-rate target: {TARGET:.2f}  '
              f'(peak rate ≈ e^{TARGET:.1f} × baseline ≈ '
              f'{np.exp(TARGET):.1f} × {self.baseline_rate:.4f} ≈ '
              f'{np.exp(TARGET) * self.baseline_rate:.3f} events/frame)')
        print()
        print('  Ground truth mapping:')
        for name in all_names:
            spec = additive_specs[name]
            grp  = list(spec.keys())[0] if spec else '(none)'
            print(f'    {name:30s}  →  {grp}')

        return {
            'Y_sim':         Y_sim,
            'W_true':        W_true,
            'neuron_names':  all_names,
            'neuron_type':   neuron_type,
            'neuron_groups': neuron_groups,
        }

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path):
        """Save simulation data and ground truth to a pickle file."""
        if self.Y_sim is None:
            raise RuntimeError('Call build_all_neurons() before save()')

        data = {
            'Y_sim':         self.Y_sim,
            'W_true':        self.W_true,
            'neuron_names':  self.neuron_names,
            'neuron_type':   self.neuron_type,
            'neuron_groups': self.neuron_groups,
            'pred_info':     self.pred_info,
            'baseline_rate': self.baseline_rate,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        mb = os.path.getsize(path) / 1e6
        print(f'Saved simulation data → {path}  ({mb:.2f} MB)')

    @classmethod
    def load(cls, path):
        """
        Load previously saved simulation data.

        Returns the raw dict (Y_sim, W_true, neuron_names, ...).
        Does NOT reconstruct the SimulatedNeurons object (X / metadata not stored).
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f'Loaded simulation data from {path}')
        print(f'  Y_sim  : {data["Y_sim"].shape}')
        print(f'  W_true : {data["W_true"].shape}')
        return data

    # ------------------------------------------------------------------
    # Plotting — ground-truth weights heatmap
    # ------------------------------------------------------------------

    def plot_ground_truth_weights(self, figsize=(12, 4)):
        """
        Heatmap of W_true: rows = predictors, columns = neurons.
        Predictor groups are color-coded on the y-axis.
        Each neuron lights up exactly one group row.
        """
        if self.W_true is None:
            raise RuntimeError('Call build_all_neurons() first')

        group_colors = {
            'position_only':  '#9B59B6',
            'audio_context':  '#27AAE1',
            'audio_stim':     '#1A7AAF',
            'visual_context': '#EC008C',
            'visual_stim':    '#A8005E',
            'choice_spatial': '#E67E22',
            'turn_onset':     '#E74C3C',
            'outcome':        '#F39C12',
        }

        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=150,
                                 gridspec_kw={'width_ratios': [0.04, 1]})

        # Left strip: group color bar
        ax_strip = axes[0]
        color_img = np.zeros((self.n_features, 1, 3))
        for group, info in self.pred_info.items():
            color = group_colors.get(group, '#AAAAAA')
            rgb = plt.matplotlib.colors.to_rgb(color)
            color_img[info['start']:info['end'], 0, :] = rgb
        ax_strip.imshow(color_img, aspect='auto', interpolation='none')
        ax_strip.set_xticks([])
        ax_strip.set_yticks(
            [(info['start'] + info['end']) // 2 for info in self.pred_info.values()]
        )
        ax_strip.set_yticklabels(list(self.pred_info.keys()), fontsize=6)
        ax_strip.tick_params(length=0)

        # Right: weight heatmap
        ax = axes[1]
        W_plot = self.W_true.copy()
        vmax = np.nanpercentile(np.abs(W_plot[W_plot != 0]), 99) if np.any(W_plot != 0) else 1.0
        im = ax.imshow(W_plot, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, interpolation='none')

        for info in self.pred_info.values():
            ax.axhline(info['start'] - 0.5, color='white', linewidth=0.4, alpha=0.6)

        ax.set_xticks(range(len(self.neuron_names)))
        ax.set_xticklabels(
            [n.split('_', 1)[1] for n in self.neuron_names],
            rotation=45, ha='right', fontsize=6
        )
        ax.set_yticks([])
        ax.set_xlabel('Simulated neuron', fontsize=8)
        ax.set_title(
            f'Ground-truth weights (W_true)  —  '
            f'{self.n_features} features × {len(self.neuron_names)} neurons',
            fontsize=8
        )

        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label='weight')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Plotting — trial-average activity
    # ------------------------------------------------------------------

    def plot_trial_average_turn_aligned(
            self,
            neuron_idx,
            Y_pred=None,
            n_pre_turn_frames=120,
            n_post_turn_frames=75,
            smooth_sigma=2.0,
            figsize=(13, 3.5),
    ):
        """
        Plot trial-average activity aligned to turn onset.

        Split variable is chosen automatically from the neuron name:
          • audio_stim neurons  → ctx × audio stim identity (0° vs 90°)
          • visual_stim neurons → ctx × visual stim identity (−90° vs +90°)
          • outcome neurons     → ctx × outcome (correct vs error)
          • all others          → ctx × choice (left vs right)

        X-axis: frames relative to turn onset (pre: −n_pre … −1, post: 0 … n_post−1).
        """
        if self.Y_sim is None:
            raise RuntimeError('Call build_all_neurons() first')
        if 'turn_frame_in_trial' not in self.metadata:
            raise KeyError(
                'metadata is missing turn_frame_in_trial. '
                'Re-run stack_design_matrices() with the updated session_pipeline.py.'
            )

        name = self.neuron_names[neuron_idx]
        Y_n  = self.Y_sim[:, neuron_idx]
        Y_p  = Y_pred[:, neuron_idx] if Y_pred is not None else None

        ctx  = self.metadata['context']
        ch   = self.metadata['choice']
        frit = self.metadata['frame_in_trial']
        tf   = self.metadata['turn_frame_in_trial']
        oc   = self.metadata['outcome']
        as_  = self.metadata.get('audio_stim',  None)
        vs_  = self.metadata.get('visual_stim', None)

        rel = frit - tf

        pre_bins  = np.arange(-n_pre_turn_frames, 0)
        post_bins = np.arange(0, n_post_turn_frames)
        all_bins  = np.concatenate([pre_bins, post_bins])
        x_axis    = all_bins

        ctx_styles = {
            CTX_CG:     ('purple',   'CG'),
            CTX_VISUAL: ('#EC008C',  'Visual'),
            CTX_AUDIO:  ('#27AAE1',  'Audio'),
        }

        # Split-mode lookup tables
        splits = {
            'choice':       ({CHOICE_LEFT: '-',     CHOICE_RIGHT: '--'},
                             {CHOICE_LEFT: 'Left',  CHOICE_RIGHT: 'Right'},
                             ch),
            'outcome':      ({OUTCOME_CORRECT: '-', OUTCOME_ERROR: '--'},
                             {OUTCOME_CORRECT: 'Correct', OUTCOME_ERROR: 'Error'},
                             oc),
            'audio_stim':   ({0.0: '-',   90.0: '--'},
                             {0.0: 'L cue (0°)',  90.0: 'R cue (90°)'},
                             as_),
            'visual_stim':  ({-90.0: '-', 90.0: '--'},
                             {-90.0: 'L (−90°)', 90.0: 'R (+90°)'},
                             vs_),
        }

        # Determine split mode from neuron name
        if any(t in name for t in ['audio_stim', 'N03', 'N04']):
            split_mode = 'audio_stim'
        elif any(t in name for t in ['visual_stim', 'N06', 'N07']):
            split_mode = 'visual_stim'
        elif any(t in name for t in ['outcome', 'correct', 'error', 'N12', 'N13']):
            split_mode = 'outcome'
        else:
            split_mode = 'choice'

        ls_map, lbl_map, split_arr = splits[split_mode]

        fig, axes = plt.subplots(
            1, 3,
            figsize=figsize, dpi=150,
            sharey=True,
            gridspec_kw={'wspace': 0.08}
        )
        axes = list(axes)

        def _compute_trace(Y_vals, cond_mask):
            trace = np.full(len(all_bins), np.nan)
            for i, b in enumerate(all_bins):
                m = cond_mask & (rel == b)
                if m.sum() >= 2:
                    trace[i] = Y_vals[m].mean()
            return self._smooth(trace, smooth_sigma)

        y_all = []

        def _plot_panel(ax, ctx_val, color, ctx_name):
            for c_val, ls in ls_map.items():
                cond_mask = (ctx == ctx_val) & (split_arr == c_val)
                trace = _compute_trace(Y_n, cond_mask)
                pred  = _compute_trace(Y_p, cond_mask) if Y_p is not None else None

                ax.plot(x_axis, trace, color=color, linestyle=ls,
                        linewidth=1.5, label=lbl_map[c_val], alpha=0.9)
                if pred is not None:
                    ax.plot(x_axis, pred, color=color, linestyle=ls,
                            linewidth=0.8, alpha=0.5)
                y_all.extend(trace[~np.isnan(trace)].tolist())

            ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
            ax.axhline(self.baseline_rate, color='grey', linewidth=0.5,
                       linestyle=':', alpha=0.4)
            ax.set_title(f'{ctx_name}', fontsize=7, color=color, fontweight='bold')
            ax.set_xlabel('Frames from turn onset', fontsize=6)
            ax.tick_params(labelsize=6)
            sns.despine(ax=ax)

        for col_i, (c_ctx, (color, ctx_name)) in enumerate(ctx_styles.items()):
            ax = axes[col_i]
            _plot_panel(ax, c_ctx, color, ctx_name)
            if col_i == 0:
                ax.set_ylabel('Mean activity\n(events/frame)', fontsize=6)
                ax.legend(fontsize=5.5, frameon=False, loc='upper left')
            else:
                ax.set_yticks([])
                sns.despine(ax=ax, left=True)

        if y_all:
            y_max = np.nanpercentile(y_all, 99) * 1.25
            for ax in axes:
                ax.set_ylim(0, max(y_max, self.baseline_rate * 2))

        frac_dev_str = ''
        if hasattr(self, '_last_results') and self._last_results is not None:
            fd = self._last_results['test_frac_dev'][neuron_idx]
            frac_dev_str = f'  |  frac dev = {fd:.3f}'

        split_label_map = {
            'choice':      'ctx × choice',
            'outcome':     'ctx × outcome',
            'audio_stim':  'ctx × audio stim identity',
            'visual_stim': 'ctx × visual stim identity',
        }
        pred_label   = '  thick=data, thin=model' if Y_pred is not None else ''
        smooth_label = f'  σ={smooth_sigma}fr' if smooth_sigma > 0 else ''
        fig.suptitle(
            f'[{self.neuron_type[neuron_idx]}]  {name}{frac_dev_str}\n'
            f'Turn-aligned  ({n_pre_turn_frames} pre / {n_post_turn_frames} post)'
            f'{smooth_label}{pred_label}  |  {split_label_map[split_mode]}',
            fontsize=7
        )
        plt.tight_layout()
        plt.show()

    def plot_all_neurons_summary(
            self,
            n_pre_turn_frames=120,
            n_post_turn_frames=75,
            smooth_sigma=2.0,
            ncols=4,
            figsize_per=(3.5, 2.5),
    ):
        """
        Grid of turn-aligned trial-average traces for all simulated neurons.

        Split variable per neuron:
          • audio_stim neurons  → audio stim identity (0° vs 90°)
          • visual_stim neurons → visual stim identity (−90° vs +90°)
          • outcome neurons     → outcome (correct vs error)
          • all others          → choice (left vs right)
        """
        if self.Y_sim is None:
            raise RuntimeError('Call build_all_neurons() first')
        if 'turn_frame_in_trial' not in self.metadata:
            raise KeyError(
                'metadata is missing turn_frame_in_trial. '
                'Re-run stack_design_matrices() with the updated session_pipeline.py.'
            )

        n_neurons = len(self.neuron_names)
        nrows     = int(np.ceil(n_neurons / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
            dpi=120
        )
        axes = axes.flatten()

        frit = self.metadata['frame_in_trial']
        tf   = self.metadata['turn_frame_in_trial']
        rel  = frit - tf

        ctx  = self.metadata['context']
        ch   = self.metadata['choice']
        oc   = self.metadata['outcome']
        as_  = self.metadata.get('audio_stim',  None)
        vs_  = self.metadata.get('visual_stim', None)

        pre_bins  = np.arange(-n_pre_turn_frames, 0)
        post_bins = np.arange(0, n_post_turn_frames)
        all_bins  = np.concatenate([pre_bins, post_bins])

        ctx_styles = {
            CTX_CG:     ('purple',   'CG'),
            CTX_VISUAL: ('#EC008C',  'Vis'),
            CTX_AUDIO:  ('#27AAE1',  'Aud'),
        }
        choice_ls      = {CHOICE_LEFT: '-',     CHOICE_RIGHT: '--'}
        outcome_ls     = {OUTCOME_CORRECT: '-', OUTCOME_ERROR:  ':'}
        audio_stim_ls  = {0.0: '-',   90.0: '--'}
        visual_stim_ls = {-90.0: '-', 90.0: '--'}

        for i, name in enumerate(self.neuron_names):
            ax  = axes[i]
            Y_n = self.Y_sim[:, i]

            if any(t in name for t in ['audio_stim', 'N03', 'N04']):
                split_dict = audio_stim_ls
                split_arr  = as_
            elif any(t in name for t in ['visual_stim', 'N06', 'N07']):
                split_dict = visual_stim_ls
                split_arr  = vs_
            elif any(t in name for t in ['outcome', 'correct', 'error', 'N12', 'N13']):
                split_dict = outcome_ls
                split_arr  = oc
            else:
                split_dict = choice_ls
                split_arr  = ch

            for c_ctx, (color, ctx_lbl) in ctx_styles.items():
                for c_val, ls in split_dict.items():
                    cond_mask = (ctx == c_ctx) & (split_arr == c_val)
                    trace = np.array([
                        Y_n[cond_mask & (rel == b)].mean()
                        if (cond_mask & (rel == b)).sum() >= 2 else np.nan
                        for b in all_bins
                    ])
                    trace = self._smooth(trace, smooth_sigma)
                    ax.plot(all_bins, trace, color=color, linestyle=ls,
                            linewidth=1.0, alpha=0.85)

            ax.axvline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)
            ax.axhline(self.baseline_rate, color='black', linewidth=0.5,
                       linestyle=':', alpha=0.4)
            ax.set_title(
                name.split('_', 1)[1].replace('_', ' '),
                fontsize=6
            )
            ax.tick_params(labelsize=5)
            ax.set_xlabel('Frames from turn', fontsize=5)
            ax.set_ylabel('Activity', fontsize=5)
            sns.despine(ax=ax)

        for j in range(n_neurons, len(axes)):
            axes[j].set_visible(False)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='purple',  lw=1.2, label='CG'),
            Line2D([0], [0], color='#EC008C', lw=1.2, label='Visual'),
            Line2D([0], [0], color='#27AAE1', lw=1.2, label='Audio'),
            Line2D([0], [0], color='grey', lw=1.2, ls='-',
                   label='Left / Correct / L-stim'),
            Line2D([0], [0], color='grey', lw=1.2, ls='--',
                   label='Right / Error / R-stim'),
        ]
        fig.legend(handles=legend_elements, loc='lower center',
                   ncol=5, fontsize=6, frameon=False,
                   bbox_to_anchor=(0.5, -0.01))

        smooth_label = f'  |  smoothed σ={smooth_sigma} frames' if smooth_sigma > 0 else ''
        plt.suptitle(
            f'Simulated neurons — turn-aligned trial-average activity\n'
            f'({n_pre_turn_frames} frames pre-turn  |  {n_post_turn_frames} frames post-turn)'
            f'{smooth_label}',
            fontsize=9, y=1.01
        )
        plt.tight_layout()
        plt.show()
