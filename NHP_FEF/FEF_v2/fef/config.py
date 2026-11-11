"""
config.py
=========
Paths, session inventory, and the parameters that were settled empirically in
v1. Every value with a measurement behind it carries the measurement.

Override paths with the NHP_SESSIONS / NHP_RESULTS environment variables.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------- paths

SESSIONS_DIR = os.environ.get(
    "NHP_SESSIONS", "/Volumes/Akhil Data/Akhil/FEF_Data/data")
RESULTS_DIR = os.environ.get(
    "NHP_RESULTS", "/Volumes/Akhil Data/Akhil/FEF_Data/model_results_v2")

#: Session-file discovery pattern. The test suite globs SESSIONS_DIR with this
#: and nothing else -- v1's suite also globbed the repo, so when the data moved
#: to the server 23 data-dependent tests skipped silently and the run was green
#: while exercising nothing. One source, and WP0 gates on zero skips.
SESSION_GLOB = "*_g0-*.mat"

#: All 26 files carry this export hash. v1's results were computed on
#: `171af6ac94149c62` — a DIFFERENT export. See docs/03_PORT_AUDIT.md; validate
#: before trusting any inherited constant.
EXPORT_HASH = "044e61186379fd29"
V1_EXPORT_HASH = "171af6ac94149c62"

# ---------------------------------------------------------------- inventory

MONKEY_S_SESSIONS = [
    "0136a", "0137a", "0138a", "0139a", "0140a", "0141a", "0142a", "0143a",
    "0144a", "0145a", "0146a", "0147a", "0149a", "0158a", "0159a", "0160a",
]
MONKEY_Y_SESSIONS = [
    "Ya_250429_s385", "Ya_250430_s386", "Ya_250506_s390", "Ya_250514_s394",
    "Ya_250515_s395", "Ya_250520_s397", "Ya_250522_s399", "Ya_250609_s405",
    "Ya_250619_s412", "Ya_250721_s422",
]

#: Analysed in v1. The other 15 sessions have never been fit.
V1_SESSIONS = ["0136a", "0137a", "0138a", "0140a", "0141a", "0143a", "0144a",
               "0146a", "0147a", "0149a", "0160a"]

#: MEASURED IN WP0. Four Monkey Y sessions recorded ONE hemisphere, so the
#: dataset is 48 probes, NOT the 52 (26 x 2) that PLAN.md assumed. Every
#: "N of 52" acceptance criterion has to be restated, and WP11 (cross-hemisphere
#: coupling) can only run on the 22 sessions that have two simultaneous probes.
#:
#: These four also store `sorting.*` as PLAIN ARRAYS rather than a (1, n) cell of
#: object references, because MATLAB does not wrap a scalar cell. `unit_depths()`
#: and any direct `sorting` access must branch on `dtype == object` or they raise
#: `TypeError: Accessing a group is done with bytes or str, not numpy.float64`.
SINGLE_PROBE_SESSIONS = ["Ya_250429_s385", "Ya_250430_s386",
                         "Ya_250506_s390", "Ya_250514_s394"]

#: Probe counts by arm, measured. Use these, never `2 * len(sessions)`.
N_PROBES = {"S": 32, "Y": 16, "all": 48}
N_BILATERAL_SESSIONS = {"S": 16, "Y": 6, "all": 22}      # WP11's real ceiling

#: Total sorted units, pre-QC, measured in WP0: 23,904 on Monkey S and 13,427 on
#: Monkey Y, median 752 per probe against v1's 402 on 0136a. Fit cost is
#: per-neuron, so the primary (Monkey S) arm is ~46 h for a base pass and the
#: full 48 probes would be ~87 h. See WP0.5 -- QC is expected to remove most of
#: this, since the authors' own pipeline rejected 67% of clusters.
N_UNITS_PRE_QC = {"S": 23904, "Y": 13427, "all": 37331}

#: DECISION (2026-08-04, SUPERSEDES the 2026-08-03 pooling decision).
#: The monkeys are NOT pooled. WP0 measured what "Monkey Y" actually is and it
#: is a DIFFERENT PROTOCOL, not a second sample of the same one
#: (results/WP0_data_validation.md Result 6): backward-step pursuit ONLY in all
#: 10 sessions, a single 15 deg/s ramp speed, a single eccentricity, and <=25
#: correct saccade trials per direction in 7 of 10 sessions.
#:
#: Monkey S is the PRIMARY arm; Monkey Y is a SEPARATE replication/contrast arm,
#: fit and reported on its own. This restores a genuine held-out set -- which is
#: what STANDING_RULES S1's four guards were written to substitute for. Keep the
#: guards anyway (they are good practice), but S1's framing is now conservative
#: rather than load-bearing.
POOL_MONKEYS = False
PRIMARY_MONKEY = "S"
REPLICATION_MONKEY = "Y"

#: Sessions with repeated task blocks. WP0 found FOUR, not the two v1 knew about:
#: `0145a` (mdir1/mdir2/purs1/purs2) and `Ya_250522_s399` (purs1/purs2/purs3 --
#: the first three-way repeat in the dataset) are new. All four decode to
#: DISTINCT tables; the three MCOS witnesses resolve every one.
#:
#: v1's argument for pooling them was that they are the only sessions where the
#: blocks interleave in time, so recording drift is separable from task identity
#: (0146a's time-vs-task correlation 0.821 -> 0.512, 0147a 0.871 -> 0.773; every
#: other session sits at 0.72-0.87). That correlation has NOT been re-measured
#: for the two new sessions -- do so before relying on it.
REPEATED_BLOCK_SESSIONS = ["0145a", "0146a", "0147a", "Ya_250522_s399"]
POOL_TASK_BLOCKS = True

#: Condition levels, MEASURED on all trials of all 26 sessions in WP0 (not just
#: correct trials -- a level that exists but is never performed correctly is a
#: balance problem, whereas a level never presented is a design difference, and
#: only the second makes a predictor group degenerate).
#:
#: CORRECTION: v1 recorded Monkey Y's ramp speed as 10 deg/s. It is 15, in all
#: 10 sessions, never 10.
MONKEY_CONDITIONS = {
    "S": {"eccentricity_deg": (10.0, 20.0), "ramp_speed_dps": (10.0, 20.0)},
    "Y": {"eccentricity_deg": (10.0,), "ramp_speed_dps": (15.0,)},
}

#: Sessions whose condition levels differ from their animal's default above.
#: `magnitude` is therefore not universal even within Monkey S.
SESSION_CONDITION_EXCEPTIONS = {
    "0142a": {"ramp_speed_dps": (15.0, 20.0)},      # not (10, 20)
    "0147a": {"eccentricity_deg": (10.0, 15.0, 20.0)},   # THREE levels
}

#: Pursuit step direction, measured over all pursuit trials per session.
#: Monkey S splits into two protocol eras and Monkey Y has only one level, so
#: `jump` / `step_direction` are DEGENERATE FOR MONKEY Y exactly as `magnitude`
#: is -- a case PLAN.md WP4 step 4 did not anticipate.
STEP_DIRECTION_ERAS = {
    "S_era1": ["0136a", "0137a", "0138a", "0139a", "0140a", "0141a"],   # ~50% fwd
    "S_era2": ["0142a", "0143a", "0144a", "0145a", "0146a", "0147a",
               "0149a", "0158a", "0159a", "0160a"],                     # ~25% fwd
}
JUMP_DEGENERATE_MONKEYS = ("Y",)      # backward step only, 0% forward

# ------------------------------------------------------------------ contexts

CTX_SACCADE, CTX_PURSUIT = 1, 2
CONTEXT_LABEL = {CTX_SACCADE: "saccade", CTX_PURSUIT: "pursuit"}

#: Block names. `SACCADE_BLOCK`/`PURSUIT_BLOCK` are the single-block defaults;
#: the PREFIX forms are what POOL_TASK_BLOCKS uses to sweep up mdir2/purs2 on
#: the two repeated-block sessions.
SACCADE_BLOCK, PURSUIT_BLOCK = "mdir1", "purs1"
SACCADE_BLOCK_PREFIX, PURSUIT_BLOCK_PREFIX = "mdir", "purs"

N_DIRECTIONS = 8
DIRECTIONS_DEG = tuple(range(0, 360, 45))

# ------------------------------------------------------------ eye kinematics

#: eyePos/eyeVel/eyeAcc are (n_ms, 2) = (horizontal, vertical) at 1 ms.
#: VERIFIED in degrees on 0136a: measured saccade amplitudes 10.45 / 19.51 deg
#: against commanded 10 / 20, direction error median 2.1 deg, peak speeds
#: 485-840 deg/s. WP0 re-checks this on the new export.
EYE_H, EYE_V = 0, 1

#: A detected saccade closer than this to movement onset is the primary
#: movement, not a catch-up saccade.
CATCHUP_EXCLUDE_MS = 150.0

# ------------------------------------------------------------------- binning

#: 40 ms, not 25. v1's most consequential empirical result: pursuit is
#: shot-noise-limited, not predictor-limited. On gated data with the full design,
#: 25 -> 40 -> 50 ms roughly DOUBLED median FDE_pursuit (0.012 -> 0.022), raised
#: pursuit PSTH R^2 from 0.20 to 0.34 and responsive units from 18% to 32% on the
#: same neurons (rank corr 0.97), while the rejected kinematic candidates each
#: moved it by <= 0.0008.
#: CAVEAT: the EARLIER pre-gate bin sweep (25 units, v1.1 design) found the
#: opposite -- FDE_pur peaking at 25 ms (0.017) and dipping at 50 ms (0.016).
#: The two have never been reconciled on the same footing. Re-run the sweep in
#: Stage 3; it decides this value.
BIN_MS = 40
WIN_PRE_MS, WIN_POST_MS = -200, 1500      # relative to target onset

#: WP4 DISCOVERY (2026-08-11, MEASURED on all 10 Y sessions): Monkey Y's
#: trials are SHORTER than the 1500 ms window assumes. END_TRIAL - target_on,
#: 5th percentile per session: pursuit 1488-1529 ms (median 1490 on s412),
#: saccade 1360-1420 ms. The converter's window filter requires
#: target_on + WIN_POST <= END_TRIAL, so at 1500 ms it silently dropped 100%
#: of Y pursuit trials and ~40% of Y saccade trials — the whole Y arm would
#: have been unfittable (or saccade-only) without this. v1 never fit Y, so
#: the incompatibility was latent since the port. 1350 keeps >= ~95% of both
#: tasks' correct trials in every Y session; Y reward p95 is 1262 (pursuit) so
#: the reward kernels stay supported. The arms are never pooled (A1), so a
#: per-monkey window is legitimate; every Y fit must report it.
WIN_POST_MS_BY_MONKEY = {"S": 1500, "Y": 1350}

#: Noneman et al. analysis epochs, ms relative to their anchor events. Used for
#: the model-free tuning curves in fef/readout/tuning.py.
EPOCH_VISUAL = (50, 150)      # after target flash (saccade task)
EPOCH_SACCADE = (-50, 50)     # around saccade onset
EPOCH_PURSUIT = (-50, 50)     # around pursuit onset
EPOCH_BASELINE = (-100, 0)    # before target onset, per task

#: Development subsets. DEV4 is a STRICT SUBSET of DEV8 so a pipeline validated
#: on DEV4 (STANDING_RULES S7) extends to the DEV8 ablation arm without
#: re-validation.
#:
#: REDEFINED 2026-08-04. The old definition -- 0136a + Ya_250429_s385, "both
#: hemispheres = 8 probes" -- was never 8 probes: BOTH Monkey Y sessions it named
#: are SINGLE-PROBE, so DEV4 was 3 probes and DEV8 was 6, and WP5's ">=6/8
#: probes" adoption rule was undefined. With the monkeys no longer pooled the
#: dev subsets are Monkey S only, and they now span BOTH step-direction eras so
#: an ablation validated here generalises across the whole primary arm.
#:
#: Chosen for: one session per era, standard condition levels, healthy trial
#: counts, and the SMALLEST unit counts that satisfy those -- the S7 pass has to
#: be cheap to be worth running. 0136a is kept because it is the reference
#: session every inherited fact is pinned to.
#:   DEV4 = 0136a (era 1, 780 units) + 0160a (era 2, 736 units)   ~3.5 h
#:   DEV8 = +      0137a (era 1, 906) + 0143a (era 2, 1597)       ~9.3 h
#: Deliberately avoided: 0140a (128-unit lFEF), 0158a/0159a (>2500 units, slow),
#: 0142a/0147a (non-standard condition levels).
DEV4_SESSIONS = ["0136a", "0160a"]
DEV8_SESSIONS = DEV4_SESSIONS + ["0137a", "0143a"]

#: Monkey Y is a separate arm, so it needs its own validation session. The
#: healthiest Y session: bilateral, 1008 correct saccade / 708 correct pursuit.
DEV_Y_SESSIONS = ["Ya_250619_s412"]

# ---------------------------------------------------------- trial selection

DROP_DEAD_TRIALS = True       # population rate < DEAD_RATE_FRAC x alive baseline
DEAD_RATE_FRAC = 0.30
EQUALIZE_CONTEXTS = True      # cap both tasks at the global per-direction min

#: VERIFIED in v1: the `result` categorical decodes to a LABEL STRING. The
#: numeric 150 lives in `dat.result`, not in `tbl`.
RESULT_CORRECT = "CORRECT"

MIN_REPS_PER_CONDITION = 5    # min trials per (task, direction)
ARTIFACT_SD = 3.0             # drop trials whose population rate exceeds this
                              # many SD above the session mean (the paper's
                              # HIGH-outlier rejection; DROP_DEAD_TRIALS above
                              # removes the LOW/dead tail)

#: None = keep BOTH pursuit step directions. v1 kept only backward, discarding
#: 50.3% of correct pursuit trials. Pooling requires the `jump` /
#: `step_direction` predictors -- without them the target APPEARS at angle+180
#: on backward trials and pooling makes FDE_pursuit worse (-0.0059 on lFEF);
#: with them, +0.0029, and the sac/pur imbalance falls (lFEF 1.23 -> 1.04,
#: rFEF 1.81 -> 1.50). The data change and the predictors are one package.
PURSUIT_JUMP_KEEP = None

# -------------------------------------------------------------- v1 readout

RESP_THRESHOLD = 0.1          # inherited from calcium; report the matched cut too
GATE_MODE = "either"          # responsive if EITHER per-task FDE clears
MIN_TASK_FDE = 0.02           # for the `credible` flag
CI_THRESH, CD_THRESH = 0.33, 0.67

#: The mouse PAPER defines CI as |CPI| < 0.1 (n = 6,634) and CD as |CPI| >= 0.8
#: (n = 3,456). At those thresholds the median FEF probe has only 10 CI units and
#: 5 probes have fewer than 6 -- too few for a 5-PC coupling source, which is why
#: CI_THRESH/CD_THRESH above are 0.33/0.67 (median 34 CI / 30 CD, 19 usable
#: probes). State the threshold with every number, and show a sweep.
MOUSE_CI_THRESH, MOUSE_CD_THRESH = 0.1, 0.8

# -------------------------------------------------------------- v2 readout

RELIABILITY_FLOOR = 0.2       # split-half floor for DGI; below it, correction is noise
DGI_N_SPLITS = 50

#: PROVISIONAL DEFAULTS ONLY. `THRESHOLDS.json`, written by WP2 step 1 before the
#: DGI distribution is inspected, is the AUTHORITATIVE source for the CI/CD cuts;
#: WP2 writes its derived values back over these. Do not read these constants in
#: any analysis module -- read THRESHOLDS.json. Two competing sources of truth for
#: the partition is exactly how a threshold gets tuned to a result.
DGI_CI_THRESH = 0.7
DGI_CD_THRESH = 0.3

#: GATE B (PLAN.md WP2) is defined on this ABSOLUTE cut, deliberately NOT on the
#: proportion-matched CI/CD cuts: a proportion-matched cut is a quantile, so a
#: fixed fraction sits above it by construction and a "no mixture" distribution
#: could never be detected.
DGI_GATE_B_ABS = 0.7
DGI_GATE_B_FRAC = 0.85

# -------------------------------------------------------------- coupling

COUPLING_N_PCS = 5
COUPLING_N_LAG_BASES = 4
COUPLING_LAG_FRAMES = 12      # 12 x 40 ms ~ 480 ms, matching the mouse's ~0.5 s

#: TRUE, unlike v1. v1 passed exclude_col=None, reasoning that self-share is
#: ~1/n. That holds for the population MEAN but not for the PCs: a high-variance
#: unit loads on PC1 and partly predicts itself, inflating its own gain. High
#: variance means high rate, which is exactly the direction of the Figure-4d
#: confound. subpop_loo.py already did this correctly; the base model must too.
COUPLING_EXCLUDE_SELF = True

# -------------------------------------------------------------- mouse ref

#: RECONCILED 2026-08-04 (WP2 step 1). The two tallies were never in conflict:
#: they are ONE file at two cut definitions, plus a one-session difference.
#: `analysis/split_coupling_neurons.csv` holds 40 sessions = 62,519 / 25,406
#: responsive; the manuscript drops KO-6-00/2026-04-06 (2,024 neurons, 292
#: responsive), which reproduces 60,495 / 25,114 and 41.5% EXACTLY. On that
#: 39-session set: CI/Int/CD = 62.8/19.7/17.5 at the 0.33/0.67 cuts, and
#: 26.4/59.8/13.8 at the manuscript's 0.1/0.8 (n = 6,634 / 15,024 / 3,456).
#: Both of the old comment's numbers verify; only its framing was wrong.
#: THRESHOLDS.json -> dgi.mouse_reference is authoritative and registers BOTH
#: cut definitions as the proportion-matching targets.
MOUSE_PPC = {
    "n_total": 60495, "n_responsive": 25114,          # 39 sessions, 41.5%
    "abs_cpi_median": 0.2246,                         # recomputed, 39 sessions
    "CI": 0.628, "Int": 0.197, "CD": 0.175,           # at 0.33/0.67
    "CI_paper": 0.264, "Int_paper": 0.598, "CD_paper": 0.138,   # at 0.1/0.8
    "coupling_slope_r": -0.27,
    "decode_within_visual": 0.93, "decode_within_audio": 0.91,
    "decode_across": 0.73,
    "ci_frac_vs_across_decoding_r": 0.693,
}

# ----------------------------------------------------- design matrix (WP4)
# Ported from v1 nhp/config.py 2026-08-11 (WP4a). Every measured comment is
# carried; only the v2 changes are new. See prompts/WP4_IMPLEMENTATION_PLAN.md.

#: Event-kernel basis sizes and windows (raised cosines).
N_TARGET_BASES = 6
N_MOVE_BASES = 6
N_GO_BASES = 4
N_REWARD_BASES = 4
TARGET_WIN_MS = (0.0, 500.0)
MOVE_WIN_MS = (-150.0, 450.0)
GO_WIN_MS = (0.0, 400.0)
REWARD_WIN_MS = (0.0, 500.0)
#: Narrower reward window used when SHARED_PREDICTORS_ONLY is set: at 500 ms
#: the last basis falls past the end of the pursuit window (reward sits later
#: in the pursuit trial), leaving it live on saccade trials only.
REWARD_WIN_SHARED_MS = (0.0, 250.0)

#: PER-MONKEY kernel windows (WP4e QC verdict, decided by Akhil 2026-08-11;
#: results/WP4_design.md Result 2). Monkey Y's saccades execute at the trial
#: END (move onset median 1083-1133 ms, p95 1253-1329) and its pursuit reward
#: is at a FIXED ~1260 ms, so kernel tails past ~+200 ms (movement) / ~+80 ms
#: (reward) fall beyond the 1350 ms Y window on one task only — making the
#: last bases single-task and near-exactly collinear with their context-signed
#: copies (cond 184-2289 in 9/10 Y sessions). Shortening the Y banks to what
#: the trial has room for restores identifiability; same principle as
#: WIN_POST_MS_BY_MONKEY (the arms are never pooled). Exceptions ONLY: a
#: monkey absent from these dicts keeps the v1-validated windows above, so the
#: S design is bit-identical to the certified one. Every Y fit reports its
#: windows. Re-certified by scripts/wp4_design_qc.py after the change.
MOVE_WIN_MS_BY_MONKEY = {"Y": (-150.0, 200.0)}
REWARD_WIN_MS_BY_MONKEY = {"Y": (0.0, 80.0)}       # 4 bases kept (S-comparable)

#: Per-monkey eye-lag bank sizes (exceptions only; same WP4e decision). The
#: window fix alone left cond 120-179 in 8/10 Y sessions: with ONE ramp speed
#: and few saccade trials, Y's eye position is ~linear in time on most trials
#: (and its tracking velocity ~constant), so the ±200 ms lagged copies of the
#: position/velocity traces are near-collinear — the smallest Gram eigenvector
#: was the second-difference of the position lags in every failing session.
#: Two position lags span a line exactly; three velocity lags keep the
#: ~50-100 ms pursuit lead expressible. Measured result: cond 45-93 in 10/10
#: (S arm's own range is 71-99). Y default design = 208 columns.
EYE_LAG_BASES_BY_MONKEY = {"Y": 3}     # eye_velocity bank (4 signals x lags)
POS_LAG_BASES_BY_MONKEY = {"Y": 2}     # eye_position bank (2 signals x lags)

DIRECTION_HARMONICS = 2       # [cos t, sin t, cos 2t, sin 2t]

#: v2 DIRECTION PARAMETERIZATION (new). 'shared_diff' emits direction_shared
#: (harmonics x target/movement kernels, both tasks) + direction_diff (the same
#: columns x the sum-coded task indicator), so per-task tuning is recoverable
#: in closed form: w_sac = w_shared + w_diff, w_pur = w_shared - w_diff
#: (fef/design/reparam.py). 'per_task' emits direction_sac / direction_pur
#: instead (the WP7 comparison variant; exactly X_shared_diff @ T).
DIRECTION_PARAM = "shared_diff"

#: Sustained basis (covers the pursuit ramp). v1 measurement: without it, 28%
#: of pursuit bins carried <2% of peak predictor energy and FDE_pursuit was
#: floored at 0 for 270/374 units on 0136a_lFEF.
INCLUDE_SUSTAINED = True
N_SUSTAINED_BASES = 8
SUSTAINED_WIN_MS = (0.0, 1500.0)

INCLUDE_SIGNED_VELOCITY = True    # eye vx/vy: the pursuit direction signal
INCLUDE_SIGNED_ACCEL = False      # rejected in v1 ablation (+0.0005)

#: Catch-up saccades: 4.35 per pursuit trial, 100% of trials — the largest
#: identifiable unmodelled event class. Nuisance kernels shared across tasks.
INCLUDE_CATCHUP = True
N_CATCHUP_BASES = 6
CATCHUP_WIN_MS = (-100.0, 300.0)
INCLUDE_CATCHUP_DIRECTION = True
CATCHUP_HARMONICS = 1             # [cos t, sin t] per catch-up saccade

#: NEW in v2 (was inventoried, never implemented): each catch-up saccade's
#: kernel weighted by its amplitude (deg, raw — decided pre-WP5, see plan §5
#: open item 2). OFF until the WP5 ablation rules on it.
INCLUDE_CATCHUP_AMPLITUDE = False

#: Eye-kinematic temporal basis; negative lags let FEF activity LEAD the eye
#: by the ~50-100 ms pursuit lead.
INCLUDE_EYE_LAGS = True
N_EYE_LAG_BASES = 5
EYE_LAG_WIN_MS = (-200.0, 200.0)

#: Shared-predictor constraint and the symmetric task-specific budget
#: (v1 measurements in nhp/config.py): go+delay are the saccade side,
#: jump (+ stim_offset anchored per task) the pursuit side.
SHARED_PREDICTORS_ONLY = True
INCLUDE_GO = True
INCLUDE_DELAY = True
INCLUDE_STEP_DIRECTION = True     # re-signs target-locked harmonics; 0 cols
INCLUDE_JUMP = True               # step direction x target kernels, pursuit side
INCLUDE_STIM_OFFSET = True
N_STIM_OFFSET_BASES = 4
STIM_OFFSET_WIN_MS = (0.0, 400.0)

#: All three S9-settled rejections stay off with their v1 evidence:
INCLUDE_TARGET_VELOCITY = False   # +0.0003 over eye_velocity alone (ablated)
INCLUDE_RETINAL_SLIP = False      # redundant with eye velocity during tracking
INCLUDE_DRIFT = False             # r = 0.886 with context (time-segregated blocks)
N_DRIFT_BASES = 4

#: Underpowered in v1 (mean +0.00045, 1/4 probes significant); OFF pending the
#: WP5 ablation. DEGENERATE FOR MONKEY Y (single level in both tasks) — the
#: builder drops it data-drivenly; see fef/design/predictors.py.
INCLUDE_MAGNITUDE = False

#: NEW in v2: pupil, within-task centered. The v2 converter does NOT yet
#: extract the per-trial pupil trace — wiring it is a WP5 decision (plan §5
#: open item). The flag exists so the design contract is stable.
INCLUDE_PUPIL = False
N_PUPIL_LAGS = 5                  # reuses EYE_LAG_WIN_MS

# ------------------------------------------------------------------ helpers


def session_files(monkey=None):
    """Absolute paths to the session .mat files, optionally filtered by monkey.

    ``monkey=None`` returns all 26 (both animals pooled -- the default).
    """
    names = []
    if monkey in (None, "S"):
        names += [f"kendra_scrappy_{s}" for s in MONKEY_S_SESSIONS]
    if monkey in (None, "Y"):
        names += list(MONKEY_Y_SESSIONS)
    return [os.path.join(SESSIONS_DIR, f"{n}_g0-{EXPORT_HASH}.mat")
            for n in names]
