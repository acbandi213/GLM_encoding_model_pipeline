# Monkey FEF model fitting

This folder fits the same Poisson GLM as the mouse pipeline to Neuropixels
recordings from monkey frontal eye field (FEF). It has its own data reader and
design matrix, and reuses the repository's `core/` for stacking, fitting,
population coupling and lesion refits.

## The data

- One `.mat` file per session (`*_g0-*.mat`), one or two probes per session.
- Two tasks, coded as contexts: memory-guided saccade (1) and step-ramp
  pursuit (2).
- Spikes are counted in 40 ms bins, from 200 ms before target onset to 1500 ms
  after it (1350 ms for the second monkey).
- The session files are not part of this repository.

Tell the code where things are:

```bash
export NHP_SESSIONS=/path/to/session_mat_files     # input
export NHP_RESULTS=/path/to/results                # stacked data and fits
```

## The two steps

Run everything from this folder (`NHP_FEF/FEF_v2`).

**1. Build the stacked data (once).** Reads each `.mat` file, builds the design
matrix, and saves one file per probe to `{NHP_RESULTS}/stacked/`. It runs one
session at a time and skips probes that are already built.

```bash
python scripts/wp6_build_stacked.py                     # all sessions
python scripts/wp6_build_stacked.py --sessions 0136a    # one session
```

**2. Fit.** Each probe is fitted in its own process.

```bash
python models/GLM_base.py --workers 6            # full model, then lesions
python models/GLM_base.py --stages base          # full model only
python models/GLM_base.py --monkey S             # one monkey
python models/GLM_base.py --probe-key 0136a_lFEF # one probe, in this process
python models/GLM_base_coupling.py --workers 6   # with population coupling
```

Results are saved to `{NHP_RESULTS}/fits/{session}_{probe}/GLM_base_results/`
as `glm_base_full.pkl` and one `glm_base_lesion_{group}.pkl` per predictor
group, in the same format as the mouse results.

## Predictor groups

228 predictors in 16 groups for the first monkey (208 in 15 groups for the
second, which has no `jump` group). Always look groups up by name in
`predictor_info`, never by column position.

| Group | Columns | What it is |
|---|---|---|
| `target` | 6 | Response locked to target onset |
| `movement` | 6 | Response locked to movement onset |
| `go` | 4 | Go cue (saccade task only) |
| `delay` | 1 | Delay period (saccade task only) |
| `jump` | 6 | Target step (pursuit task only) |
| `reward` | 4 | Reward delivery |
| `stim_offset` | 4 | Target offset |
| `direction_shared` | 48 | Direction tuning common to both tasks |
| `direction_diff` | 48 | Difference in direction tuning between tasks |
| `context` | 13 | Task identity over the trial |
| `sustained`, `sustained_direction` | 8, 32 | Sustained activity, overall and by direction |
| `catchup_saccade`, `catchup_direction` | 6, 12 | Catch-up saccades during pursuit |
| `eye_position`, `eye_velocity` | 10, 20 | Continuous eye signals |

## How it differs from the mouse fit

- There is no left/right choice. Trials are balanced on target direction
  (eight directions), and both tasks contribute the same number of trials.
- The fitting settings are fixed in `fef/fit/invariants.py` and checked before
  every fit, so two models can always be compared.

## Layout

```
fef/config.py            paths, session lists, settled parameters
fef/io/                  .mat reader and session converter
fef/design/              design matrix builder and its checks
fef/fit/stacking.py      step 1: per-probe stacked data
fef/fit/model_runner.py  step 2: fit, lesions, parallel driver
fef/readout/tuning.py    direction-tuning readouts
models/                  GLM_base.py, GLM_base_coupling.py
tests/                   need the session files (set NHP_SESSIONS)
```
