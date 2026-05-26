#!/usr/bin/env python
"""
run_unisensory_models.py
=================
Master runner for all 6 GLM model scripts.

Two execution modes
-------------------
SESSION-FIRST (default, recommended)
    For each session → run all selected models → memory fully cleared.
    Each (session × model) runs as a separate subprocess so the OS
    reclaims ALL memory (numpy arrays, TF/Keras state, everything)
    before the next combination starts.  Use this when total RAM is
    a concern.

    Supports parallel execution via --max-workers N (default 2).
    Each worker runs one subprocess at a time; N subprocesses run
    simultaneously.  Recommended: 2–3 on M2 Ultra with Metal GPU.
    Already-completed sessions are automatically skipped (skip_existing
    logic inside each GLM script checks for *_full.pkl / *_lesion_all.pkl).

    Order:  all (session × model) jobs submitted to a thread pool;
            up to --max-workers run concurrently.

MODEL-FIRST  (--model-first)
    For each model → run all sessions → next model.
    Each model script is launched once as a subprocess.  Within that
    subprocess, sessions are run sequentially with per-session gc calls.
    Use this when you want each model's full results on disk before
    the next model starts (e.g. for intermediate inspection).
    Note: --max-workers has no effect in model-first mode.

    Order:  model1 × [session1, session2, …]
            model2 × [session1, session2, …]
            …

Available models
----------------
    GLM_all_context           contexts=[0,1,2]  no coupling
    GLM_all_context_coupling  contexts=[0,1,2]  + coupling
    GLM_uni                   contexts=[1,2]    no coupling
    GLM_uni_coupling          contexts=[1,2]    + coupling
    GLM_CG                    contexts=[0]      no coupling
    GLM_CG_coupling           contexts=[0]      + coupling

Usage
-----
# Run all 6 models, 2 parallel workers (default):
python run_unisensory_models.py

# Run with 3 parallel workers (watch GPU / RAM before going higher):
python run_unisensory_models.py --max-workers 3

# Run sequentially (original behaviour):
python run_unisensory_models.py --max-workers 1

# Run specific models:
python run_unisensory_models.py --models GLM_uni GLM_CG

# Skip all coupling variants:
python run_unisensory_models.py --no-coupling

# Model-first ordering instead:
python run_unisensory_models.py --model-first

# Force re-run with a different metric:
python run_unisensory_models.py --no-skip --metric dff

# Build missing glm_data_stacked.pkl files before running models:
python run_unisensory_models.py --build-stacked

# List available model names:
python run_unisensory_models.py --list
"""

import argparse
import gc
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Protects all writes to the shared master log file and to stdout headers/footers
# so that output from parallel workers does not interleave at the file level.
_LOG_LOCK = threading.Lock()

# Script is in runners/, need to go up 1 level to GLM_encoding_model root
_RUNNER_DIR = os.path.dirname(os.path.abspath(__file__))
GLM_DIR = os.path.dirname(_RUNNER_DIR)
ICLOUD_GLM_DIR = ('/Users/akhilbandi/Library/Mobile Documents/'
                  'com~apple~CloudDocs/Documents/Portfolio/code_projects/'
                  'Context Paper Code/GLM_encoding_model')
for _p in (GLM_DIR, ICLOUD_GLM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.session_registry import PPC_AVC
from core.session_pipeline import (
    BASE_DATA_PATH,
    extract_data,
    generate_predictors,
    stack_design_matrices,
    save_glm_data,
)

# ── Model registry ────────────────────────────────────────────────────────────
ALL_MODELS = [
    dict(name='GLM_all_context',          module='GLM_all_context',          script='models/unisensory/GLM_all_context.py',          coupling=False, contexts=[0, 1, 2]),
    dict(name='GLM_all_context_coupling', module='GLM_all_context_coupling', script='models/unisensory/GLM_all_context_coupling.py', coupling=True,  contexts=[0, 1, 2]),
    dict(name='GLM_uni',                  module='GLM_uni',                  script='models/unisensory/GLM_uni.py',                  coupling=False, contexts=[1, 2]),
    dict(name='GLM_uni_coupling',         module='GLM_uni_coupling',         script='models/unisensory/GLM_uni_coupling.py',         coupling=True,  contexts=[1, 2]),
    dict(name='GLM_CG',                   module='GLM_CG',                   script='models/unisensory/GLM_CG.py',                   coupling=False, contexts=[0]),
    dict(name='GLM_CG_coupling',          module='GLM_CG_coupling',          script='models/unisensory/GLM_CG_coupling.py',          coupling=True,  contexts=[0]),
]
_NAME_TO_MODEL = {m['name']: m for m in ALL_MODELS}

_HELPER = os.path.join(GLM_DIR, 'runners', '_run_single_session.py')


# ── Pre-flight: stacked data helpers ─────────────────────────────────────────

def _find_missing_stacked(sessions_dict):
    """Return list of (mouse_ID, date) pairs that are missing glm_data_stacked.pkl."""
    missing = []
    for mouse_ID, dates in sessions_dict.items():
        for date in dates:
            path = os.path.join(BASE_DATA_PATH, mouse_ID, date, 'glm_data_stacked.pkl')
            if not os.path.exists(path):
                missing.append((mouse_ID, date))
    return missing


def _build_stacked_for_sessions(missing, metric='deconv', log_path=None):
    """
    Build glm_data_stacked.pkl for each (mouse_ID, date) in *missing*.

    Calls the pipeline functions directly (no subprocess) so memory is freed
    after each session via gc.collect().  Aborts on the first failure.
    """
    n = len(missing)
    msg = f'[pre-flight] Building glm_data_stacked.pkl for {n} session(s)...'
    print(f'\n{msg}')
    if log_path:
        with open(log_path, 'a') as f:
            f.write(msg + '\n')

    for i, (mouse_ID, date) in enumerate(missing, 1):
        label = f'  [{i}/{n}]  {mouse_ID} / {date}'
        print(label)
        if log_path:
            with open(log_path, 'a') as f:
                f.write(label + '\n')

        glm_data_dict = design_matrices = X = Y = None
        predictor_names = predictor_info = metadata = group_indices = None
        try:
            glm_data_dict = extract_data(mouse_ID, date)
            design_matrices, predictor_names, predictor_info = \
                generate_predictors(glm_data_dict)
            X, Y, metadata, group_indices = stack_design_matrices(
                glm_data_dict, design_matrices, predictor_info,
                neural_metric=metric,
            )
            save_glm_data(
                X, Y, design_matrices, predictor_names, predictor_info,
                metadata, group_indices, mouse_ID, date,
            )
            ok_msg = f'    ✓ saved  {mouse_ID}/{date}'
            print(ok_msg)
            if log_path:
                with open(log_path, 'a') as f:
                    f.write(ok_msg + '\n')

        except Exception as exc:
            err_msg = f'    ✗ FAILED  {mouse_ID}/{date} — {exc}'
            print(err_msg)
            if log_path:
                with open(log_path, 'a') as f:
                    f.write(err_msg + '\n')
            raise SystemExit(1) from exc

        finally:
            del glm_data_dict, design_matrices, X, Y
            del predictor_names, predictor_info, metadata, group_indices
            gc.collect()

    done_msg = f'[pre-flight] ✓ All stacked data ready.\n'
    print(done_msg)
    if log_path:
        with open(log_path, 'a') as f:
            f.write(done_msg)


# ── Subprocess helpers ────────────────────────────────────────────────────────

def _run_subprocess(cmd, label, master_log_path):
    """
    Run cmd as a subprocess, stream output live, capture to master log.
    Returns (exit_code, elapsed_str).

    Thread-safe: all writes to master_log_path and to stdout headers/footers
    are protected by _LOG_LOCK.  Per-line stdout output is not locked so lines
    from parallel workers will interleave in the terminal — this is expected.
    The log file itself is written atomically per job (header at start, full
    output + footer at end) so the log stays readable.
    """
    t0 = datetime.now()
    sep = '─' * 60
    header = f'\n{sep}\n  {label}\n  [{t0.strftime("%H:%M:%S")}]\n{sep}\n'

    with _LOG_LOCK:
        print(header, flush=True)
        with open(master_log_path, 'a') as f:
            f.write(header)
            f.write(f'  cmd: {" ".join(cmd)}\n\n')

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines = []
    for line in proc.stdout:
        print(line, end='', flush=True)
        output_lines.append(line)
    proc.wait()

    elapsed   = str(datetime.now() - t0).split('.')[0]
    exit_code = proc.returncode
    status    = '✓ SUCCESS' if exit_code == 0 else f'✗ FAILED (exit {exit_code})'
    footer    = f'\n  {status}  [{elapsed}]\n{sep}\n'

    with _LOG_LOCK:
        print(footer, flush=True)
        with open(master_log_path, 'a') as f:
            f.writelines(output_lines)
            f.write(footer)

    return exit_code, elapsed


def run_single_session(model, mouse_ID, date, extra_args, master_log_path):
    """Session-first: one subprocess per (session, model) pair."""
    label = f'{model["name"]}  |  {mouse_ID} / {date}'
    cmd   = [sys.executable, _HELPER,
             model['module'], mouse_ID, date] + extra_args
    return _run_subprocess(cmd, label, master_log_path)


def run_full_model(model, extra_args, master_log_path):
    """Model-first: one subprocess per model (all sessions inside)."""
    label = (f'{model["name"]}  |  contexts={model["contexts"]}  '
             f'coupling={model["coupling"]}')
    cmd   = [sys.executable,
             os.path.join(GLM_DIR, model['script'])] + extra_args
    return _run_subprocess(cmd, label, master_log_path)


# ── Utilities ─────────────────────────────────────────────────────────────────

def list_models():
    print('\nAvailable models (default run order):')
    print(f'  {"Name":<32}  {"Contexts":<14}  Coupling')
    print('  ' + '─' * 55)
    for m in ALL_MODELS:
        print(f'  {m["name"]:<32}  {str(m["contexts"]):<14}  '
              f'{"yes" if m["coupling"] else "no"}')
    print()


def _init_master_log(log_path, selected, mode, extra_args):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        f.write(f'run_all_models.py — started {datetime.now()}\n')
        f.write(f'Mode: {mode}\n')
        f.write(f'Models ({len(selected)}): '
                f'{", ".join(m["name"] for m in selected)}\n')
        sessions_total = sum(len(v) for v in PPC_AVC.values())
        f.write(f'Sessions: {len(PPC_AVC)} mice | {sessions_total} total\n')
        f.write(f'extra_args: {extra_args}\n')
        f.write('=' * 70 + '\n\n')
    print(f'\n[master log] {log_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run any subset of the 6 GLM model scripts.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--models', nargs='+', metavar='MODEL',
        help=f'Models to run (space-separated). '
             f'Available: {", ".join(_NAME_TO_MODEL)}.')
    parser.add_argument(
        '--no-coupling', action='store_true',
        help='Skip all coupling variants.')
    parser.add_argument(
        '--model-first', action='store_true',
        help='Model-first ordering: complete each model across all sessions '
             'before starting the next model.  Default is session-first.')
    parser.add_argument(
        '--no-skip', action='store_true',
        help='Force re-run (passed through to each model script).')
    parser.add_argument(
        '--metric', type=str, default='deconv',
        help='Neural metric: deconv | dff | z_dff  (default: deconv).')
    parser.add_argument(
        '--max-workers', type=int, default=2,
        help='Number of parallel subprocesses in session-first mode (default: 2). '
             'Recommended: 2–3 on M2 Ultra with Metal GPU. '
             'Use 1 for fully sequential behaviour. '
             'Has no effect in --model-first mode.')
    parser.add_argument(
        '--build-stacked', action='store_true',
        help='Before running models, check all sessions for a missing '
             'glm_data_stacked.pkl and build any that are absent. '
             'Safe to pass even when all files already exist (no-op).')
    parser.add_argument(
        '--list', action='store_true',
        help='List available model names and exit.')
    args = parser.parse_args()

    if args.list:
        list_models()
        sys.exit(0)

    # ── Select models ──────────────────────────────────────────────────────
    if args.models:
        unknown = [n for n in args.models if n not in _NAME_TO_MODEL]
        if unknown:
            print(f'ERROR: unknown model name(s): {unknown}')
            print('Run with --list to see available names.')
            sys.exit(1)
        selected = [m for m in ALL_MODELS if m['name'] in args.models]
    else:
        selected = list(ALL_MODELS)

    if args.no_coupling:
        selected = [m for m in selected if not m['coupling']]

    if not selected:
        print('No models selected — nothing to run.')
        sys.exit(0)

    # ── Extra args forwarded to each model script ──────────────────────────
    extra_args = ['--metric', args.metric]
    if args.no_skip:
        extra_args.append('--no-skip')

    mode = 'model-first' if args.model_first else 'session-first'

    # ── Print plan ─────────────────────────────────────────────────────────
    sessions_total = sum(len(v) for v in PPC_AVC.values())
    print(f'\nMode: {mode}')
    print(f'Models queued ({len(selected)}):')
    for i, m in enumerate(selected, 1):
        print(f'  {i}. {m["name"]:<32}  contexts={m["contexts"]}  '
              f'coupling={m["coupling"]}')
    print(f'\nSessions: {len(PPC_AVC)} mice | {sessions_total} total')
    for mouse_ID, dates in PPC_AVC.items():
        print(f'  {mouse_ID}: {dates}')

    if mode == 'session-first':
        n_jobs = sessions_total * len(selected)
        print(f'\nSubprocesses to launch: {n_jobs}  '
              f'({sessions_total} sessions × {len(selected)} models)')
        print(f'Parallel workers: {args.max_workers}  '
              f'(already-completed sessions are auto-skipped)\n')

    # ── Master log ─────────────────────────────────────────────────────────
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(GLM_DIR, 'logs', f'run_all_models_{ts}.log')
    _init_master_log(log_path, selected, mode, extra_args)

    # ── Pre-flight: build missing stacked data ─────────────────────────────
    if args.build_stacked:
        missing = _find_missing_stacked(PPC_AVC)
        if missing:
            print(f'\n[pre-flight] {len(missing)} session(s) missing '
                  f'glm_data_stacked.pkl:')
            for mouse_ID, date in missing:
                print(f'  • {mouse_ID} / {date}')
            _build_stacked_for_sessions(missing, metric=args.metric,
                                        log_path=log_path)
        else:
            print('\n[pre-flight] All sessions have glm_data_stacked.pkl — '
                  'nothing to build.\n')

    results  = []
    wall_t0  = datetime.now()

    # ── SESSION-FIRST (default) ────────────────────────────────────────────
    if not args.model_first:
        print(f'{"=" * 70}')
        print(f'  SESSION-FIRST  —  {args.max_workers} parallel worker(s)  '
              f'—  memory fully cleared after each subprocess')
        print(f'{"=" * 70}\n')

        # Build the full flat job list: every (session × model) combination.
        all_jobs = [
            (model, mouse_ID, date)
            for mouse_ID, dates in PPC_AVC.items()
            for date in dates
            for model in selected
        ]

        results_lock = threading.Lock()   # guards append to `results` list

        def _run_job(job):
            model, mouse_ID, date = job
            return run_single_session(model, mouse_ID, date, extra_args, log_path)

        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            future_to_job = {pool.submit(_run_job, job): job for job in all_jobs}
            n_done = 0
            for future in as_completed(future_to_job):
                model, mouse_ID, date = future_to_job[future]
                exit_code, elapsed = future.result()
                n_done += 1
                with results_lock:
                    results.append(dict(
                        label=f'{model["name"]}  {mouse_ID}/{date}',
                        exit_code=exit_code,
                        elapsed=elapsed,
                    ))
                icon = '✓' if exit_code == 0 else '✗'
                with _LOG_LOCK:
                    print(f'  {icon}  [{n_done}/{len(all_jobs)}]  '
                          f'{model["name"]}  {mouse_ID}/{date}  [{elapsed}]',
                          flush=True)

    # ── MODEL-FIRST ────────────────────────────────────────────────────────
    else:
        print(f'{"=" * 70}')
        print(f'  MODEL-FIRST  —  each model completes all sessions before the next')
        print(f'{"=" * 70}\n')

        for model in selected:
            exit_code, elapsed = run_full_model(model, extra_args, log_path)
            results.append(dict(
                label=model['name'],
                exit_code=exit_code,
                elapsed=elapsed,
            ))

    # ── Final summary ──────────────────────────────────────────────────────
    total_elapsed = str(datetime.now() - wall_t0).split('.')[0]
    n_ok   = sum(1 for r in results if r['exit_code'] == 0)
    n_fail = len(results) - n_ok

    lines = [
        '',
        '=' * 70,
        f'ALL DONE  [{total_elapsed} total wall time]',
        f'  ✓ {n_ok} succeeded   ✗ {n_fail} failed   ({len(results)} total)',
        '',
    ]
    for r in results:
        icon = '✓' if r['exit_code'] == 0 else '✗'
        lines.append(f'  {icon}  {r["label"]:<50} [{r["elapsed"]}]')
    lines.append('=' * 70)
    summary = '\n'.join(lines) + '\n'

    print(summary)
    with open(log_path, 'a') as f:
        f.write(summary)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
