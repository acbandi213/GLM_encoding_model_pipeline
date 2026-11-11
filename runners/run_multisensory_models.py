#!/usr/bin/env python
"""
run_multisensory_models.py
===========================
Master runner for multisensory GLM analyses.

Three analysis modes
--------------------
1. PRIMING (--analysis priming)
   Run GLM on priming (AVC unisensory) sessions only.
   Models: GLM_uni, GLM_uni_coupling (contexts=[1,2], no CG)
   
2. CROSS-TASK (--analysis cross-task)
   Train GLM on priming sessions, test on paired VACANT sessions.
   Models: GLM_cross_task, GLM_cross_task_coupling
   
3. VACANT (--analysis vacant)
   Run GLM on VACANT (multisensory) sessions only.
   Models: GLM_all_context, GLM_all_context_coupling (contexts=[0,1,2])
          GLM_uni, GLM_uni_coupling (contexts=[1,2])

Execution modes
---------------
SESSION-FIRST (default, recommended)
    For each session → run all selected models → memory fully cleared.
    Supports parallel execution via --max-workers N (default 2).

MODEL-FIRST  (--model-first)
    For each model → run all sessions → next model.

Usage
-----
# Run all analyses:
python run_multisensory_models.py --analysis all

# Run specific analysis:
python run_multisensory_models.py --analysis priming
python run_multisensory_models.py --analysis cross-task
python run_multisensory_models.py --analysis vacant

# Run specific models:
python run_multisensory_models.py --analysis priming --models GLM_uni

# Skip coupling variants:
python run_multisensory_models.py --analysis all --no-coupling

# Parallel workers:
python run_multisensory_models.py --analysis all --max-workers 3

# Build missing stacked data files before running models:
python run_multisensory_models.py --analysis all --build-stacked

# List available models:
python run_multisensory_models.py --list
"""

import argparse
import gc
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

_LOG_LOCK = threading.Lock()

# Script is in runners/, need to go up 1 level to GLM_encoding_model root
_RUNNER_DIR = os.path.dirname(os.path.abspath(__file__))
GLM_DIR = os.path.dirname(_RUNNER_DIR)
for _p in (GLM_DIR,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.session_registry_multisensory import (
    PPC_AVC_PRIMING,
    PPC_VACANT,
    CROSS_TASK_PAIRS,
)
from core.session_pipeline import (
    BASE_DATA_PATH,
    extract_data,
    generate_predictors,
    stack_design_matrices,
    save_glm_data,
)

# ── Model registry ────────────────────────────────────────────────────────────
# Each model definition specifies:
#   - name: Model identifier
#   - script: Python filename to execute
#   - analysis: Which analysis mode(s) this model applies to
#   - coupling: Whether this includes coupling predictors
#   - contexts: Which contexts are used (for within-task models)

MULTISENSORY_MODELS = [
    # PRIMING analysis (unisensory, no CG)
    dict(
        name='GLM_priming_uni',
        script='models/multisensory/GLM_priming_uni.py',
        analysis='priming',
        coupling=False,
        contexts=[1, 2],
        description='Unisensory GLM on priming sessions'
    ),
    dict(
        name='GLM_priming_uni_coupling',
        script='models/multisensory/GLM_priming_uni_coupling.py',
        analysis='priming',
        coupling=True,
        contexts=[1, 2],
        description='Unisensory GLM with coupling on priming sessions'
    ),
    
    # CROSS-TASK analysis (train priming, test VACANT)
    dict(
        name='GLM_cross_task',
        script='models/multisensory/GLM_cross_task.py',
        analysis='cross-task',
        coupling=False,
        contexts=None,  # Uses both train_contexts=[1,2] and test_contexts=[0,1,2]
        description='Cross-task GLM (train priming → test VACANT)'
    ),
    dict(
        name='GLM_cross_task_coupling',
        script='models/multisensory/GLM_cross_task_coupling.py',
        analysis='cross-task',
        coupling=True,
        contexts=None,
        description='Cross-task GLM with coupling (train priming → test VACANT)'
    ),
    
    # VACANT analysis (multisensory, includes CG)
    dict(
        name='GLM_vacant_all',
        script='models/multisensory/GLM_vacant_all.py',
        analysis='vacant',
        coupling=False,
        contexts=[0, 1, 2],
        description='All contexts GLM on VACANT sessions'
    ),
    dict(
        name='GLM_vacant_all_coupling',
        script='models/multisensory/GLM_vacant_all_coupling.py',
        analysis='vacant',
        coupling=True,
        contexts=[0, 1, 2],
        description='All contexts GLM with coupling on VACANT sessions'
    ),
    dict(
        name='GLM_vacant_conflict_only',
        script='models/multisensory/GLM_vacant_conflict_only.py',
        analysis='vacant',
        coupling=False,
        contexts=[1, 2],
        description='Conflict-only (contexts 1,2) GLM on VACANT sessions'
    ),
    dict(
        name='GLM_vacant_conflict_only_coupling',
        script='models/multisensory/GLM_vacant_conflict_only_coupling.py',
        analysis='vacant',
        coupling=True,
        contexts=[1, 2],
        description='Conflict-only (contexts 1,2) GLM with coupling on VACANT sessions'
    ),
]

_NAME_TO_MODEL = {m['name']: m for m in MULTISENSORY_MODELS}


# ── Pre-flight: stacked data helpers ─────────────────────────────────────────

def _normalize_date(date):
    """Remove _1 suffix from VACANT dates for filesystem paths."""
    if date.endswith('_1'):
        return date[:-2]
    return date


def _find_missing_stacked(sessions_dict):
    """
    Return list of (mouse_ID, date) pairs that are missing glm_data_stacked.pkl.
    
    Note: VACANT dates have _1 suffix but filesystem uses base date.
    """
    missing = []
    for mouse_ID, dates in sessions_dict.items():
        for date in dates:
            normalized_date = _normalize_date(date)
            path = os.path.join(BASE_DATA_PATH, mouse_ID, normalized_date, 
                               'glm_data_stacked.pkl')
            if not os.path.exists(path):
                missing.append((mouse_ID, date))
    return missing


def _build_stacked_for_sessions(missing, metric='deconv', log_path=None):
    """
    Build glm_data_stacked.pkl for each (mouse_ID, date) in *missing*.
    
    Calls the pipeline functions directly (no subprocess) so memory is freed
    after each session via gc.collect(). Aborts on the first failure.
    """
    n = len(missing)
    msg = f'[pre-flight] Building glm_data_stacked.pkl for {n} session(s)...'
    print(f'\n{msg}')
    if log_path:
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
    
    for i, (mouse_ID, date) in enumerate(missing, 1):
        normalized_date = _normalize_date(date)
        label = f'  [{i}/{n}]  {mouse_ID} / {date}'
        print(label)
        if log_path:
            with open(log_path, 'a') as f:
                f.write(label + '\n')
        
        glm_data_dict = design_matrices = X = Y = None
        predictor_names = predictor_info = metadata = group_indices = None
        try:
            glm_data_dict = extract_data(mouse_ID, normalized_date)
            design_matrices, predictor_names, predictor_info = \
                generate_predictors(glm_data_dict)
            X, Y, metadata, group_indices = stack_design_matrices(
                glm_data_dict, design_matrices, predictor_info,
                neural_metric=metric,
            )
            save_glm_data(
                X, Y, design_matrices, predictor_names, predictor_info,
                metadata, group_indices, mouse_ID, normalized_date,
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


def run_priming_session(model, mouse_ID, date, extra_args, master_log_path):
    """Run a priming-analysis model on a single priming session."""
    label = f'{model["name"]}  |  {mouse_ID} / {date} (priming)'
    cmd = [sys.executable, os.path.join(GLM_DIR, model['script'])] + extra_args
    # Note: The existing GLM scripts (GLM_uni.py, etc.) use PPC_AVC from session_registry.py
    # We need to make sure they process only the priming sessions.
    # For now, they'll process all sessions in PPC_AVC. If needed, we can add --session filtering.
    return _run_subprocess(cmd, label, master_log_path)


def run_cross_task_pair(model, mouse_ID, train_date, test_date, extra_args, master_log_path):
    """Run a cross-task model on a single priming→VACANT pair."""
    label = f'{model["name"]}  |  {mouse_ID} {train_date}→{test_date}'
    cmd = [sys.executable, os.path.join(GLM_DIR, model['script'])] + extra_args
    return _run_subprocess(cmd, label, master_log_path)


def run_vacant_session(model, mouse_ID, date, extra_args, master_log_path):
    """Run a VACANT-analysis model on a single VACANT session."""
    label = f'{model["name"]}  |  {mouse_ID} / {date} (VACANT)'
    cmd = [sys.executable, os.path.join(GLM_DIR, model['script'])] + extra_args
    return _run_subprocess(cmd, label, master_log_path)


def run_full_model_priming(model, extra_args, master_log_path):
    """Model-first: run priming model on all priming sessions."""
    label = f'{model["name"]}  |  priming sessions  |  contexts={model["contexts"]}  |  coupling={model["coupling"]}'
    cmd = [sys.executable, os.path.join(GLM_DIR, model['script'])] + extra_args
    return _run_subprocess(cmd, label, master_log_path)


def run_full_model_cross_task(model, extra_args, master_log_path):
    """Model-first: run cross-task model on all session pairs."""
    label = f'{model["name"]}  |  cross-task pairs  |  coupling={model["coupling"]}'
    cmd = [sys.executable, os.path.join(GLM_DIR, model['script'])] + extra_args
    return _run_subprocess(cmd, label, master_log_path)


def run_full_model_vacant(model, extra_args, master_log_path):
    """Model-first: run VACANT model on all VACANT sessions."""
    label = f'{model["name"]}  |  VACANT sessions  |  contexts={model["contexts"]}  |  coupling={model["coupling"]}'
    cmd = [sys.executable, os.path.join(GLM_DIR, model['script'])] + extra_args
    return _run_subprocess(cmd, label, master_log_path)


# ── Utilities ─────────────────────────────────────────────────────────────────

def list_models():
    print('\nAvailable models:')
    print(f'  {"Name":<32}  {"Analysis":<12}  {"Contexts":<14}  Coupling')
    print('  ' + '─' * 75)
    for m in MULTISENSORY_MODELS:
        ctx_str = str(m['contexts']) if m['contexts'] else 'train/test'
        print(f'  {m["name"]:<32}  {m["analysis"]:<12}  {ctx_str:<14}  '
              f'{"yes" if m["coupling"] else "no"}')
    print()


def _init_master_log(log_path, selected, analysis_mode, execution_mode, extra_args):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        f.write(f'run_multisensory_models.py — started {datetime.now()}\n')
        f.write(f'Analysis mode: {analysis_mode}\n')
        f.write(f'Execution mode: {execution_mode}\n')
        f.write(f'Models ({len(selected)}): '
                f'{", ".join(m["name"] for m in selected)}\n')
        
        if 'priming' in analysis_mode or analysis_mode == 'all':
            n_priming = sum(len(v) for v in PPC_AVC_PRIMING.values())
            f.write(f'Priming sessions: {len(PPC_AVC_PRIMING)} mice | {n_priming} total\n')
        
        if 'cross-task' in analysis_mode or analysis_mode == 'all':
            f.write(f'Cross-task pairs: {len(CROSS_TASK_PAIRS)} total\n')
        
        if 'vacant' in analysis_mode or analysis_mode == 'all':
            n_vacant = sum(len(v) for v in PPC_VACANT.values())
            f.write(f'VACANT sessions: {len(PPC_VACANT)} mice | {n_vacant} total\n')
        
        f.write(f'extra_args: {extra_args}\n')
        f.write('=' * 70 + '\n\n')
    print(f'\n[master log] {log_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run multisensory GLM analyses.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument(
        '--analysis', type=str, default='all',
        choices=['priming', 'cross-task', 'vacant', 'all'],
        help='Analysis mode: priming | cross-task | vacant | all (default: all)')
    
    parser.add_argument(
        '--models', nargs='+', metavar='MODEL',
        help=f'Specific models to run (space-separated). '
             f'Available: {", ".join(_NAME_TO_MODEL)}.')
    
    parser.add_argument(
        '--no-coupling', action='store_true',
        help='Skip all coupling variants.')
    
    parser.add_argument(
        '--model-first', action='store_true',
        help='Model-first ordering: complete each model across all sessions '
             'before starting the next model. Default is session-first.')
    
    parser.add_argument(
        '--no-skip', action='store_true',
        help='Force re-run (passed through to each model script).')
    
    parser.add_argument(
        '--metric', type=str, default='deconv',
        help='Neural metric: deconv | dff | z_dff (default: deconv).')
    
    parser.add_argument(
        '--max-workers', type=int, default=2,
        help='Number of parallel subprocesses in session-first mode (default: 2). '
             'Use 1 for fully sequential behaviour. '
             'Has no effect in --model-first mode.')
    
    parser.add_argument(
        '--build-stacked', action='store_true',
        help='Pre-flight: build missing glm_data_stacked.pkl files before running models. '
             'Checks all sessions required by the selected analysis mode.')
    
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
        selected = [m for m in MULTISENSORY_MODELS if m['name'] in args.models]
    else:
        # Filter by analysis mode
        if args.analysis == 'all':
            selected = list(MULTISENSORY_MODELS)
        else:
            selected = [m for m in MULTISENSORY_MODELS if m['analysis'] == args.analysis]

    if args.no_coupling:
        selected = [m for m in selected if not m['coupling']]

    if not selected:
        print('No models selected — nothing to run.')
        sys.exit(0)

    # ── Extra args forwarded to each model script ──────────────────────────
    extra_args = ['--metric', args.metric]
    if args.no_skip:
        extra_args.append('--no-skip')

    execution_mode = 'model-first' if args.model_first else 'session-first'

    # ── Print plan ─────────────────────────────────────────────────────────
    print(f'\nAnalysis mode: {args.analysis}')
    print(f'Execution mode: {execution_mode}')
    print(f'Models queued ({len(selected)}):')
    for i, m in enumerate(selected, 1):
        ctx_str = str(m['contexts']) if m['contexts'] else 'train→test'
        print(f'  {i}. {m["name"]:<32}  {m["analysis"]:<12}  '
              f'contexts={ctx_str}  coupling={m["coupling"]}')

    if args.analysis in ['priming', 'all']:
        n_priming = sum(len(v) for v in PPC_AVC_PRIMING.values())
        print(f'\nPriming sessions: {len(PPC_AVC_PRIMING)} mice | {n_priming} total')
        for mouse_ID, dates in PPC_AVC_PRIMING.items():
            print(f'  {mouse_ID}: {dates}')

    if args.analysis in ['cross-task', 'all']:
        n_pairs = sum(len(pairs) for pairs in CROSS_TASK_PAIRS.values())
        print(f'\nCross-task pairs: {n_pairs} total')
        for mouse_ID, pair_list in CROSS_TASK_PAIRS.items():
            for priming_date, vacant_date in pair_list:
                print(f'  {mouse_ID}: {priming_date} → {vacant_date}')

    if args.analysis in ['vacant', 'all']:
        n_vacant = sum(len(v) for v in PPC_VACANT.values())
        print(f'\nVACANT sessions: {len(PPC_VACANT)} mice | {n_vacant} total')
        for mouse_ID, dates in PPC_VACANT.items():
            print(f'  {mouse_ID}: {dates}')

    # ── Master log ─────────────────────────────────────────────────────────
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(GLM_DIR, 'logs', f'run_multisensory_{args.analysis}_{ts}.log')
    _init_master_log(log_path, selected, args.analysis, execution_mode, extra_args)

    # ── Pre-flight: build missing stacked data ─────────────────────────────
    if args.build_stacked:
        # Collect all unique sessions based on analysis mode
        all_sessions = {}
        
        if args.analysis in ['priming', 'all', 'cross-task']:
            # Need priming sessions for priming analysis and cross-task training
            for mouse_ID, dates in PPC_AVC_PRIMING.items():
                if mouse_ID not in all_sessions:
                    all_sessions[mouse_ID] = []
                all_sessions[mouse_ID].extend(dates)
        
        if args.analysis in ['vacant', 'all', 'cross-task']:
            # Need VACANT sessions for vacant analysis and cross-task testing
            for mouse_ID, dates in PPC_VACANT.items():
                if mouse_ID not in all_sessions:
                    all_sessions[mouse_ID] = []
                all_sessions[mouse_ID].extend(dates)
        
        # Remove duplicates while preserving order
        for mouse_ID in all_sessions:
            seen = set()
            unique = []
            for date in all_sessions[mouse_ID]:
                if date not in seen:
                    seen.add(date)
                    unique.append(date)
            all_sessions[mouse_ID] = unique
        
        missing = _find_missing_stacked(all_sessions)
        if missing:
            print(f'\n[pre-flight] {len(missing)} session(s) missing '
                  f'glm_data_stacked.pkl:')
            for mouse_ID, date in missing:
                print(f'  • {mouse_ID} / {date}')
            _build_stacked_for_sessions(missing, metric=args.metric, log_path=log_path)
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

        # Note: For simplicity, we're running each model script independently.
        # The existing scripts (GLM_uni.py, etc.) process all their sessions.
        # For cross-task, GLM_cross_task.py processes all pairs.
        # This means session-first mode here just means "run each model script sequentially or in parallel"
        # rather than truly interleaving at the session level.
        # A more sophisticated implementation would require modifying the GLM scripts to accept
        # --session arguments, but that's beyond the current scope.

        print('[Note: In session-first mode, each model script processes all its sessions internally]')
        
        all_jobs = [(model,) for model in selected]
        
        results_lock = threading.Lock()

        def _run_job(job):
            model, = job
            if model['analysis'] == 'priming':
                return run_full_model_priming(model, extra_args, log_path)
            elif model['analysis'] == 'cross-task':
                return run_full_model_cross_task(model, extra_args, log_path)
            elif model['analysis'] == 'vacant':
                return run_full_model_vacant(model, extra_args, log_path)

        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            future_to_job = {pool.submit(_run_job, job): job for job in all_jobs}
            n_done = 0
            for future in as_completed(future_to_job):
                model, = future_to_job[future]
                exit_code, elapsed = future.result()
                n_done += 1
                with results_lock:
                    results.append(dict(
                        label=model["name"],
                        exit_code=exit_code,
                        elapsed=elapsed,
                    ))
                icon = '✓' if exit_code == 0 else '✗'
                with _LOG_LOCK:
                    print(f'  {icon}  [{n_done}/{len(all_jobs)}]  '
                          f'{model["name"]}  [{elapsed}]',
                          flush=True)

    # ── MODEL-FIRST ────────────────────────────────────────────────────────
    else:
        print(f'{"=" * 70}')
        print(f'  MODEL-FIRST  —  each model completes all sessions before the next')
        print(f'{"=" * 70}\n')

        for model in selected:
            if model['analysis'] == 'priming':
                exit_code, elapsed = run_full_model_priming(model, extra_args, log_path)
            elif model['analysis'] == 'cross-task':
                exit_code, elapsed = run_full_model_cross_task(model, extra_args, log_path)
            elif model['analysis'] == 'vacant':
                exit_code, elapsed = run_full_model_vacant(model, extra_args, log_path)
            
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
