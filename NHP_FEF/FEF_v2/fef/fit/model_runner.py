"""
model_runner.py
===============
The shared engine behind the FEF model scripts in `FEF_v2/models/` (the
mouse-organization analog: each model script declares an identity block and
this module supplies the machinery the mouse scripts carried inline —
resumable per-probe results, master log, lesion loop, and the M2-Ultra
parallel driver).

Stage layout (mirrors the mouse):

    stacked cache  →  [filter contexts]  →  [+ coupling]  →
    prepare_features  →  fit_glm_single_session  →  run_lesion_models

Parallelism: fitting is TensorFlow-on-CPU, so the driver runs N probes as
independent SUBPROCESSES (fresh TF per probe, memory fully returned) with
per-worker thread caps — `--workers 6` × 4 threads is a good default on the
24-core M2 Ultra. The stacked caches are plain pickles and parallel-safe to
read; only the .mat stage (Stage 1, wp6_build_stacked.py) must stay serial.

Results land at {RESULTS_DIR}/fits/{probe_key}/{model_dir}/ via the core
`fit_glm_single_session` saver, so every downstream mouse-style loader works
unchanged.
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

_V2 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', '..'))
_REPO = os.path.dirname(os.path.dirname(_V2))
for _p in (_V2, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Thread caps must be set BEFORE TensorFlow initializes. The driver exports
# FEF_TF_THREADS to each worker; a worker importing this module applies it.
_TF_THREADS = int(os.environ.get('FEF_TF_THREADS', '0'))
if _TF_THREADS:
    os.environ.setdefault('OMP_NUM_THREADS', str(_TF_THREADS))
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', str(_TF_THREADS))

from fef import config as C                              # noqa: E402
from fef.fit import stacking                             # noqa: E402
from fef.fit.invariants import (PRODUCTION_FIT_KWARGS,   # noqa: E402
                                assert_production_balancing)

CONTEXT_NAMES = {C.CTX_SACCADE: 'saccade', C.CTX_PURSUIT: 'pursuit'}


def fits_dir():
    return os.path.join(C.RESULTS_DIR, 'fits')


def _result_dir(model_cfg, key):
    return os.path.join(fits_dir(), key, model_cfg['model_dir'])


def expected_lesion_groups(model_cfg, stack):
    """Per-probe lesion-group list — derived from the probe's OWN split
    predictor_info (Monkey Y designs lack `jump`; a global constant would
    mislabel resume state), plus the coupling groups when enabled."""
    groups = list(stack['predictor_info_split'].keys())
    if model_cfg['include_coupling']:
        groups += [f'network_pc{i + 1}'
                   for i in range(model_cfg['coupling_params']['n_pcs'])]
    return groups


def check_probe_results(model_cfg, key, stack=None):
    """Completion status for one probe (mouse check_session_results analog).
    Without `stack`, lesion completeness falls back to 'full model exists'."""
    d = _result_dir(model_cfg, key)
    pfx = model_cfg['model_prefix']
    full = os.path.exists(os.path.join(d, f'{pfx}_full.pkl'))
    missing = []
    if stack is not None:
        for g in expected_lesion_groups(model_cfg, stack):
            if not os.path.exists(os.path.join(d, f'{pfx}_lesion_{g}.pkl')):
                missing.append(g)
    return dict(full_model=full, missing_lesion_groups=missing,
                all_complete=full and not missing)


def fit_one_probe(model_cfg, key, stages=('base', 'lesions'),
                  skip_existing=True):
    """
    Fit one probe end-to-end inside THIS process. The parallel driver calls
    this in a subprocess; call it directly for debugging.
    """
    import core.session_pipeline as sp
    from core.session_pipeline import (add_coupling_predictor,
                                       prepare_features_for_glm,
                                       run_lesion_models)
    if _TF_THREADS:
        try:
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(_TF_THREADS)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except (ImportError, RuntimeError):
            pass

    t0 = time.time()
    stack = stacking.load_probe_stack(key)
    X, Y, meta = stack['X'], stack['Y'], stack['metadata']
    prov = stack['provenance']
    print(f"[{model_cfg['model_name']}] {key}: X {X.shape}, "
          f"Y {Y.shape} (monkey {prov['monkey']}, "
          f"window +{prov['win_post']:.0f} ms, "
          f"dropped {list(prov['dropped_groups']) or 'none'})")

    # contexts filter (both tasks by default — kept for mouse-shape fidelity)
    mask = np.isin(meta['context'], model_cfg['contexts'])
    X, Y = X[mask], Y[mask]
    meta = {k: v[mask] for k, v in meta.items()}

    pinfo = stack['predictor_info']
    if model_cfg['include_coupling']:
        X, pinfo, _ = add_coupling_predictor(
            X, Y, pinfo, **model_cfg['coupling_params'])
    import contextlib, io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        pinfo_split, fgs = prepare_features_for_glm(pinfo)
    if int(fgs.sum()) != X.shape[1]:
        raise ValueError('feature_group_size mismatch after staging')

    fit_kwargs = dict(PRODUCTION_FIT_KWARGS, context_names=CONTEXT_NAMES)
    assert_production_balancing(fit_kwargs,
                                where=f"{model_cfg['model_name']}/{key}")

    sp.BASE_DATA_PATH = fits_dir()
    d = _result_dir(model_cfg, key)
    pfx = model_cfg['model_prefix']
    full_path = os.path.join(d, f'{pfx}_full.pkl')

    if 'base' in stages and not (skip_existing
                                 and os.path.exists(full_path)):
        model_cv, res, _, _ = sp.fit_glm_single_session(
            X, Y, meta, fgs, pinfo_split, key, model_cfg['model_dir'],
            model_dir='.', model_prefix=pfx, **fit_kwargs)
        # core saves to BASE/key/model_dir/./prefix_full.pkl — flatten the '.'
        del model_cv
        gc.collect()
    else:
        with open(full_path, 'rb') as f:
            res = pickle.load(f)
        print(f'  base model loaded from cache ({full_path})')

    if 'lesions' in stages:
        run_lesion_models(
            X, Y, meta, res, pinfo_split, mouse_ID=key,
            date=model_cfg['model_dir'], skip_existing=skip_existing,
            model_dir='.', model_prefix=pfx)

    print(f"[{model_cfg['model_name']}] {key} done in "
          f'{(time.time() - t0) / 60:.1f} min')


# ── master log (mouse MasterLog, trimmed) ────────────────────────────────────
class MasterLog:
    def __init__(self, path, header):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(header + f'\nstarted {datetime.now()}\n' + '=' * 70 + '\n')
        print(f'[log] {path}')

    def write(self, line):
        with open(self.path, 'a') as f:
            f.write(f'[{datetime.now().strftime("%H:%M:%S")}] {line}\n')
        print(line)


def _run_worker(model_cfg, key, stages, skip_existing, script_path,
                tf_threads, log_dir):
    """Launch one probe fit as a subprocess; stream its output to a file."""
    log_path = os.path.join(log_dir, f'{key}.log')
    cmd = [sys.executable, script_path, '--probe-key', key,
           '--stages', *stages]
    if not skip_existing:
        cmd.append('--no-skip')
    env = dict(os.environ, FEF_TF_THREADS=str(tf_threads),
               OMP_NUM_THREADS=str(tf_threads),
               VECLIB_MAXIMUM_THREADS=str(tf_threads))
    with open(log_path, 'w') as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return key, r.returncode, log_path


def main(model_cfg, script_path):
    """The CLI shared by every model script. Two modes:
      * --probe-key KEY : fit that probe in-process (the worker mode)
      * default         : parallel driver over every cached probe
    """
    ap = argparse.ArgumentParser(description=model_cfg['model_name'])
    ap.add_argument('--probe-key', default=None,
                    help='fit one probe in-process (worker mode)')
    ap.add_argument('--workers', type=int, default=6,
                    help='parallel probe fits (subprocesses)')
    ap.add_argument('--tf-threads', type=int, default=4,
                    help='TF/BLAS threads per worker')
    ap.add_argument('--stages', nargs='+', default=['base', 'lesions'],
                    choices=['base', 'lesions'])
    ap.add_argument('--dev', action='store_true',
                    help='DEV4 probes only (S7 subset-first)')
    ap.add_argument('--monkey', choices=['S', 'Y'], default=None)
    ap.add_argument('--probes', nargs='*', default=None,
                    help='explicit probe keys')
    ap.add_argument('--no-skip', action='store_true')
    a = ap.parse_args()
    skip = not a.no_skip

    if a.probe_key:                                    # worker mode
        fit_one_probe(model_cfg, a.probe_key, stages=tuple(a.stages),
                      skip_existing=skip)
        return

    keys = stacking.available_probe_keys()
    if not keys:
        sys.exit(f'no stacked caches in {stacking.stacked_dir()} — '
                 f'run scripts/wp6_build_stacked.py first')
    if a.probes:
        keys = [k for k in keys if k in set(a.probes)]
    if a.dev:
        keys = [k for k in keys
                if any(k.startswith(s + '_') for s in C.DEV4_SESSIONS)]
    if a.monkey:
        pref = (C.MONKEY_S_SESSIONS if a.monkey == 'S'
                else C.MONKEY_Y_SESSIONS)
        keys = [k for k in keys if any(k.startswith(s + '_') for s in pref)]

    # resume filter: skip probes whose requested stages are already done.
    # (Lesion completeness needs the probe's own group list, so when lesions
    # are requested the WORKER resumes per-group via run_lesion_models'
    # skip_existing — the driver only prunes the base-only case.)
    todo = []
    for k in keys:
        st = check_probe_results(model_cfg, k)
        if skip and st['full_model'] and 'lesions' not in a.stages:
            continue
        todo.append(k)

    tag = model_cfg['model_tag']
    log_dir = os.path.join(_V2, 'logs', model_cfg['model_name'])
    os.makedirs(log_dir, exist_ok=True)
    log = MasterLog(
        os.path.join(_V2, 'logs',
                     f'run_{tag}_{datetime.now():%Y%m%d_%H%M%S}.log'),
        f"{model_cfg['model_name']} | coupling="
        f"{model_cfg['include_coupling']} | stages={a.stages} | "
        f'{len(todo)}/{len(keys)} probes | workers={a.workers} '
        f'x {a.tf_threads} threads')

    t0 = time.time()
    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_run_worker, model_cfg, k, a.stages, skip,
                          script_path, a.tf_threads, log_dir)
                for k in todo]
        for fut in futs:
            key, rc, lp = fut.result()
            if rc == 0:
                n_ok += 1
                log.write(f'  ✓ {key}')
            else:
                n_fail += 1
                log.write(f'  ✗ {key}  (exit {rc}; see {lp})')
    log.write(f'FINISHED: {n_ok} ok, {n_fail} failed in '
              f'{(time.time() - t0) / 3600:.1f} h')
    if n_fail:
        sys.exit(1)
