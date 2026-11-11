#!/usr/bin/env python
"""
verify_split_results.py
=======================
Audit every _full.pkl written by GLM_uni_coupling_split.py (any --mask) and delete
any that cannot be verified.

Why this exists
---------------
fit_glm_single_session writes _full.pkl ITSELF, before the runner's integrity
asserts can run. The runner deletes the artifact when an assert raises, but that
only covers AssertionError — a SIGKILL, a crash, or an unmounted drive in that
window leaves an unverified pkl on disk. check_session_results is existence-only,
so such a file is silently treated as a completed session forever and would feed
the analysis a fit whose train/test split was never confirmed identical to
GLM_uni (constraint 4).

Run this after any interrupted sweep, and before trusting an analysis.

Checks per session
------------------
1. loads, and has the fields the analysis reads
2. combined_train/test_mask + per-context test_masks IDENTICAL to GLM_uni_results
   (the split is a pure function of metadata + fixed seeds, so it MUST match;
   a mismatch means FDE differences are contaminated by split noise)
3. train and test do not overlap
4. the coupling split is live: w_group1 != w_group2 for at least some neurons
   (a no-op split would make every downstream number meaningless)
5. per-context FDE present for contexts 1 and 2

Usage
-----
python testing/verify_split_results.py                 # report only
python testing/verify_split_results.py --delete-bad    # delete unverifiable pkls
"""
import argparse
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
GLM_DIR = os.path.dirname(_HERE)
if GLM_DIR not in sys.path:
    sys.path.insert(0, GLM_DIR)

from core.session_pipeline import BASE_DATA_PATH
from config.session_registry import PPC_AVC

VARIANTS = {
    'context': ('GLM_uni_coupling_split_results', 'glm_uni_cpl_split', ('vis', 'aud')),
    'sham':    ('GLM_uni_coupling_sham_results',  'glm_uni_cpl_sham',  ('sham1', 'sham2')),
    'time':    ('GLM_uni_coupling_time_results',  'glm_uni_cpl_time',  ('early', 'late')),
}
REF_DIR, REF_PREFIX = 'GLM_uni_results', 'glm_uni'
N_PCS, N_LAG_BASES = 5, 4


def verify_one(path, ref, labels):
    """Return (ok, reason)."""
    try:
        with open(path, 'rb') as f:
            r = pickle.load(f)
    except Exception as e:
        return False, f'unpicklable ({type(e).__name__})'

    for k in ('weights', 'test_frac_dev', 'test_frac_dev_per_ctx', 'predictor_info',
              'combined_train_mask', 'combined_test_mask', 'test_masks'):
        if k not in r:
            return False, f'missing field {k!r}'

    for k in ('combined_train_mask', 'combined_test_mask'):
        if not np.array_equal(r[k], ref[k]):
            return False, f'{k} DIFFERS from {REF_DIR}'
    for ctx in ref['test_masks']:
        if ctx not in r['test_masks'] or not np.array_equal(r['test_masks'][ctx], ref['test_masks'][ctx]):
            return False, f'test_masks[{ctx}] DIFFERS from {REF_DIR}'
    if (r['combined_train_mask'] & r['combined_test_mask']).any():
        return False, 'train/test overlap'

    for c in (1, 2):
        if c not in r['test_frac_dev_per_ctx']:
            return False, f'no per-context FDE for context {c}'

    pi, w = r['predictor_info'], r['weights']
    try:
        c1 = np.concatenate([np.arange(pi[f'network_pc{i+1}_{labels[0]}']['start'],
                                       pi[f'network_pc{i+1}_{labels[0]}']['end']) for i in range(N_PCS)])
        c2 = np.concatenate([np.arange(pi[f'network_pc{i+1}_{labels[1]}']['start'],
                                       pi[f'network_pc{i+1}_{labels[1]}']['end']) for i in range(N_PCS)])
    except KeyError as e:
        return False, f'coupling group {e} missing from predictor_info'
    n_differ = int((np.abs(w[c1] - w[c2]).max(axis=0) > 1e-8).sum())
    if n_differ == 0:
        return False, 'split is a NO-OP (w_1 == w_2 for every neuron)'
    return True, f'ok ({n_differ}/{w.shape[1]} neurons differ)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--delete-bad', action='store_true',
                    help='delete pkls that fail verification so a re-run redoes them')
    args = ap.parse_args()
    if not os.path.isdir(BASE_DATA_PATH):
        raise SystemExit(f'  ✗ drive not mounted: {BASE_DATA_PATH}')

    total_ok = total_bad = 0
    for vname, (mdir, prefix, labels) in VARIANTS.items():
        rows = []
        for mouse, dates in PPC_AVC.items():
            for date in dates:
                p = os.path.join(BASE_DATA_PATH, mouse, date, mdir, f'{prefix}_full.pkl')
                if not os.path.exists(p):
                    continue
                ref_p = os.path.join(BASE_DATA_PATH, mouse, date, REF_DIR, f'{REF_PREFIX}_full.pkl')
                if not os.path.exists(ref_p):
                    rows.append((mouse, date, False, f'no {REF_DIR} reference', p))
                    continue
                with open(ref_p, 'rb') as f:
                    ref = pickle.load(f)
                ok, why = verify_one(p, ref, labels)
                rows.append((mouse, date, ok, why, p))
        if not rows:
            print(f'\n{vname:8s}: no results on disk yet')
            continue
        n_ok = sum(1 for r in rows if r[2])
        n_bad = len(rows) - n_ok
        total_ok += n_ok
        total_bad += n_bad
        print(f'\n{vname:8s}: {len(rows)} on disk | ✓ {n_ok} verified | ✗ {n_bad} FAILED')
        for mouse, date, ok, why, p in rows:
            if not ok:
                print(f'  ✗ {mouse}/{date}: {why}')
                if args.delete_bad:
                    os.remove(p)
                    print(f'      deleted {os.path.basename(p)} — a re-run will redo this session')

    print(f'\n{"=" * 66}')
    print(f'TOTAL: ✓ {total_ok} verified | ✗ {total_bad} failed')
    if total_bad and not args.delete_bad:
        print('  re-run with --delete-bad to remove them so the sweep redoes those sessions')
    if total_bad == 0:
        print('  every split/sham/time result on disk has a verified, GLM_uni-identical split')
    return 0 if total_bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
