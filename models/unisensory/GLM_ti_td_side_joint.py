#!/usr/bin/env python
"""
GLM_ti_td_side_joint.py
=======================
Joint version of GLM_ti_td_side_coupling.py, for a coupling-WEIGHT readout that matches the FEF single-neuron model.

The main run fits one model per source block and reads the coupling GAIN (ΔFDE) of each.  Here all five blocks of the
SAME sources go into ONE fit per fold,
    [behaviour | TI_L | TI_R | TD_L | TD_R | CTL]      (single-neuron predictors, one group-lasso group per source neuron)
and the readout is the fitted coupling weight of every target <- source pair (signed sum and group norm of the 4 lag
weights) — the quantity the FEF model reports, so both species can be drawn on the same axes.

Sources and targets are read from the main run's `side_fold{k}_design.pkl` (identical neurons, identical split), so a
target is never a source.  The same-batch base fit is reused from the main run.  Requires the main run to be finished.

Outputs  ({BASE_DATA_PATH}/{mouse}/{date}/GLM_ti_td_side_results/)
    side_fold{k}_JOINT_full.pkl       the joint fit
    side_joint_session.pkl            long table: one row per (target, source, fold)

Usage
-----
python GLM_ti_td_side_joint.py                         # every session that has a finished main run
python GLM_ti_td_side_joint.py --sessions IS-2-1L/2025-03-10
"""
import argparse
import gc
import os
import pickle
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
import GLM_ti_td_side_coupling as M                                        # noqa: E402
from GLM_ti_td_side_coupling import BASE_DATA_PATH, PPC_AVC, open_glm_stacked_data   # noqa: E402

MODEL_DIR = M.VARIANTS['main']


def run_session(mouse_ID, date, skip_existing=True, log=print):
    d = os.path.join(BASE_DATA_PATH, mouse_ID, date, MODEL_DIR)
    out = os.path.join(d, 'side_joint_session.pkl')
    if not os.path.exists(os.path.join(d, f'{M.PREFIX}_session.pkl')):
        return 'no main run'
    if skip_existing and os.path.exists(out):
        log(f'[SKIP] {mouse_ID}/{date}')
        return 'skipped'
    with open(os.path.join(d, f'{M.PREFIX}_session.pkl'), 'rb') as f:
        S = pickle.load(f)
    side_of = {(int(t), int(k)): s for t, k, s in zip(S['target_idx'], S['fold'], S['side'])}
    dprime = {int(t): float(v) for t, v in zip(S['target_idx'], S['dprime'])}

    X, Y, metadata, _, pred_info, _ = open_glm_stacked_data(mouse_ID, date, print_summary=False)
    m = np.isin(metadata['context'], M.CONTEXTS)
    X, Y, metadata = X[m], Y[m], {k: v[m] for k, v in metadata.items()}
    d_all = M.choice_side(Y, metadata)                                     # side of the SOURCES (same definition as the main run)
    rows = []
    for k in (1, 2):
        with open(os.path.join(d, f'{M.PREFIX}_fold{k}_design.pkl'), 'rb') as f:
            D = pickle.load(f)
        targets, tcls = np.asarray(D['targets'], int), np.asarray(D['target_cls']).astype(str)
        blocks = [(b, int(s)) for b in M.BLOCKS for s in D['sources'][b]]
        src = np.array([s for _, s in blocks], int)
        assert not (set(targets) & set(src)), 'target inside a source block'
        pi = dict(pred_info)
        for j, (b, s) in enumerate(blocks):
            pi[f'cpl_{b}_n{s}'] = {'start': X.shape[1] + j * M.N_LAG_BASES, 'end': X.shape[1] + (j + 1) * M.N_LAG_BASES,
                                   'names': [f'cpl_{b}_n{s}_lag{q}' for q in range(M.N_LAG_BASES)]}
        log(f'  fold {k}: {len(targets)} targets | {len(src)} source neurons in one fit ({D["n_block"]} per block)')
        res = M.fit_model(np.hstack([X, M.lag_features(Y[:, src].astype(np.float32))]), Y[:, targets], metadata, pi, mouse_ID, date,
                          MODEL_DIR, f'{M.PREFIX}_fold{k}_JOINT', skip_existing, log)
        with open(os.path.join(d, f'{M.PREFIX}_fold{k}_base_full.pkl'), 'rb') as f:
            base = pickle.load(f)
        assert np.array_equal(res['combined_test_mask'], base['combined_test_mask']), 'joint fit scored on different test trials'
        W = np.asarray(res['weights'])[X.shape[1]:, :].reshape(len(src), M.N_LAG_BASES, len(targets))   # source x lag x target
        gain = np.asarray(res['test_frac_dev']) - np.asarray(base['test_frac_dev'])
        for ti, (t, c) in enumerate(zip(targets, tcls)):
            st = side_of[(int(t), k)]
            for j, (b, s) in enumerate(blocks):
                ss = 'R' if d_all[s] > 0 else 'L'
                rows.append(dict(target=int(t), target_cls=c, side_t=st, fold=k, source=s, block=b, src_cls=b[:2] if b != 'CTL' else 'CTL',
                                 same_side=(ss == st), w_sum=float(W[j, :, ti].sum()), w_norm=float(np.linalg.norm(W[j, :, ti])),
                                 joint_gain=float(gain[ti]), n_block=int(D['n_block']), dprime_t=dprime[int(t)]))
        gc.collect()
    T = pd.DataFrame(rows)
    T['unit'], T['animal'] = f'{mouse_ID}_{date}', mouse_ID
    T.to_pickle(out)
    g = T[T.target_cls == 'TI'].assign(cat=lambda x: np.where(x.src_cls == 'CTL', 'control', x.src_cls + np.where(x.same_side, ' same', ' opp')))
    log('    TI targets, mean signed weight: ' + '  '.join(f'{c} {v:+.4f}' for c, v in g.groupby('cat').w_sum.mean().items()))
    del X, Y, metadata
    gc.collect()
    return 'success'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', nargs='*', default=None)
    ap.add_argument('--no-skip', action='store_true')
    a = ap.parse_args()
    sessions = [(m, d) for m, ds in PPC_AVC.items() for d in ds]
    if a.sessions:
        want = {s for x in a.sessions for s in x.split(',') if s}
        sessions = [(m, d) for m, d in sessions if f'{m}/{d}' in want]
    log_dir = os.path.join(BASE_DATA_PATH, 'ti_td_split_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'GLM_ti_td_side_joint_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}.log')

    def log(line):
        stamp = f'[{datetime.now():%H:%M:%S}] {line}'
        print(stamp, flush=True)
        with open(log_path, 'a') as f:
            f.write(stamp + '\n')

    log(f'GLM_ti_td_side_joint: {len(sessions)} sessions | one joint fit per fold, blocks {M.BLOCKS}')
    for m, d in sessions:
        t0 = datetime.now()
        try:
            st = run_session(m, d, skip_existing=not a.no_skip, log=log)
        except Exception as e:
            traceback.print_exc()
            st = f'error: {e}'
        if st != 'no main run':
            log(f'  -> {m}/{d}: {st}  ({(datetime.now() - t0).total_seconds()/60:.1f} min)')


if __name__ == '__main__':
    main()
