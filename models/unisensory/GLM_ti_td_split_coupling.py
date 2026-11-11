#!/usr/bin/env python
"""
GLM_ti_td_split_coupling.py
===========================
Split-source coupling model for the snCCGP classes: are TI neurons coupled more to
other TI neurons than to TD neurons (and vice versa)?  Same machinery as
GLM_cd_split_coupling.py (no-CI default path) with the TI/TD labels swapped in.

Design (per session, contexts 1 + 2 = visual + audio only)
-----------------------------------------------------------
Labels: the Updated_Figure_4 snCCGP quartile classes, read from the dial cache
    {BASE_DATA_PATH}/analysis/_ti_td_snccgp_dial_cache/mouse_raw_index.csv   (half == 'all')
    TI  : snCCGP >= pooled 75th percentile of the Choice-class neurons
    TD  : snCCGP <= pooled 25th percentile
    Int : the rest (targets only, never a source)
Cuts are pooled over ALL sessions in the cache (as `classify()` in the notebook), so the
whole file is classified before a session is selected.

SPLIT-HALF so that no target is ever inside a source PCA (no self-prediction):
    TI -> halves T1, T2 ;  TD -> halves D1, D2          (fixed per-session seed)
    fold 1 : sources T1, D1 -> targets T2 U D2 U Int
    fold 2 : sources T2, D2 -> targets T1 U D1 U Int
The larger source half is subsampled to the size of the smaller, so both source blocks
are built from the same number of neurons.  Every TI/TD neuron is a target exactly once;
Int neurons are targets in both folds (one row per fold).

TWO joint fits per fold on the SAME seeded train/test trials (fit_glm_single_session):
    ti : [behaviour | TI 5 PCs]
    td : [behaviour | TD 5 PCs]
Each stores held-out FDE overall AND per test context.  The behaviour-only base (GLM_uni)
is scored on the identical test trials.

--null : label-shuffle control.  The session's TI U TD neurons are randomly re-partitioned
into pseudo-TI / pseudo-TD of the original sizes before the split; everything else is
identical.  Results dir GLM_ti_td_split_null_results.

Outputs  ({BASE_DATA_PATH}/{mouse}/{date}/{MODEL_DIR}/)
    {prefix}_fold{k}_{ti,td}_full.pkl      per-fold fits
    {prefix}_fold{k}_design.pkl            target / source indices for the fold
    {prefix}_session.pkl                   per-(target, fold) table merged over both folds
    {BASE_DATA_PATH}/ti_td_split_logs/GLM_ti_td_split_coupling_<timestamp>.log

Usage
-----
python GLM_ti_td_split_coupling.py                          # all PPC_AVC sessions, split seed 0
python GLM_ti_td_split_coupling.py --sessions IS-2-1L/2025-03-10 KO-5-1L/2025-11-20
python GLM_ti_td_split_coupling.py --split 1                # a second, independent neuron split
python GLM_ti_td_split_coupling.py --null                   # label-shuffle control
"""
import argparse
import gc
import os
import pickle
import sys
import traceback
import zlib
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    _HAS_TF = True
except ImportError:
    _HAS_TF = False

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GLM_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if GLM_DIR not in sys.path:
    sys.path.insert(0, GLM_DIR)

from core.session_pipeline import (
    BASE_DATA_PATH, open_glm_stacked_data, prepare_features_for_glm,
    fit_glm_single_session, _make_pc_lag_features,
)
from config.session_registry import PPC_AVC

def _cache_dir(name, base=BASE_DATA_PATH):
    """Analysis caches sit directly under ProcessedData (they were under ProcessedData/analysis until 2026-09-20)."""
    for d in (os.path.join(base, name), os.path.join(base, 'analysis', name)):
        if os.path.isdir(d):
            return d
    return os.path.join(base, name)


# ==============================================================================
MODEL_NAME = 'GLM_ti_td_split_coupling'
CONTEXTS   = [1, 2]                        # visual + audio only (label-model regime)

VARIANTS = {                               # null -> (results dir, file prefix stem)
    False: ('GLM_ti_td_split_results', 'titd'),
    True:  ('GLM_ti_td_split_null_results', 'titd_null'),
}

N_PCS, N_LAG_BASES, LAG_FRAMES = 5, 4, 8

LABEL_CACHE = os.path.join(_cache_dir('_ti_td_snccgp_dial_cache'), 'mouse_raw_index.csv')
LABEL_HALF  = 'all'
PROP_TI, PROP_TD = 0.25, 0.25
BASE_MODEL_DIR, BASE_MODEL_PREFIX = 'GLM_uni_results', 'GLM_uni'
MIN_PER_CLASS = 12                         # >= 6 per half -> 5 PCs from every source block
SPLIT_SEED    = 20260919
# ==============================================================================

_INDEX = None


def _filter_contexts(X, Y, metadata):
    m = np.isin(metadata['context'], CONTEXTS)
    return X[m], Y[m], {k: v[m] for k, v in metadata.items()}


def load_index():
    """The cached snCCGP table with the pooled quartile classes (Updated_Figure_4 `classify`)."""
    global _INDEX
    if _INDEX is None:
        idx = pd.read_csv(LABEL_CACHE)
        idx['half'] = idx.half.astype(str)
        idx = idx[(idx.half == LABEL_HALF) & np.isfinite(idx.snCCGP)].copy()
        hi, lo = float(idx.snCCGP.quantile(1 - PROP_TI)), float(idx.snCCGP.quantile(PROP_TD))
        idx['cls'] = np.where(idx.snCCGP >= hi, 'TI', np.where(idx.snCCGP <= lo, 'TD', 'Int'))
        idx.attrs['cuts'] = (hi, lo)
        _INDEX = idx
    return _INDEX


def get_labels(mouse_ID, date):
    """TI / TD / Int neuron indices from the snCCGP cache + base FDE from GLM_uni (contexts 1/2)."""
    idx = load_index()
    s = idx[idx.sid == f'{mouse_ID}_{date}']
    p = os.path.join(BASE_DATA_PATH, mouse_ID, date, BASE_MODEL_DIR, f'{BASE_MODEL_PREFIX}_full.pkl')
    with open(p, 'rb') as f:
        res = pickle.load(f)
    fde = np.asarray(res['test_frac_dev'], float)
    fv = np.asarray(res['test_frac_dev_per_ctx'][1], float)
    fa = np.asarray(res['test_frac_dev_per_ctx'][2], float)
    n = len(fde)
    sn, snw = np.full(n, np.nan), np.full(n, np.nan)
    sn[s.col.values], snw[s.col.values] = s.snCCGP.values, s.sn_within.values
    assert len(s) == 0 or s.col.max() < n, 'cached neuron index exceeds the GLM population'
    return dict(ti=np.sort(s.col.values[s.cls.values == 'TI']),
                td=np.sort(s.col.values[s.cls.values == 'TD']),
                int=np.sort(s.col.values[s.cls.values == 'Int']),
                snCCGP=sn, sn_within=snw, cuts=idx.attrs['cuts'],
                cpi=(fv - fa) / (fv + fa + 1e-8), fde_base=fde, fde_base_vis=fv, fde_base_aud=fa,
                base_test_mask=res.get('combined_test_mask'))


def make_split(labels, mouse_ID, date, split, null):
    """Random halves of TI and TD, fixed per-session seed.  --null re-partitions TI U TD first."""
    seed = (SPLIT_SEED + 1000 * split + zlib.crc32(f'{mouse_ID}/{date}'.encode())) % (2**32)
    rng = np.random.default_rng(seed)
    cls_idx = {'ti': labels['ti'], 'td': labels['td']}
    if null:
        pool = rng.permutation(np.concatenate([labels['ti'], labels['td']]))
        cls_idx = {'ti': pool[: len(labels['ti'])], 'td': pool[len(labels['ti']):]}
    halves = {}
    for cls in ('ti', 'td'):
        idx = rng.permutation(cls_idx[cls])
        halves[cls] = (np.sort(idx[: len(idx) // 2]), np.sort(idx[len(idx) // 2:]))
    return halves, seed, rng


def build_design(X, Y, pred_info, name, idx):
    """[X | block PCs] for one source block, lag-convolved and z-scored."""
    c, pi, _, k = _make_pc_lag_features(Y[:, idx], group_prefix=name, n_pcs=N_PCS,
                                        n_lag_bases=N_LAG_BASES, lag_frames=LAG_FRAMES,
                                        base_col_offset=X.shape[1])
    pinfo = dict(pred_info); pinfo.update(pi)
    return np.hstack([X, c]), pinfo, k


def fit_model(X_aug, Y_t, metadata, pinfo, mouse_ID, date, model_dir, prefix, skip_existing, log):
    """One joint fit via fit_glm_single_session (cached at {prefix}_full.pkl)."""
    full_path = os.path.join(BASE_DATA_PATH, mouse_ID, date, model_dir, f'{prefix}_full.pkl')
    if skip_existing and os.path.exists(full_path):
        with open(full_path, 'rb') as f:
            res = pickle.load(f)
        log(f'    [{prefix}] loaded from cache (median FDE {np.median(res["test_frac_dev"]):.3f})')
        return res
    pinfo_split, fgs = prepare_features_for_glm(pinfo)
    t0 = datetime.now()
    _, res, _, _ = fit_glm_single_session(X_aug, Y_t, metadata, fgs, pinfo_split, mouse_ID, date,
                                          model_dir=model_dir, model_prefix=prefix, contexts=CONTEXTS)
    log(f'    [{prefix}] fitted in {(datetime.now() - t0).total_seconds()/60:.1f} min '
        f'(median FDE {np.median(res["test_frac_dev"]):.3f})')
    gc.collect()
    if _HAS_TF:
        tf.keras.backend.clear_session()
    return res


def run_fold(k, X, Y, metadata, pred_info, labels, halves, rng, mouse_ID, date, model_dir, prefix_base,
             skip_existing, log):
    src_half, tgt_half = (0, 1) if k == 1 else (1, 0)
    ti_src, td_src = halves['ti'][src_half], halves['td'][src_half]
    ti_tgt, td_tgt = halves['ti'][tgt_half], halves['td'][tgt_half]
    n_src = min(len(ti_src), len(td_src))                     # size-match the two source blocks
    ti_src = np.sort(rng.choice(ti_src, n_src, replace=False))
    td_src = np.sort(rng.choice(td_src, n_src, replace=False))
    targets = np.concatenate([ti_tgt, td_tgt, labels['int']])
    target_cls = np.array(['TI'] * len(ti_tgt) + ['TD'] * len(td_tgt) + ['Int'] * len(labels['int']))
    prefix = f'{prefix_base}_fold{k}'
    log(f'  fold {k}: sources TI={len(ti_src)} TD={len(td_src)} | '
        f'targets TI={len(ti_tgt)} TD={len(td_tgt)} Int={len(labels["int"])}')
    assert not (set(targets) & set(ti_src)) and not (set(targets) & set(td_src)), 'target inside a source block'
    Y_t = Y[:, targets]
    n = len(targets)
    rows = dict(target_idx=targets, target_cls=target_cls, fold=np.full(n, k), n_src=np.full(n, n_src))
    results, n_pcs_actual = {}, {}
    for name, src in (('ti', ti_src), ('td', td_src)):
        X_aug, pinfo, n_pcs_actual[name] = build_design(X, Y, pred_info, name, src)
        res = fit_model(X_aug, Y_t, metadata, pinfo, mouse_ID, date, model_dir, f'{prefix}_{name}',
                        skip_existing, log)
        results[name] = res
        rows.update({f'fde_{name}': res['test_frac_dev'],
                     f'fde_{name}_vis': res['test_frac_dev_per_ctx'][1],
                     f'fde_{name}_aud': res['test_frac_dev_per_ctx'][2]})
        del X_aug
    m1, m2 = results['ti']['combined_test_mask'], results['td']['combined_test_mask']
    assert np.array_equal(m1, m2), 'the two models were scored on different test trials'
    same_split = None
    if labels['base_test_mask'] is not None and len(labels['base_test_mask']) == len(m1):
        same_split = bool(np.array_equal(labels['base_test_mask'], m1))
    log(f'    test trials identical to GLM_uni: {same_split}')

    design = dict(fold=k, targets=targets, target_cls=target_cls, ti_src=ti_src, td_src=td_src,
                  n_pcs_actual=n_pcs_actual, same_test_split_as_base=same_split, contexts=CONTEXTS,
                  n_pcs=N_PCS, n_lag_bases=N_LAG_BASES, lag_frames=LAG_FRAMES)
    with open(os.path.join(BASE_DATA_PATH, mouse_ID, date, model_dir, f'{prefix}_design.pkl'), 'wb') as f:
        pickle.dump(design, f)
    del Y_t
    gc.collect()
    return rows


def run_session(mouse_ID, date, split=0, null=False, skip_existing=True, log=print):
    model_dir, stem = VARIANTS[null]
    prefix_base = f'{stem}_s{split}'
    save_dir = os.path.join(BASE_DATA_PATH, mouse_ID, date, model_dir)
    session_path = os.path.join(save_dir, f'{prefix_base}_session.pkl')
    if skip_existing and os.path.exists(session_path):
        log(f'[SKIP] {mouse_ID}/{date}: session file exists')
        return 'skipped'

    labels = get_labels(mouse_ID, date)
    n_ti, n_td, n_int = len(labels['ti']), len(labels['td']), len(labels['int'])
    if n_ti < MIN_PER_CLASS or n_td < MIN_PER_CLASS:
        log(f'[EXCLUDE] {mouse_ID}/{date}: TI={n_ti} TD={n_td} (< {MIN_PER_CLASS})')
        return 'excluded'
    os.makedirs(save_dir, exist_ok=True)

    X, Y, metadata, _, pred_info, _ = open_glm_stacked_data(mouse_ID, date, print_summary=False)
    X, Y, metadata = _filter_contexts(X, Y, metadata)
    assert Y.shape[1] == len(labels['fde_base']), 'stacked Y and GLM_uni disagree on the neuron count'
    halves, seed, rng = make_split(labels, mouse_ID, date, split, null)
    log(f'[{MODEL_NAME}{" NULL" if null else ""} split {split}] {mouse_ID}/{date}: TI={n_ti} TD={n_td} Int={n_int} '
        f'| cuts TI>={labels["cuts"][0]:.3f} TD<={labels["cuts"][1]:.3f} | seed {seed} | {X.shape[0]:,} frames')

    fold_rows = [run_fold(k, X, Y, metadata, pred_info, labels, halves, rng, mouse_ID, date, model_dir,
                          prefix_base, skip_existing, log) for k in (1, 2)]
    merged = {key: np.concatenate([fr[key] for fr in fold_rows]) for key in fold_rows[0]}
    t = merged['target_idx']
    true_cls = np.full(len(labels['fde_base']), 'NA', dtype=object)
    for name, key in (('TI', 'ti'), ('TD', 'td'), ('Int', 'int')):
        true_cls[labels[key]] = name
    merged.update(true_cls=true_cls[t].astype(str), snCCGP=labels['snCCGP'][t], sn_within=labels['sn_within'][t],
                  cpi=labels['cpi'][t], fde_base=labels['fde_base'][t],
                  fde_base_vis=labels['fde_base_vis'][t], fde_base_aud=labels['fde_base_aud'][t],
                  mouse_ID=mouse_ID, date=date, split=split, null=null, split_seed=seed, halves=halves,
                  cuts=labels['cuts'], contexts=CONTEXTS)
    m_cls = merged['target_cls'] != 'Int'
    assert len(np.unique(t[m_cls])) == m_cls.sum() == n_ti + n_td, 'targets do not partition TI U TD'
    with open(session_path, 'wb') as f:
        pickle.dump(merged, f)

    for cls, own, oth in (('TI', 'fde_ti', 'fde_td'), ('TD', 'fde_td', 'fde_ti')):
        m = merged['target_cls'] == cls
        log(f'    {cls} targets (n={m.sum()}): FDE own-block − base {np.mean(merged[own][m] - merged["fde_base"][m]):+.4f}  '
            f'other-block − base {np.mean(merged[oth][m] - merged["fde_base"][m]):+.4f}  '
            f'| own > other in {np.mean(merged[own][m] > merged[oth][m])*100:.0f}% of targets')
    del X, Y, metadata
    gc.collect()
    return 'success'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', nargs='*', default=None, help='MOUSE/DATE list (space- or comma-separated)')
    ap.add_argument('--mouse', default=None)
    ap.add_argument('--split', type=int, default=0, help='index of the neuron split (independent seeds)')
    ap.add_argument('--null', action='store_true', help='label-shuffle control')
    ap.add_argument('--no-skip', action='store_true')
    args = ap.parse_args()

    sessions = [(m, d) for m, ds in PPC_AVC.items() for d in ds]
    if args.sessions:
        want = {s for a in args.sessions for s in a.split(',') if s}
        sessions = [(m, d) for m, d in sessions if f'{m}/{d}' in want]
    if args.mouse:
        sessions = [(m, d) for m, d in sessions if m == args.mouse]

    log_dir = os.path.join(BASE_DATA_PATH, 'ti_td_split_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{MODEL_NAME}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}.log')

    def log(line):
        stamp = f'[{datetime.now():%H:%M:%S}] {line}'
        print(stamp, flush=True)
        with open(log_path, 'a') as f:
            f.write(stamp + '\n')

    model_dir, _ = VARIANTS[args.null]
    log(f'{MODEL_NAME}: {len(sessions)} sessions | null={args.null} split={args.split} | results dir {model_dir} | '
        f'contexts {CONTEXTS} | n_pcs {N_PCS} | labels {LABEL_CACHE} (half={LABEL_HALF}, '
        f'TI top {PROP_TI:.0%} / TD bottom {PROP_TD:.0%}, min {MIN_PER_CLASS} per class) | '
        f'skip_existing={not args.no_skip}')
    status = {}
    for m, d in sessions:
        t0 = datetime.now()
        try:
            status[(m, d)] = run_session(m, d, split=args.split, null=args.null,
                                         skip_existing=not args.no_skip, log=log)
        except Exception as e:
            traceback.print_exc()
            status[(m, d)] = f'error: {e}'
            log(f'[ERROR] {m}/{d}: {e}')
        log(f'  -> {m}/{d}: {status[(m, d)]}  ({(datetime.now() - t0).total_seconds()/60:.1f} min)')
        gc.collect()
        if _HAS_TF:
            tf.keras.backend.clear_session()
    log('SUMMARY: ' + ', '.join(f'{k[0]}/{k[1]}={v}' for k, v in status.items()))


if __name__ == '__main__':
    main()
