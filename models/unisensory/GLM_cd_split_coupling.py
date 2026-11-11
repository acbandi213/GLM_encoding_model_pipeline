#!/usr/bin/env python
"""
GLM_cd_split_coupling.py
========================
Split-source coupling model: are CD-visual neurons coupled more to other
CD-visual neurons than to CD-auditory neurons (and vice versa)?

Design (per session, contexts 1 + 2 = visual + audio only)
-----------------------------------------------------------
Labels from the base unisensory model (GLM_uni, same regime as this fit):
    CDvis : responsive &  CPI  >= +0.8
    CDaud : responsive &  CPI  <= -0.8
    (CI   : responsive & |CPI| <  0.1 — only used with --include-ci)

SPLIT-HALF so that no target is ever inside a source PCA (no self-prediction):
    CDvis -> halves V1, V2 ;  CDaud -> halves A1, A2   (fixed per-session seed)
    fold 1 : sources V1, A1 -> targets V2 U A2
    fold 2 : sources V2, A2 -> targets V1 U A1
Every CD neuron is a target exactly once.

DEFAULT (no CI block, no reduced/lesion fits — Akhil 2026-09-14): TWO joint fits per fold
on the SAME seeded train/test trials, each via fit_glm_single_session:
    cdvis : [behaviour | CDvis 5 PCs]
    cdaud : [behaviour | CDaud 5 PCs]
Each stores held-out FDE overall AND per test context (visual / audio frames).
The behaviour-only base (GLM_uni) is scored on the identical test trials.

Legacy variant (--include-ci --reduced): [behaviour | CI | CDvis | CDaud] full fit plus
ci_cdvis / ci_cdaud / ci_only reduced refits (results dir GLM_cd_split_coupling_results).

Outputs  ({BASE_DATA_PATH}/{mouse}/{date}/{MODEL_DIR}/)
    {prefix}_fold{k}_{model}_full.pkl      per-fold fits (fit_glm_single_session output)
    {prefix}_fold{k}_design.pkl            target / source indices for the fold
    {prefix}_session.pkl                   per-target table merged over both folds
    {BASE_DATA_PATH}/cd_split_logs/GLM_cd_split_coupling_<timestamp>.log

Usage
-----
python GLM_cd_split_coupling.py                       # all PPC_AVC sessions, no-CI, skip existing
python GLM_cd_split_coupling.py --sessions KO-6-00/2026-03-12
python GLM_cd_split_coupling.py --mouse IS-2-1L --no-skip
python GLM_cd_split_coupling.py --include-ci --reduced   # legacy with-CI + reduced refits
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
import core.glm_class as glm_class
from core.glm_class import make_prediction, deviance
from config.session_registry import PPC_AVC

# ==============================================================================
MODEL_NAME = 'GLM_cd_split_coupling'
CONTEXTS   = [1, 2]                        # visual + audio only (label-model regime)

VARIANTS = {                               # include_ci -> (results dir, file prefix)
    False: ('GLM_cd_split_noci_results', 'cdsplit_noci'),
    True:  ('GLM_cd_split_coupling_results', 'cdsplit'),
}

N_PCS, N_LAG_BASES, LAG_FRAMES = 5, 4, 8

LABEL_MODEL_DIR, LABEL_MODEL_PREFIX = 'GLM_uni_results', 'GLM_uni'
RESP_THRESHOLD, INVARIANT_THRESH, CD_THRESH = 0.1, 0.1, 0.8
MIN_PER_CLASS = 4                          # need >= 2 per half; sessions below are skipped
SPLIT_SEED    = 20260914

REDUCED_MODELS = {                         # legacy (--include-ci --reduced): prefixes REMOVED from 'both'
    'ci_cdvis': ['cdaud_pc'],
    'ci_cdaud': ['cdvis_pc'],
    'ci_only':  ['cdvis_pc', 'cdaud_pc'],
}
# ==============================================================================


def _filter_contexts(X, Y, metadata):
    m = np.isin(metadata['context'], CONTEXTS)
    return X[m], Y[m], {k: v[m] for k, v in metadata.items()}


def get_labels(mouse_ID, date):
    """CI / CDvis / CDaud masks from the base GLM_uni model (contexts 1/2)."""
    p = os.path.join(BASE_DATA_PATH, mouse_ID, date, LABEL_MODEL_DIR, f'{LABEL_MODEL_PREFIX}_full.pkl')
    with open(p, 'rb') as f:
        res = pickle.load(f)
    fde = np.asarray(res['test_frac_dev'], float)
    fv = np.asarray(res['test_frac_dev_per_ctx'][1], float)
    fa = np.asarray(res['test_frac_dev_per_ctx'][2], float)
    cpi = (fv - fa) / (fv + fa + 1e-8)
    resp = fde > RESP_THRESHOLD
    return dict(ci=resp & (np.abs(cpi) < INVARIANT_THRESH),
                cdvis=resp & (cpi >= CD_THRESH),
                cdaud=resp & (cpi <= -CD_THRESH),
                cpi=cpi, fde_base=fde, fde_base_vis=fv, fde_base_aud=fa,
                base_test_mask=res.get('combined_test_mask'))


def make_split(labels, mouse_ID, date):
    """Random halves of CDvis and CDaud, fixed per-session seed."""
    seed = (SPLIT_SEED + zlib.crc32(f'{mouse_ID}/{date}'.encode())) % (2**32)
    rng = np.random.default_rng(seed)
    halves = {}
    for cls in ('cdvis', 'cdaud'):
        idx = rng.permutation(np.where(labels[cls])[0])
        halves[cls] = (np.sort(idx[: len(idx) // 2]), np.sort(idx[len(idx) // 2:]))
    return halves, seed


def build_design(X, Y, pred_info, blocks):
    """[X | block PCs ...] for blocks = [(name, source_idx), ...], lag-convolved and z-scored."""
    pinfo, offset, ranges, n_pcs_actual, cols = dict(pred_info), X.shape[1], {}, {}, []
    for name, idx in blocks:
        c, pi, _, k = _make_pc_lag_features(Y[:, idx], group_prefix=name, n_pcs=N_PCS,
                                            n_lag_bases=N_LAG_BASES, lag_frames=LAG_FRAMES,
                                            base_col_offset=offset)
        pinfo.update(pi); ranges[name] = (offset, offset + c.shape[1]); n_pcs_actual[name] = k
        offset += c.shape[1]; cols.append(c)
    return np.hstack([X] + cols), pinfo, ranges, n_pcs_actual


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


def fit_reduced(X, Y, metadata, all_results, pred_info_split, drop_prefixes, name, save_path,
                skip_existing=True):
    """Legacy: refit the 'both' model without the named blocks; same hyper-parameters, same trials."""
    if skip_existing and os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            return pickle.load(f)
    groups = list(pred_info_split.keys())
    fgs = np.asarray(all_results['feature_group_size'])
    assert len(groups) == len(fgs), 'feature_group_size / predictor_info misaligned'
    drop = [i for i, g in enumerate(groups) if any(g.startswith(p) for p in drop_prefixes)]
    col_keep = np.ones(X.shape[1], dtype=bool)
    for i in drop:
        col_keep[pred_info_split[groups[i]]['start']:pred_info_split[groups[i]]['end']] = False
    fgs_red = np.delete(fgs, drop)
    assert col_keep.sum() == fgs_red.sum(), 'column bookkeeping mismatch'
    tr, te = all_results['combined_train_mask'], all_results['combined_test_mask']
    t0 = datetime.now()
    model = glm_class.GLM_CV(
        n_folds=3, auto_split=True, split_by_group=True, skip_final_fold=True,
        activation='exp', loss_type='poisson', regularization='group_lasso',
        lambda_series=10.0 ** np.linspace(-1, -5, 5), l1_ratio=0.95, smooth_strength=0.1,
        optimizer='adam', learning_rate=1e-3, max_iter_per_lambda=5000, convergence_tol=1e-5,
    )
    model.fit(X[tr][:, col_keep], Y[tr], feature_group_size=fgs_red,
              group_idx=metadata['trial_id'][tr], verbose=False)
    model.select_model(se_fraction=0., make_fig=False)
    w, w0 = model.selected_w, model.selected_w0.flatten()
    pred = make_prediction(X[te][:, col_keep], w, w0, activation='exp')
    fd, _, _ = deviance(pred, Y[te]); fd = np.maximum(fd, 0.0)
    fd_ctx = {}
    for ctx, m in all_results['test_masks'].items():
        f_ctx, _, _ = deviance(make_prediction(X[m][:, col_keep], w, w0, activation='exp'), Y[m])
        fd_ctx[ctx] = np.maximum(f_ctx, 0.0)
    out = dict(model=name, dropped=drop_prefixes, test_frac_dev=fd, test_frac_dev_per_ctx=fd_ctx,
               cv_frac_dev=np.maximum(model.selected_frac_dev_expl_cv, 0.0), weights=w, w0=w0,
               col_keep=col_keep, groups_kept=[g for i, g in enumerate(groups) if i not in drop],
               feature_group_size=fgs_red, elapsed_s=(datetime.now() - t0).total_seconds())
    with open(save_path, 'wb') as f:
        pickle.dump(out, f)
    del model, pred
    gc.collect()
    if _HAS_TF:
        tf.keras.backend.clear_session()
    return out


def _rows_from(res, name, n_targets):
    return {f'fde_{name}': res['test_frac_dev'],
            f'fde_{name}_vis': res['test_frac_dev_per_ctx'][1],
            f'fde_{name}_aud': res['test_frac_dev_per_ctx'][2]}


def run_fold(k, X, Y, metadata, pred_info, labels, halves, mouse_ID, date, model_dir, prefix_base,
             include_ci, reduced, skip_existing, log):
    src_half, tgt_half = (0, 1) if k == 1 else (1, 0)
    vis_src, aud_src = halves['cdvis'][src_half], halves['cdaud'][src_half]
    vis_tgt, aud_tgt = halves['cdvis'][tgt_half], halves['cdaud'][tgt_half]
    ci_idx = np.where(labels['ci'])[0]
    targets = np.concatenate([vis_tgt, aud_tgt])
    target_cls = np.array(['CDvis'] * len(vis_tgt) + ['CDaud'] * len(aud_tgt))
    prefix = f'{prefix_base}_fold{k}'
    log(f'  fold {k}: sources CDvis={len(vis_src)} CDaud={len(aud_src)}'
        f'{f" CI={len(ci_idx)}" if include_ci else ""} | targets CDvis={len(vis_tgt)} CDaud={len(aud_tgt)}')
    assert not (set(targets) & set(vis_src)) and not (set(targets) & set(aud_src)), 'target inside a source block'
    if include_ci:
        assert not (set(targets) & set(ci_idx)), 'target inside the CI block'
    Y_t = Y[:, targets]
    n = len(targets)
    rows = dict(target_idx=targets, target_cls=target_cls, fold=np.full(n, k),
                n_src_cdvis=np.full(n, len(vis_src)), n_src_cdaud=np.full(n, len(aud_src)),
                n_src_ci=np.full(n, len(ci_idx) if include_ci else 0))
    results, n_pcs_actual, same_split = {}, {}, None

    if not include_ci:
        # ── DEFAULT: two independent fits, [beh | CDvis] and [beh | CDaud] ────────
        for name, src in (('cdvis', vis_src), ('cdaud', aud_src)):
            X_aug, pinfo, _, npa = build_design(X, Y, pred_info, [(name, src)])
            n_pcs_actual.update(npa)
            res = fit_model(X_aug, Y_t, metadata, pinfo, mouse_ID, date, model_dir, f'{prefix}_{name}',
                            skip_existing, log)
            results[name] = res
            rows.update(_rows_from(res, name, n))
            del X_aug
        m1, m2 = results['cdvis']['combined_test_mask'], results['cdaud']['combined_test_mask']
        assert np.array_equal(m1, m2), 'the two models were scored on different test trials'
    else:
        # ── legacy: [beh | CI | CDvis | CDaud] full fit (+ optional reduced refits) ──
        X_aug, pinfo, _, n_pcs_actual = build_design(X, Y, pred_info, [('ci', ci_idx), ('cdvis', vis_src), ('cdaud', aud_src)])
        both = fit_model(X_aug, Y_t, metadata, pinfo, mouse_ID, date, model_dir, prefix, skip_existing, log)
        results['both'] = both
        rows.update(_rows_from(both, 'both', n))
        if reduced:
            pinfo_split, _ = prepare_features_for_glm(pinfo)
            for name, drop in REDUCED_MODELS.items():
                r = fit_reduced(X_aug, Y_t, metadata, both, pinfo_split, drop, name,
                                os.path.join(BASE_DATA_PATH, mouse_ID, date, model_dir, f'{prefix}_model_{name}.pkl'),
                                skip_existing=skip_existing)
                results[name] = r
                rows.update(_rows_from(r, name, n))
        m1 = both['combined_test_mask']
        del X_aug

    if labels['base_test_mask'] is not None and len(labels['base_test_mask']) == len(m1):
        same_split = bool(np.array_equal(labels['base_test_mask'], m1))
    log(f'    test trials identical to GLM_uni: {same_split}')

    design = dict(fold=k, targets=targets, target_cls=target_cls, cdvis_src=vis_src, cdaud_src=aud_src,
                  ci_src=ci_idx if include_ci else None, include_ci=include_ci, reduced=reduced,
                  models=list(results.keys()), n_pcs_actual=n_pcs_actual,
                  same_test_split_as_base=same_split, contexts=CONTEXTS,
                  n_pcs=N_PCS, n_lag_bases=N_LAG_BASES, lag_frames=LAG_FRAMES)
    with open(os.path.join(BASE_DATA_PATH, mouse_ID, date, model_dir, f'{prefix}_design.pkl'), 'wb') as f:
        pickle.dump(design, f)
    del Y_t
    gc.collect()
    return rows


def run_session(mouse_ID, date, include_ci=False, reduced=False, skip_existing=True, log=print):
    model_dir, prefix_base = VARIANTS[include_ci]
    save_dir = os.path.join(BASE_DATA_PATH, mouse_ID, date, model_dir)
    os.makedirs(save_dir, exist_ok=True)
    session_path = os.path.join(save_dir, f'{prefix_base}_session.pkl')
    if skip_existing and os.path.exists(session_path):
        log(f'[SKIP] {mouse_ID}/{date}: session file exists')
        return 'skipped'

    labels = get_labels(mouse_ID, date)
    n_vis, n_aud = int(labels['cdvis'].sum()), int(labels['cdaud'].sum())
    if n_vis < MIN_PER_CLASS or n_aud < MIN_PER_CLASS:
        log(f'[EXCLUDE] {mouse_ID}/{date}: CDvis={n_vis} CDaud={n_aud} (< {MIN_PER_CLASS})')
        return 'excluded'

    X, Y, metadata, _, pred_info, _ = open_glm_stacked_data(mouse_ID, date, print_summary=False)
    X, Y, metadata = _filter_contexts(X, Y, metadata)
    halves, seed = make_split(labels, mouse_ID, date)
    log(f'[{MODEL_NAME}{"" if include_ci else " no-CI"}] {mouse_ID}/{date}: CDvis={n_vis} CDaud={n_aud} '
        f'CI={int(labels["ci"].sum())} | split seed {seed} | {X.shape[0]:,} frames')

    fold_rows = [run_fold(k, X, Y, metadata, pred_info, labels, halves, mouse_ID, date, model_dir,
                          prefix_base, include_ci, reduced, skip_existing, log) for k in (1, 2)]
    merged = {key: np.concatenate([fr[key] for fr in fold_rows]) for key in fold_rows[0]}
    order = np.argsort(merged['target_idx'])
    merged = {key: v[order] for key, v in merged.items()}
    t = merged['target_idx']
    merged.update(cpi=labels['cpi'][t], fde_base=labels['fde_base'][t],
                  fde_base_vis=labels['fde_base_vis'][t], fde_base_aud=labels['fde_base_aud'][t],
                  mouse_ID=mouse_ID, date=date, split_seed=seed, halves=halves,
                  n_ci=int(labels['ci'].sum()), contexts=CONTEXTS, include_ci=include_ci, reduced=reduced)
    assert len(t) == n_vis + n_aud and len(np.unique(t)) == len(t), 'targets do not partition CD population'
    with open(session_path, 'wb') as f:
        pickle.dump(merged, f)

    xk, yk = ('fde_cdvis', 'fde_cdaud') if not include_ci else (('fde_ci_cdvis', 'fde_ci_cdaud') if reduced else (None, None))
    if xk:
        for cls in ('CDvis', 'CDaud'):
            m = merged['target_cls'] == cls
            own, oth = (xk, yk) if cls == 'CDvis' else (yk, xk)
            log(f'    {cls} targets (n={m.sum()}): FDE own-block − base {np.mean(merged[own][m] - merged["fde_base"][m]):+.4f}  '
                f'other-block − base {np.mean(merged[oth][m] - merged["fde_base"][m]):+.4f}  '
                f'| own > other in {np.mean(merged[own][m] > merged[oth][m])*100:.0f}% of targets')
    del X, Y, metadata
    gc.collect()
    return 'success'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', default=None, help='comma-separated MOUSE/DATE list')
    ap.add_argument('--mouse', default=None)
    ap.add_argument('--no-skip', action='store_true')
    ap.add_argument('--include-ci', action='store_true', help='legacy: add the CI block as a source')
    ap.add_argument('--reduced', action='store_true', help='legacy (with --include-ci): fit reduced refits')
    args = ap.parse_args()

    sessions = [(m, d) for m, ds in PPC_AVC.items() for d in ds]
    if args.sessions:
        want = set(args.sessions.split(','))
        sessions = [(m, d) for m, d in sessions if f'{m}/{d}' in want]
    if args.mouse:
        sessions = [(m, d) for m, d in sessions if m == args.mouse]

    log_dir = os.path.join(BASE_DATA_PATH, 'cd_split_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{MODEL_NAME}_{datetime.now():%Y%m%d_%H%M%S}.log')

    def log(line):
        stamp = f'[{datetime.now():%H:%M:%S}] {line}'
        print(stamp, flush=True)
        with open(log_path, 'a') as f:
            f.write(stamp + '\n')

    model_dir, _ = VARIANTS[args.include_ci]
    log(f'{MODEL_NAME}: {len(sessions)} sessions | include_ci={args.include_ci} reduced={args.reduced} | '
        f'results dir {model_dir} | contexts {CONTEXTS} | n_pcs {N_PCS} | labels {LABEL_MODEL_DIR} '
        f'(resp>{RESP_THRESHOLD}, CD>={CD_THRESH}) | skip_existing={not args.no_skip}')
    status = {}
    for m, d in sessions:
        t0 = datetime.now()
        try:
            status[(m, d)] = run_session(m, d, include_ci=args.include_ci, reduced=args.reduced,
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
