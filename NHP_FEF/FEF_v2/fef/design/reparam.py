"""
reparam.py
==========
Closed-form maps between the two direction parameterizations (WP4).

The shared/diff design emits, per trial,

    direction_shared = S            (harmonics x target/movement kernels)
    direction_diff   = S * ctx      (ctx = +1 saccade, -1 pursuit)

and the per-task design emits

    direction_sac = S * 1[saccade]
    direction_pur = S * 1[pursuit]

Since 1[sac] = (1 + ctx)/2 and 1[pur] = (1 - ctx)/2, the two designs are
related by the exact, invertible block transform

    [D_sac  D_pur] = [D_shared  D_diff] @ T,      T = 0.5 * [[I,  I],
                                                             [I, -I]]

and fitted weights convert by the transpose relationship:

    w_sac = w_shared + w_diff          w_shared = (w_sac + w_pur) / 2
    w_pur = w_shared - w_diff          w_diff   = (w_sac - w_pur) / 2

CONVENTIONS THAT MATTER DOWNSTREAM (WP7 reads these):

* The recovery above holds for weights expressed against the UN-scaled
  columns. `stack_design_matrices` z-scores X, so weights from a fit must
  first be divided by each column's sd (pass `col_sd`).
* Target-locked direction columns are in STIMULUS coordinates (step-direction
  re-signed: on backward-step pursuit trials the target appears at
  angle+180), movement-locked columns in MOVEMENT coordinates. Recovered
  per-task target-locked tuning is therefore stimulus-referenced.
* MONKEY Y (all-backward pursuit): on every pursuit trial the harmonic-1
  target-locked stimulus basis equals MINUS the movement basis, i.e. within
  the pursuit task tbasis_h1 = -dbasis_h1 exactly (cos/sin of angle+180), so
  shared_h1_targ and diff_h1_targ swap roles up to sign on the pursuit side.
  Harmonic 2 does not flip (cos 2(t+180) = cos 2t). The design stays full
  rank; only the INTERPRETATION of Y's h1 target-locked shared-vs-diff split
  changes. Verified by tests/test_design.py.
"""

from __future__ import annotations

import numpy as np


def shared_diff_to_per_task_transform(n_dir_cols):
    """The exact (2n, 2n) transform T with [D_sac D_pur] = [D_sh D_df] @ T."""
    eye = np.eye(n_dir_cols)
    return 0.5 * np.block([[eye, eye], [eye, -eye]])


def recover_per_task_direction_weights(weights, predictor_info, col_sd=None):
    """
    Per-task direction weights from a shared_diff fit.

    Parameters
    ----------
    weights : (n_features,) fitted weight vector
    predictor_info : the builder's group dict (must contain direction_shared
        and direction_diff with equal widths)
    col_sd : optional (n_features,) column sds used by the z-scoring at fit
        time; when given, weights are first converted to the un-scaled column
        space where the closed form holds.

    Returns
    -------
    dict with 'w_sac', 'w_pur', 'w_shared', 'w_diff' (each (n_dir_cols,))
    and 'names' (the direction_shared column names).
    """
    w = np.asarray(weights, float).copy()
    if col_sd is not None:
        sd = np.asarray(col_sd, float)
        safe = np.where(sd > 0, sd, 1.0)
        w = w / safe
    sh = predictor_info['direction_shared']
    df = predictor_info['direction_diff']
    n_sh = sh['end'] - sh['start']
    n_df = df['end'] - df['start']
    if n_sh != n_df:
        raise ValueError(f'direction_shared ({n_sh}) and direction_diff '
                         f'({n_df}) widths differ')
    w_shared = w[sh['start']:sh['end']]
    w_diff = w[df['start']:df['end']]
    return {
        'w_shared': w_shared, 'w_diff': w_diff,
        'w_sac': w_shared + w_diff, 'w_pur': w_shared - w_diff,
        'names': list(sh['names']),
    }


def stack_direction_columns(design_matrices, predictor_info,
                            groups=('direction_shared', 'direction_diff')):
    """Vertically stacked direction columns of a design, for the transform
    test: X_per_task == X_shared_diff @ T to numerical precision."""
    big = np.vstack(design_matrices)
    cols = []
    for g in groups:
        info = predictor_info[g]
        cols.append(big[:, info['start']:info['end']])
    return np.hstack(cols)
