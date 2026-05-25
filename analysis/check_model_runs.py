"""
check_model_runs.py
===================
Checks which GLM models have been run for every session in PPC_AVC.

For each (mouse_ID, date, model) combination it looks for:
  {BASE_DATA_PATH}/{mouse_ID}/{date}/{MODEL_DIR}/{MODEL_PREFIX}_full.pkl
  {BASE_DATA_PATH}/{mouse_ID}/{date}/{MODEL_DIR}/{MODEL_PREFIX}_lesion_all.pkl

Prints a summary table and a list of any missing runs.

Usage
-----
python check_model_runs.py               # check all 6 models
python check_model_runs.py --no-coupling # skip coupling variants
"""

import os
import sys
import argparse

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config.session_registry import PPC_AVC
from core.session_pipeline import BASE_DATA_PATH

# ── Model registry ─────────────────────────────────────────────────────────────
ALL_MODELS = [
    dict(name='GLM_all_context',          model_dir='GLM_all_context_results',          prefix='glm_all_ctx',     coupling=False),
    dict(name='GLM_all_context_coupling', model_dir='GLM_all_context_coupling_results', prefix='glm_all_ctx_cpl', coupling=True),
    dict(name='GLM_uni',                  model_dir='GLM_uni_results',                  prefix='glm_uni',         coupling=False),
    dict(name='GLM_uni_coupling',         model_dir='GLM_uni_coupling_results',         prefix='glm_uni_cpl',     coupling=True),
    dict(name='GLM_CG',                   model_dir='GLM_CG_results',                   prefix='glm_cg',          coupling=False),
    dict(name='GLM_CG_coupling',          model_dir='GLM_CG_coupling_results',          prefix='glm_cg_cpl',      coupling=True),
]


def check_session(mouse_ID, date, model):
    """Return (full_done, lesion_done) booleans for one session × model."""
    res_dir = os.path.join(BASE_DATA_PATH, mouse_ID, date, model['model_dir'])
    full_done   = os.path.exists(os.path.join(res_dir, f'{model["prefix"]}_full.pkl'))
    lesion_done = os.path.exists(os.path.join(res_dir, f'{model["prefix"]}_lesion_all.pkl'))
    return full_done, lesion_done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-coupling', action='store_true',
                        help='Skip coupling model variants.')
    args = parser.parse_args()

    models = [m for m in ALL_MODELS if not (args.no_coupling and m['coupling'])]

    col_w  = 24                        # width for model name column
    sep    = '─' * (20 + col_w * len(models))

    # ── Header ────────────────────────────────────────────────────────────────
    print(f'\n{"Session":<20}', end='')
    for m in models:
        print(f'{m["name"][:col_w - 1]:<{col_w}}', end='')
    print(f'\n{sep}')

    missing = []   # (mouse_ID, date, model_name, what_is_missing)

    # ── Per-session rows ───────────────────────────────────────────────────────
    for mouse_ID, dates in PPC_AVC.items():
        for date in dates:
            label = f'{mouse_ID}/{date}'
            print(f'{label:<20}', end='')
            for m in models:
                full_done, lesion_done = check_session(mouse_ID, date, m)

                if full_done and lesion_done:
                    symbol = '✓ full+lesion'
                elif full_done:
                    symbol = '~ full only '
                    missing.append((mouse_ID, date, m['name'], 'lesion_all missing'))
                else:
                    symbol = '✗ not run   '
                    missing.append((mouse_ID, date, m['name'], 'full + lesion missing'))

                print(f'{symbol:<{col_w}}', end='')
            print()

    print(sep)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_sessions = sum(len(d) for d in PPC_AVC.values())
    total_checks   = total_sessions * len(models)
    n_missing      = len(missing)
    n_done         = total_checks - n_missing

    print(f'\nTotal checks : {total_checks}  ({total_sessions} sessions × {len(models)} models)')
    print(f'Complete     : {n_done}')
    print(f'Incomplete   : {n_missing}')

    if missing:
        print(f'\n{"─" * 60}')
        print('Incomplete runs:')
        for mouse_ID, date, model_name, reason in missing:
            print(f'  {mouse_ID}/{date}  [{model_name}]  —  {reason}')

    print()


if __name__ == '__main__':
    main()