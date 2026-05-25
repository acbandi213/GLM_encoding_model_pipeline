#!/usr/bin/env python
"""
_run_single_session.py
======================
Internal helper used by run_all_models.py (session-first mode).

Runs ONE model for ONE (mouse_ID, date) pair in a fresh subprocess.
When this process exits, the OS reclaims all memory: numpy arrays,
TF/Keras session state, everything.  This is the mechanism that prevents
memory accumulation across sessions.

Not intended to be called directly — use run_all_models.py instead.

Internal usage
--------------
    python _run_single_session.py <module_name> <mouse_ID> <date> \\
           [--no-skip] [--metric deconv|dff|z_dff]
"""

import argparse
import importlib
import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
# Script is in runners/, need to go up 1 level to GLM_encoding_model root
_RUNNER_DIR    = os.path.dirname(os.path.abspath(__file__))
GLM_DIR        = os.path.dirname(_RUNNER_DIR)
ICLOUD_GLM_DIR = ('/Users/akhilbandi/Library/Mobile Documents/'
                  'com~apple~CloudDocs/Documents/Portfolio/code_projects/'
                  'Context Paper Code/GLM_encoding_model')
for _p in (GLM_DIR, ICLOUD_GLM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

parser = argparse.ArgumentParser(
    description='Run one GLM model for one session (internal helper).')
parser.add_argument('module_name',
                    help='Python module name, e.g. GLM_uni')
parser.add_argument('mouse_ID',
                    help='Mouse identifier, e.g. KO-5-1L')
parser.add_argument('date',
                    help='Session date, e.g. 2026-03-24')
parser.add_argument('--no-skip', action='store_true',
                    help='Force re-run (ignore existing results)')
parser.add_argument('--metric', type=str, default='deconv',
                    help='Neural metric: deconv | dff | z_dff')
args = parser.parse_args()

module = importlib.import_module(args.module_name)

module.run_all_sessions(
    {args.mouse_ID: [args.date]},
    skip_existing=not args.no_skip,
    neural_metric=args.metric,
)
