"""
wp6_build_stacked.py
====================
Stage-1 driver: build the per-probe stacked caches for every session (the
FEF analog of the mouse `run_sessions.py` stacked-data build).

STRICTLY SERIAL — one HDF5/SMB reader at a time (trap 8). Measured on 0136a:
34-39 s per probe warm, so the full 48 probes are well under an hour.
Resumable (skip-existing is the default); run it once and every model script
in models/ works from the caches.

    cd FEF_v2 && python scripts/wp6_build_stacked.py               # all 26
    cd FEF_v2 && python scripts/wp6_build_stacked.py --dev         # DEV4 (S7)
    cd FEF_v2 && python scripts/wp6_build_stacked.py --sessions 0136a
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

_V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _V2 not in sys.path:
    sys.path.insert(0, _V2)

from fef import config as C                              # noqa: E402
from fef.fit import stacking                             # noqa: E402
from fef.io import NHPSessionConverter                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', nargs='*', default=None)
    ap.add_argument('--dev', action='store_true',
                    help='DEV4 sessions only (S7 subset-first)')
    ap.add_argument('--no-skip', action='store_true')
    a = ap.parse_args()

    jobs = stacking.all_probe_jobs()
    if a.dev:
        jobs = [(s, m) for s, m in jobs if s in C.DEV4_SESSIONS]
    if a.sessions:
        jobs = [(s, m) for s, m in jobs if s in set(a.sessions)]

    done = failed = skipped = 0
    t_all = time.time()
    for sess, monkey in jobs:
        path = stacking.session_mat_path(sess, monkey)
        try:
            cv = NHPSessionConverter(path, verbose=False)
        except Exception:
            print(f'!! {sess}: converter failed to open')
            traceback.print_exc()
            failed += 1
            continue
        try:
            for probe, label in enumerate(cv.probe_labels):
                out = stacking.stack_path(sess, label)
                if not a.no_skip and os.path.exists(out):
                    print(f'  [skip] {sess}_{label}')
                    skipped += 1
                    continue
                t0 = time.time()
                try:
                    stacking.write_probe_stack(sess, monkey, probe, cv=cv,
                                               verbose=False)
                    done += 1
                    print(f'  [done] {sess}_{label}  '
                          f'({time.time() - t0:.0f}s)')
                except Exception:
                    failed += 1
                    print(f'  !! {sess}_{label} FAILED')
                    traceback.print_exc()
        finally:
            cv.close()

    print(f'\nstacked: {done} built, {skipped} skipped, {failed} failed '
          f'in {(time.time() - t_all) / 60:.0f} min -> {stacking.stacked_dir()}')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
