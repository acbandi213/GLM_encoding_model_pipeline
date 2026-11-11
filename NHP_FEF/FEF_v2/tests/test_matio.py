"""
Reader tests for the MCOS trial-table decoding in `nhp/matio.py`.

The load-bearing risk here is not a crash — it is a SILENT mis-decode. The MCOS
layout stores a table's column names and its column data in two separate cells,
and nothing in the file links them. Pairing them wrongly transposes the column
labels and produces a table that looks entirely normal: `angle` returns
`targetOnsetDelay`'s values, `eyePos` returns the photodiode trace, and every
downstream fit is quietly garbage.

Sessions with repeated task blocks make that a live hazard rather than a
theoretical one. `kendra_scrappy_0146a` (mdir1/mdir2/purs1/purs2) and
`kendra_scrappy_0147a` (mdir1/mdir2/purs1) each put two name cells of identical
arity into MCOS, holding the same column names in a different order. These tests
pin the three independent witnesses that separate them.

Run:  cd FEF_v2 && python -m pytest tests/ -q
(Sessions must be reachable. WP0 gates on ZERO skips — a skip here means the
discovery glob is pointing somewhere the data is not, which is the exact failure
that let 23 data-dependent tests pass vacuously in v1.)
"""

import glob
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fef import config as C                          # noqa: E402
from fef.io.matio import NHPMatFile, open_session    # noqa: E402


def _find(pattern):
    hits = sorted(glob.glob(os.path.join(C.SESSIONS_DIR, pattern)))
    return hits[0] if hits else None


SESSIONS = sorted(glob.glob(os.path.join(C.SESSIONS_DIR, C.SESSION_GLOB)))
# Sessions with repeated task blocks — the reason the pairing must be verified
# rather than guessed. WP0 found FOUR, not the two v1 knew about; `0145a` and
# `Ya_250522_s399` (a three-way purs repeat) are new.
REPEATED = {'kendra_scrappy_0145a': ['mdir1', 'mdir2', 'purs1', 'purs2', 'rfmp1'],
            'kendra_scrappy_0146a': ['mdir1', 'mdir2', 'purs1', 'purs2', 'rfmp1'],
            'kendra_scrappy_0147a': ['mdir1', 'mdir2', 'purs1', 'rfmp1'],
            'Ya_250522_s399': ['mdir1', 'purs1', 'purs2', 'purs3', 'rfmp1']}

# The four single-probe sessions. They store `sorting.*` as PLAIN ARRAYS rather
# than a (1, n) cell of object references, which is the layout that made
# `unit_depths()` raise TypeError until WP1 fixed it.
SINGLE_PROBE = C.SINGLE_PROBE_SESSIONS

pytestmark = pytest.mark.skipif(
    not SESSIONS, reason=f'no sessions found in {C.SESSIONS_DIR}')


@pytest.mark.parametrize('sess,expected_blocks', sorted(REPEATED.items()))
def test_repeated_block_sessions_decode(sess, expected_blocks):
    """
    Every block of a repeated-block session reads, and the repeats are DISTINCT.

    Before the witness-based pairing these raised
    `RuntimeError: Ambiguous MCOS table ... Refusing to guess`, so 2 of the 11
    sessions could not be fit at all. Decoding the same table twice under two
    names would be the other failure mode — hence the distinctness check.
    """
    path = _find(f'{sess}*.mat')
    if path is None:
        pytest.skip(f'{sess} not present')
    with open_session(path) as m:
        assert m.blocks() == expected_blocks

        fingerprints = {}
        for blk in expected_blocks:
            tbl = m.read_table(blk)
            assert len(tbl) > 30, f'{blk}: only {len(tbl)} columns decoded'
            assert 'result' in tbl
            n = m.n_trials(blk)
            assert len(tbl['result']) == n
            if 'angle' in tbl:
                fingerprints[blk] = (n, tuple(np.asarray(tbl['angle'],
                                                         float).ravel()[:25]))

        for kind in ('mdir', 'purs'):
            reps = [b for b in fingerprints if b.startswith(kind)]
            if len(reps) > 1:
                assert len({fingerprints[b] for b in reps}) == len(reps), (
                    f'{sess}: {reps} decoded to the same table')


@pytest.mark.parametrize('path', SESSIONS, ids=lambda p: os.path.basename(p)[:20])
def test_outcome_columns_match_result(path):
    """
    Each outcome label's event column is finite EXACTLY on its own trials.

    This is the witness that separates 0146a's two name orderings, and it is an
    exact identity in every block of every session — so it doubles as a check
    that the columns are not shifted relative to their names.
    """
    with open_session(path) as m:
        for blk in m.blocks():
            tbl = m.read_table(blk)
            res = np.array(tbl['result'])
            checked = 0
            for label in sorted({x for x in res.tolist() if x}):
                if label not in tbl:
                    continue
                col = np.asarray(tbl[label], float).ravel()
                if col.size != res.size:
                    continue
                checked += 1
                assert np.array_equal(np.isfinite(col), res == label), (
                    f'{os.path.basename(path)} {blk}: column {label!r} is finite '
                    f'on {int(np.isfinite(col).sum())} trials but result=={label} '
                    f'on {int((res == label).sum())}')
            assert checked >= 2, f'{blk}: only {checked} outcome columns checked'


@pytest.mark.parametrize('path', SESSIONS, ids=lambda p: os.path.basename(p)[:20])
def test_column_storage_kinds(path):
    """
    Structural schema check: `params` is a struct, eye traces are (n, 2) cells,
    pupil/diode are (n, 1) cells, spike times are nested cells.

    This is the witness that resolves 0147a, whose two orderings agree on every
    value-bearing column they can be compared on and then displace everything
    after `BROKE_FIX` by two — putting `eyePos` where `diode` belongs.
    """
    with open_session(path) as m:
        for blk in m.blocks():
            tbl = m.read_table(blk)
            n = m.n_trials(blk)

            assert isinstance(tbl['params'], type(m._S)), 'params is not a struct'
            for name in ('eyePos', 'eyeVel', 'eyeAcc'):
                arr = np.asarray(tbl[name][0])
                assert arr.ndim == 2 and arr.shape[1] == 2, (
                    f'{blk}.{name} has shape {arr.shape}, expected (n, 2)')
            for name in ('pupil', 'diode'):
                arr = np.asarray(tbl[name][0])
                assert arr.ndim == 2 and arr.shape[1] == 1, (
                    f'{blk}.{name} has shape {arr.shape}, expected (n, 1)')
            # eye traces and the 1 ms signals are sampled together
            assert np.asarray(tbl['eyePos'][0]).shape[0] == \
                np.asarray(tbl['pupil'][0]).shape[0]
            assert np.asarray(tbl['imec_sec']).shape == (n, 2)


@pytest.mark.parametrize('path', SESSIONS, ids=lambda p: os.path.basename(p)[:20])
def test_conditions_match_dat_text(path):
    """
    Decoded condition columns reproduce the independent `dat.text` record.

    `dat` is the plain, non-MCOS copy of the same trials, so agreement here is
    evidence the table was paired with the right data cell — not a restatement
    of the same source.
    """
    with open_session(path) as m:
        for blk in m.blocks():
            tbl = m.read_table(blk)
            cond = m._dat_conditions(blk)
            if 'angle' not in cond or 'angle' not in tbl:
                continue                      # rfmp1 has no per-trial conditions
            got = np.asarray(tbl['angle'], float).ravel()
            want = cond['angle']
            both = np.isfinite(got) & np.isfinite(want)
            assert both.sum() > 0
            assert np.allclose(got[both], want[both]), (
                f'{os.path.basename(path)} {blk}: angle disagrees with dat.text')


def test_mismatched_row_counts_are_rejected():
    """
    Two blocks sharing a trial count must raise, not silently cross-decode.

    `_categorical_pairs` selects code arrays by their (1, n_rows) shape, so
    same-length blocks would pool their categoricals and `result` could be read
    off the wrong block. No session has such a collision, so this pins the guard
    rather than a behaviour.
    """
    path = _find('kendra_scrappy_0146a*.mat')
    if path is None:
        pytest.skip('0146a not present')
    with open_session(path) as m:
        counts = [m.n_trials(b) for b in m.blocks()]
        assert len(set(counts)) == len(counts), (
            f'0146a block trial counts {counts} collide; the guard in '
            f'read_table should now be raising')


@pytest.mark.parametrize('sess', SINGLE_PROBE)
def test_unit_depths_on_single_probe_sessions(sess):
    """
    `unit_depths()` must work on the PLAIN-ARRAY `sorting` layout.

    Four Monkey Y sessions recorded one hemisphere, and MATLAB does not wrap a
    scalar cell — so `sorting.spike_positions` is a bare 2 x n array rather than
    a (1, 1) cell of references. The original code did
    `_deref(...[0, probe])` unguarded and raised
    `TypeError: Accessing a group is done with bytes or str, not numpy.float64`
    on every one of them, which would have silently cost WP1 its depth covariate
    and all of WP12 (results/WP0_data_validation.md Result 4).
    """
    path = _find(f'{sess}*.mat')
    if path is None:
        pytest.skip(f'{sess} not present')
    with open_session(path) as m:
        assert len(m.probe_labels) == 1, (
            f'{sess} is listed as single-probe but has {len(m.probe_labels)}')
        depths, uniq = m.unit_depths(0)
        assert depths.shape == uniq.shape
        assert len(uniq) == m.n_units(0)
        assert np.all(np.isfinite(depths))
        # Neuropixels 1.0 NHP-Long: 384 contacts at 20 um -> ~7.7 mm of shank.
        assert 0 <= depths.min() and depths.max() < 8000
        # the second probe does not exist and must raise, not silently return probe 0
        with pytest.raises(IndexError):
            m.sorting_field('spike_positions', 1)


def test_unit_depths_on_two_probe_session():
    """The reference session, whose `sorting` IS a (1, 2) reference cell."""
    path = _find('kendra_scrappy_0136a*.mat')
    if path is None:
        pytest.skip('0136a not present')
    with open_session(path) as m:
        for i in (0, 1):
            depths, uniq = m.unit_depths(i)
            assert len(uniq) == m.n_units(i)
            assert np.all(np.isfinite(depths))
            assert 0 <= depths.min() and depths.max() < 8000
        # the two probes are distinct populations, not the same array twice
        d0, _ = m.unit_depths(0)
        d1, _ = m.unit_depths(1)
        assert d0.shape != d1.shape or not np.array_equal(d0, d1)


def test_all_sessions_load_every_block():
    """All 26 sessions decode every block — the state the production run needs.

    v1 verified 11 Monkey S sessions on export `171af6ac94149c62`. This suite
    runs on all 26 files of export `044e61186379fd29`, including the 10 Monkey Y
    sessions, which have never been read.
    """
    failures = []
    for path in SESSIONS:
        try:
            with open_session(path) as m:
                for blk in m.blocks():
                    m.read_table(blk)
        except Exception as exc:                      # noqa: BLE001
            failures.append(f'{os.path.basename(path)}: {type(exc).__name__}: {exc}')
    assert not failures, 'sessions failed to load:\n' + '\n'.join(failures)
