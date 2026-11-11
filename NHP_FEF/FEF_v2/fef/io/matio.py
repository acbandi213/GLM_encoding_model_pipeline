"""
matio.py
========
Pure-Python reader for the macaque FEF session ``.mat`` files (MATLAB v7.3 / HDF5).

Why this exists
---------------
The per-block trial tables (``S.mdir1.tbl``, ``S.purs1.tbl``) are MATLAB
``table`` objects, stored MCOS-encoded: the ``tbl`` dataset itself is only a
6-element opaque reference and carries no data. The conversion plan therefore
assumed a one-time MATLAB export (``matlab/export_nhp_tables.m``) would be run
to flatten them.

MATLAB is not installed on this machine, so that path was unavailable. It turns
out not to be needed: the table's variable names and column data are both stored
as ordinary HDF5 cell arrays under ``/#subsystem#/MCOS``, and can be read
directly. This module locates and decodes them.

That matters for more than convenience — it means we read the *authors'* own
processed columns (``eyePos`` in degrees, ``saccadeOnset``, ``pursuitOnset``,
``TARG_ON``, …) rather than re-deriving event times from raw event codes and
uncalibrated eye traces, which removes the alignment risk the plan flagged as
the main hazard of the pure-Python fallback.

How the MCOS layout is decoded
------------------------------
``/#subsystem#/MCOS`` is a flat cell array. For each MATLAB table it contains,
among other things:

* a ``(K, 1)`` cell of ``K`` char arrays  — the variable names;
* a ``(K, 1)`` cell of ``K`` column values — the data.

Rather than hard-coding the observed indices (which would break on any session
saved with a different object graph), tables are located structurally: find the
names/data cell pair whose arity matches and whose columns have a trailing
dimension equal to the known trial count. Every lookup is validated, and a
mismatch raises rather than silently returning the wrong table.

Sessions with repeated task blocks
----------------------------------
Most sessions hold one block per task (``mdir1``, ``purs1``). Two do not:
``kendra_scrappy_0146a`` has ``mdir1/mdir2/purs1/purs2`` and
``kendra_scrappy_0147a`` has ``mdir1/mdir2/purs1``. Two tables of the same kind
put two *name* cells of identical arity into MCOS, holding the SAME column names
in a DIFFERENT order — the order in which each block's event codes first fired,
so a block where the monkey never made a ``NO_CHOICE`` error appends that column
last. Arity and required-column tests cannot separate them, and picking wrongly
would silently transpose column labels (``angle`` reading out ``targetOnsetDelay``,
say) with no error anywhere downstream.

The disambiguation is a check, not a heuristic. ``S.<block>.dat`` is the plain,
non-MCOS copy of the same trials, and ``dat.text`` records each trial's condition
string (``'distance=455;angle=90;targetOnsetDelay=300;delay=900;'``). Parsing it
gives an independent witness of the per-trial condition values, and only the
correct name ordering reproduces them. That check runs on EVERY table read, not
just ambiguous ones, so a mis-paired table is impossible rather than unlikely.

Categorical columns (``result``, ``trialName``, ``xmlName``) are stored as a
``(1, K)`` cell of category label strings immediately followed by a
``(1, n_rows)`` integer code array. Those pairs are collected in MCOS order and
assigned to the table's categorical columns in column order — an ordering that
is verified against the number of categorical columns found.

Verified against ``kendra_scrappy_0136a`` (see ``tests/test_matio.py``).
"""

from __future__ import annotations

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        'h5py is required to read the v7.3 .mat session files.') from exc


# MATLAB opaque-object marker that prefixes an MCOS reference array.
_MCOS_MAGIC = 3707764736  # 0xDD000000

# Columns each task block's trial table must contain. Used to disambiguate the
# MCOS table lookup, which row count alone cannot do (see _find_table).
#
# Keyed by block PREFIX so repeated blocks (mdir2, purs2 — present in 0146a and
# 0147a) get the same requirements as the first block of their kind.
_REQUIRED_COLUMNS = {
    'mdir': ('result', 'eyePos', 'angle', 'TARG_ON', 'saccadeOnset',
             'spiketimes_1'),
    'purs': ('result', 'eyePos', 'angle', 'PURSUIT_TARG_ON', 'pursuitOnset',
             'jump', 'spiketimes_1'),
}

# Table column -> the key it is written under in `dat.text`, for the columns
# whose units match on both sides. Used to verify that a name cell has been
# paired with the right data cell (see _dat_conditions / _find_table).
#
# `distance` is deliberately absent: the table stores it in degrees (10, 20) but
# `dat.text` in pixels (220, 455), so the two are not directly comparable.
_CONDITION_KEYS = {
    'angle': 'angle',
    'delay': 'delay',
    'targetOnsetDelay': 'targetOnsetDelay',
    'fixDuration': 'fixDuration',
    'jump': 'jump',
    'speed': 'pursuitSpeed',
}


# The storage KIND each column must have, independent of its values. A wrong
# name/data pairing shifts columns relative to names, so these disagree even
# when the condition and outcome witnesses cannot tell the orderings apart —
# which is the case for 0147a, whose two orderings are identical through
# `BROKE_FIX` and then displace everything after it by two columns (putting
# `eyePos` where `diode` belongs). See _classify_column for the kind names.
_COLUMN_KINDS = {
    'params': 'struct',
    'trialName': 'categorical', 'xmlName': 'categorical', 'result': 'categorical',
    'time_sec': 'pair_row', 'imec_sec': 'pair_row', 'imecLFP_samp': 'pair_row',
    'eyePos': 'xy_cell', 'eyeVel': 'xy_cell', 'eyeAcc': 'xy_cell',
    'eyedata': 'xy_cell',
    'pupil': 'scalar_cell', 'diode': 'scalar_cell',
    'saccades': 'nested_cell',
    'spiketimes_1': 'nested_cell', 'spiketimes_2': 'nested_cell',
}


def _required_columns(block: str) -> tuple:
    """Required columns for a block, matched on its non-numeric prefix."""
    for prefix, cols in _REQUIRED_COLUMNS.items():
        if block.startswith(prefix):
            return cols
    return ()


def _to_str(obj) -> str:
    """Decode a MATLAB char array (uint16 code points) to a Python str."""
    return ''.join(chr(int(c)) for c in np.asarray(obj).ravel())


def _is_char(dset) -> bool:
    return (isinstance(dset, h5py.Dataset)
            and dset.dtype in (np.uint16, np.uint8)
            and dset.size > 0)


def _is_mcos_ref(arr) -> bool:
    """True if `arr` is the 6-element opaque reference MATLAB writes for MCOS."""
    a = np.asarray(arr).ravel()
    return a.size == 6 and int(a[0]) == _MCOS_MAGIC


def _is_empty_cell(dset) -> bool:
    """
    True if `dset` is MATLAB's placeholder for an EMPTY array.

    MATLAB writes `[]` inside a cell as a small uint64 dataset holding the
    array's DIMENSIONS (e.g. [1, 0]) tagged with the attribute
    ``MATLAB_empty = 1`` — not as a zero-length dataset. Read naively those
    dimensions decode as data: a unit that was silent for a whole trial came
    back as two "spikes" at t = 1 ms and t = 0 ms. Measured on
    kendra_scrappy_0136a: 64 of 374 units in trial 0 alone, i.e. 128 fabricated
    spikes per trial.

    (Those particular values landed before the analysis window start — the
    earliest is 173 ms — so `_bin_spikes` discarded them and the fitted results
    were not affected. It is still wrong at the API level, and would corrupt
    counts for any session whose target onset came early enough to include
    t = 0 in the window.)
    """
    return bool(dset.attrs.get('MATLAB_empty', 0))


class NHPMatFile:
    """
    Reader for one macaque FEF session ``.mat`` file.

    Parameters
    ----------
    path : str
        Path to ``<session>_g0-<hash>.mat``.

    Attributes
    ----------
    sess_name    : str    e.g. ``kendra_scrappy_0136a``
    monkey       : str    e.g. ``scrappy``
    probe_labels : list   e.g. ``['lFEF', 'rFEF']``
    probe_depths : ndarray  mm, one per probe
    """

    def __init__(self, path):
        self.path = str(path)
        self._f = h5py.File(self.path, 'r')
        self._S = self._f['S']
        self._mcos = np.asarray(self._f['#subsystem#']['MCOS']).ravel()
        self._table_cache = {}

    # -- context manager ----------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self._f.close()

    def _deref(self, ref):
        return self._f[ref]

    # -- session-level metadata ---------------------------------------------
    @property
    def sess_name(self) -> str:
        return _to_str(self._S['sess_name'])

    @property
    def monkey(self) -> str:
        return _to_str(self._S['metadata']['monkey'])

    @property
    def probe_labels(self) -> list:
        return [_to_str(self._deref(r))
                for r in np.asarray(self._S['metadata']['probe_label']).ravel()]

    @property
    def probe_depths(self) -> np.ndarray:
        return np.asarray(self._S['metadata']['probe_depth_mm']).ravel()

    def blocks(self) -> list:
        """Task blocks present in the file (e.g. ['mdir1', 'purs1', 'rfmp1'])."""
        return [k for k in self._S.keys()
                if isinstance(self._S[k], h5py.Group) and 'dat' in self._S[k]]

    def n_trials(self, block: str) -> int:
        """Trial count for a block, read from the plain (non-MCOS) `dat` group."""
        return int(self._S[block]['dat']['result'].shape[1])

    # -- MCOS table discovery ------------------------------------------------
    def _name_cells(self):
        """Yield (mcos_index, [names]) for every cell of char arrays."""
        for i, ref in enumerate(self._mcos):
            obj = self._deref(ref)
            if not (isinstance(obj, h5py.Dataset) and obj.dtype == object):
                continue
            elems = np.asarray(obj).ravel()
            if elems.size == 0 or obj.shape != (elems.size, 1):
                continue
            try:
                items = [self._deref(r) for r in elems]
            except (KeyError, ValueError):
                continue
            if all(_is_char(it) for it in items):
                yield i, [_to_str(it) for it in items]

    def _column_arity(self, idx, n_rows):
        """
        If MCOS[idx] is a (K,1) cell whose entries all look like columns of an
        `n_rows`-row table, return K; else None.
        """
        obj = self._deref(self._mcos[idx])
        if not (isinstance(obj, h5py.Dataset) and obj.dtype == object):
            return None
        elems = np.asarray(obj).ravel()
        if elems.size == 0 or obj.shape != (elems.size, 1):
            return None
        n_match = 0
        for r in elems:
            try:
                col = self._deref(r)
            except (KeyError, ValueError):
                return None
            if isinstance(col, h5py.Group):          # nested struct (`params`)
                n_match += 1
            elif _is_mcos_ref(col):                  # categorical
                n_match += 1
            elif isinstance(col, h5py.Dataset) and col.shape[-1] == n_rows:
                n_match += 1
        # Require a clear majority of columns to have the table's row count;
        # categoricals / structs legitimately do not.
        return elems.size if n_match >= max(2, int(0.6 * elems.size)) else None

    # -- independent witness of the trial order (the plain `dat` group) --------
    def _dat_conditions(self, block: str) -> dict:
        """
        Per-trial condition values parsed from ``S.<block>.dat.text``.

        `dat` is the plain, non-MCOS copy of the same trials, so it is an
        INDEPENDENT record of what each trial's conditions were — which is
        exactly what is needed to check that an MCOS name cell has been paired
        with the right data cell. Each entry looks like

            'distance=455;angle=90;targetOnsetDelay=300;delay=900;'   (saccade)
            'pursuitSpeed=20;angle=180;jump=-1;fixDuration=730;'      (pursuit)

        Keys that repeat within a trial are dropped: `rfmp1` lists one
        ``xpos=…;ypos=…`` pair per flashed dot, which is not a per-trial scalar.

        Returns {key: (n_rows,) float array}, NaN where a trial did not carry
        that key. Empty if the block has no `dat.text`.
        """
        grp = self._S.get(block)
        if grp is None or 'dat' not in grp or 'text' not in grp['dat']:
            return {}
        refs = np.asarray(grp['dat']['text']).ravel()
        per_trial = []
        for r in refs:
            try:
                s = _to_str(self._deref(r))
            except (KeyError, ValueError, TypeError):
                per_trial.append({})
                continue
            d, dup = {}, set()
            for field in s.split(';'):
                key, sep, val = field.partition('=')
                if not sep:
                    continue
                key = key.strip()
                if key in d:
                    dup.add(key)
                try:
                    d[key] = float(val)
                except ValueError:
                    dup.add(key)
            for key in dup:
                d.pop(key, None)
            per_trial.append(d)

        keys = set().union(*per_trial) if per_trial else set()
        return {k: np.array([t.get(k, np.nan) for t in per_trial], float)
                for k in keys}

    def _condition_agreement(self, names, col_refs, cond):
        """
        How well a candidate (names, data) pairing reproduces `dat.text`.

        Returns (n_matching, n_comparable). A correct pairing matches EXACTLY;
        a wrong one reads a different column under each name and matches almost
        nothing, so the two are separated by a wide margin rather than a
        threshold.

        NB this witness alone does NOT separate 0146a/0147a's repeated blocks:
        their two name orderings are identical through the condition columns and
        first diverge at the event-time columns (index 15 of 45). That is what
        `_outcome_agreement` covers.
        """
        n_ok = n_tot = 0
        for name, ref in zip(names, col_refs):
            key = _CONDITION_KEYS.get(name)
            if key is None or key not in cond:
                continue
            try:
                obj = self._deref(ref)
            except (KeyError, ValueError):
                continue
            if not isinstance(obj, h5py.Dataset) or obj.dtype == object:
                continue
            vals = np.asarray(obj, float).ravel()
            want = cond[key]
            if vals.size != want.size:
                continue
            n_tot += 1
            both = np.isfinite(vals) & np.isfinite(want)
            if both.any() and np.allclose(vals[both], want[both]):
                n_ok += 1
        return n_ok, n_tot

    def _classify_column(self, obj, n_rows):
        """
        The storage kind of one decoded column, or None if unrecognised.

        'struct'      nested MATLAB struct (only `params`)
        'categorical' MCOS reference (label/code pair decoded elsewhere)
        'pair_row'    (2, n_rows) numeric — a per-trial [start, end] pair
        'scalar_row'  (1, n_rows) numeric — a per-trial scalar
        'xy_cell'     per-trial cell of (n, 2) — a two-channel signal
        'scalar_cell' per-trial cell of (n, 1) — a one-channel signal
        'nested_cell' per-trial cell OF cells — saccades, spiketimes_*
        """
        if isinstance(obj, h5py.Group):
            return 'struct'
        if _is_mcos_ref(obj):
            return 'categorical'
        if not isinstance(obj, h5py.Dataset):
            return None
        if obj.dtype != object:
            if obj.shape == (2, n_rows):
                return 'pair_row'
            if obj.shape == (1, n_rows):
                return 'scalar_row'
            return None

        elems = np.asarray(obj).ravel()
        if elems.size != n_rows:
            return None
        # Skip MATLAB's empty placeholders when sampling: a trial can legitimately
        # hold `[]` (a silent unit, a trial with no detected saccade), whose
        # stored payload is the array's dimensions rather than data.
        first = None
        for r in elems[:8]:
            try:
                cand = self._deref(r)
            except (KeyError, ValueError):
                continue
            if isinstance(cand, h5py.Dataset) and _is_empty_cell(cand):
                continue
            first = cand
            break
        if first is None:
            return None
        if isinstance(first, h5py.Dataset) and first.dtype == object:
            return 'nested_cell'
        shape = getattr(first, 'shape', ())
        if len(shape) == 2 and shape[1] == 2:
            return 'xy_cell'
        if len(shape) == 2 and shape[1] == 1:
            return 'scalar_cell'
        return 'cell'

    def _structure_agreement(self, names, col_refs, n_rows):
        """
        Check each column's storage kind against the schema (`_COLUMN_KINDS`).

        Purely structural, so it works on blocks where the value-based witnesses
        have nothing to compare — including 0147a's `mdir2`, which has only
        CORRECT and BROKE_FIX outcomes and so gives `_outcome_agreement` no
        column that the two candidate orderings disagree about.

        Returns (n_matching, n_comparable).
        """
        n_ok = n_tot = 0
        for name, ref in zip(names, col_refs):
            want = _COLUMN_KINDS.get(name)
            if want is None:
                continue
            try:
                obj = self._deref(ref)
            except (KeyError, ValueError):
                continue
            n_tot += 1
            if self._classify_column(obj, n_rows) == want:
                n_ok += 1
        return n_ok, n_tot

    def _outcome_agreement(self, names, col_refs, n_rows):
        """
        Check the outcome-event columns against the `result` categorical.

        Every outcome label is also an event-time column, and the two agree
        EXACTLY: the column named ``CORRECT`` is finite on precisely the trials
        whose ``result`` is ``'CORRECT'``, ``BROKE_FIX`` on precisely the
        broken-fixation trials, and so on. Measured on every block of every
        session, with no exceptions.

        This is the witness that separates the repeated blocks in 0146a/0147a,
        because the outcome columns are exactly where their two name orderings
        differ. `result` itself is safe to compare against: it is the third
        categorical at the same index under both orderings, so it decodes
        identically either way and is genuinely independent evidence.

        Returns (n_matching, n_comparable).
        """
        cat_pairs = self._categorical_pairs(n_rows)

        res = None
        cat_i = 0
        for name, ref in zip(names, col_refs):
            try:
                obj = self._deref(ref)
            except (KeyError, ValueError):
                continue
            if isinstance(obj, h5py.Group) or not _is_mcos_ref(obj):
                continue
            if name == 'result' and cat_i < len(cat_pairs):
                labels, codes = cat_pairs[cat_i]
                res = np.array([labels[c - 1] if 1 <= c <= len(labels) else ''
                                for c in codes])
            cat_i += 1
        if res is None or res.size != n_rows:
            return 0, 0

        by_name = dict(zip(names, col_refs))
        n_ok = n_tot = 0
        for label in sorted({x for x in res.tolist() if x}):
            ref = by_name.get(label)
            if ref is None:
                continue
            try:
                obj = self._deref(ref)
            except (KeyError, ValueError):
                continue
            if not isinstance(obj, h5py.Dataset) or obj.dtype == object:
                continue
            vals = np.asarray(obj, float).ravel()
            if vals.size != n_rows:
                continue
            n_tot += 1
            if bool((np.isfinite(vals) == (res == label)).all()):
                n_ok += 1
        return n_ok, n_tot

    def _find_table(self, n_rows, required=(), block=None):
        """
        Locate the (names, data) MCOS cell pair for a table with `n_rows` rows.

        `required` names columns the table must contain. Row count alone does
        NOT identify a table: the spike-sorting per-unit tables (374 and 430
        rows here) sit in the same MCOS array as the per-trial tables (432,
        1467, 314 rows), and mdir1's 432 trials clears the 430-unit table by a
        margin of two. On a session where those counts collided, the index-gap
        tie-break would silently return the unit table and the caller would die
        much later with a bare KeyError('result'). Requiring the block's own
        columns makes that impossible.

        `required` still cannot separate two blocks of the SAME kind (0146a's
        mdir1/mdir2, purs1/purs2), whose name cells hold identical names in a
        different order. Those are separated by checking each candidate against
        the block's own `dat.text` — see `_dat_conditions`. The check also runs
        when there is only one candidate, so a wrong pairing raises here rather
        than propagating as mislabelled columns.

        Returns (names, column_refs).
        """
        candidates = []
        for name_idx, names in self._name_cells():
            k = len(names)
            for data_idx in range(len(self._mcos)):
                if data_idx == name_idx:
                    continue
                if self._column_arity(data_idx, n_rows) == k:
                    candidates.append((name_idx, data_idx, names))

        if required:
            req = set(required)
            candidates = [c for c in candidates if req <= set(c[2])]
            if not candidates:
                raise RuntimeError(
                    f'No MCOS table with {n_rows} rows carries the required '
                    f'columns {sorted(req)} in {self.path}.')

        if not candidates:
            raise RuntimeError(
                f'Could not locate an MCOS table with {n_rows} rows in {self.path}. '
                f'The file layout differs from the verified '
                f'kendra_scrappy_0136a schema.')

        cond = self._dat_conditions(block) if block is not None else {}

        scored = []
        for name_idx, data_idx, names in candidates:
            col_refs = np.asarray(self._deref(self._mcos[data_idx])).ravel()
            ok_c, tot_c = self._condition_agreement(names, col_refs, cond)
            ok_o, tot_o = self._outcome_agreement(names, col_refs, n_rows)
            ok_s, tot_s = self._structure_agreement(names, col_refs, n_rows)
            scored.append((name_idx, data_idx, names, col_refs,
                           ok_c + ok_o + ok_s, tot_c + tot_o + tot_s))

        checkable = [s for s in scored if s[5] > 0]
        if checkable:
            good = [s for s in checkable if s[4] == s[5]]
            if not good:
                best = max(checkable, key=lambda s: s[4])
                raise RuntimeError(
                    f'No MCOS table for {n_rows} rows in {self.path} is '
                    f'consistent with {block}.dat.text and the `result` column '
                    f'(best candidate matched {best[4]}/{best[5]} checks). The '
                    f'file layout differs from the verified kendra_scrappy_0136a '
                    f'schema.')
            # Distinct orderings that BOTH pass every check would mean the
            # witnesses cannot separate them; guessing between them is exactly
            # what this method exists to avoid.
            if len({tuple(s[2]) for s in good}) > 1:
                raise RuntimeError(
                    f'Ambiguous MCOS table for {n_rows} rows in {self.path}: '
                    f'candidates at {[(s[0], s[1]) for s in good]} are both '
                    f'consistent with {block}.dat.text and `result`. '
                    f'Refusing to guess.')
            scored = good
        elif len({tuple(s[2]) for s in scored}) > 1:
            # Nothing to check against AND the candidates disagree — same
            # refusal as before.
            raise RuntimeError(
                f'Ambiguous MCOS table for {n_rows} rows in {self.path}: '
                f'candidates at {[(s[0], s[1]) for s in scored]} have '
                f'different column sets, and block {block} carries nothing to '
                f'separate them. Refusing to guess.')

        scored.sort(key=lambda s: abs(s[1] - s[0]))
        return scored[0][2], scored[0][3]

    def _categorical_pairs(self, n_rows):
        """
        Collect (labels, codes) pairs for categoricals of an `n_rows`-row table,
        in MCOS order (== table column order).
        """
        pairs = []
        for i in range(len(self._mcos) - 1):
            codes_obj = self._deref(self._mcos[i + 1])
            if not (isinstance(codes_obj, h5py.Dataset)
                    and codes_obj.dtype in (np.uint8, np.uint16, np.uint32)
                    and codes_obj.shape == (1, n_rows)):
                continue
            lab_obj = self._deref(self._mcos[i])
            if not (isinstance(lab_obj, h5py.Dataset) and lab_obj.dtype == object):
                continue
            elems = np.asarray(lab_obj).ravel()
            if lab_obj.shape != (1, elems.size):
                continue
            try:
                labels = [_to_str(self._deref(r)) for r in elems]
            except (KeyError, ValueError, TypeError):
                continue
            pairs.append((labels, np.asarray(codes_obj).ravel()))
        return pairs

    # -- public table API ----------------------------------------------------
    def read_table(self, block: str) -> dict:
        """
        Read one block's trial table as a dict of columns.

        Returns
        -------
        dict  column name -> value, where value is one of:
            * ndarray (n_trials,)        — per-trial scalar
            * ndarray (n_trials, 2)      — per-trial pair (e.g. time_sec)
            * list of ndarray            — per-trial cell (eyePos, spiketimes_*)
            * list of str                — categorical, decoded to label strings
        """
        if block in self._table_cache:
            return self._table_cache[block]

        n_rows = self.n_trials(block)

        # `_categorical_pairs` selects code arrays by their (1, n_rows) shape,
        # so two blocks with the SAME trial count would pool their categoricals
        # and the label/code assignment below would silently cross over. Blocks
        # repeat only in 0146a/0147a, where every count differs (365/185/710/
        # 725/154 and 711/50/797/310) — but say so rather than assume it.
        same_n = [b for b in self.blocks()
                  if b != block and self.n_trials(b) == n_rows]
        if same_n:
            raise RuntimeError(
                f'{block} has {n_rows} trials, the same as {same_n} in '
                f'{self.path}. Categorical columns cannot be attributed to a '
                f'block by row count alone. Refusing to guess.')

        names, col_refs = self._find_table(n_rows, _required_columns(block),
                                           block=block)
        cat_pairs = self._categorical_pairs(n_rows)

        out = {}
        cat_i = 0
        for name, ref in zip(names, col_refs):
            obj = self._deref(ref)

            if isinstance(obj, h5py.Group):
                out[name] = obj                       # nested struct, kept raw
                continue

            if _is_mcos_ref(obj):
                if cat_i >= len(cat_pairs):
                    raise RuntimeError(
                        f'{block}.{name} is categorical but no matching '
                        f'label/code pair remained (found {len(cat_pairs)}).')
                labels, codes = cat_pairs[cat_i]
                cat_i += 1
                out[name] = [labels[c - 1] if 1 <= c <= len(labels) else None
                             for c in codes]
                continue

            if obj.dtype == object:
                out[name] = [np.asarray(self._deref(r))
                             for r in np.asarray(obj).ravel()]
                continue

            arr = np.asarray(obj)
            # MATLAB writes (n_rows,) columns as (1, n) and pairs as (2, n).
            out[name] = arr.ravel() if arr.shape[0] == 1 else arr.T

        self._table_cache[block] = out
        return out

    # -- spike sorting -------------------------------------------------------
    def n_units(self, probe: int) -> int:
        """
        Unit count for a probe.

        Note `sorting.spike_times[probe]` is the *flat* spike-time vector for the
        whole probe, not a per-unit cell — the unit count comes from the
        per-block spike cell instead, which is (1, n_units).
        """
        block = self.blocks()[0]
        cell = self._deref(self._S[block]['spike_times'][0, probe])
        return int(np.asarray(cell).size)

    def sorting_field(self, field: str, probe: int) -> np.ndarray:
        """
        A `sorting.<field>` array for one probe, under EITHER storage layout.

        Two-probe sessions store these as a (1, 2) cell of object references, one
        per probe. The four SINGLE-probe Monkey Y sessions (`Ya_250429_s385`,
        `Ya_250430_s386`, `Ya_250506_s390`, `Ya_250514_s394`) store the array
        PLAINLY — no cell, no reference — because MATLAB does not wrap a scalar
        cell. Dereferencing the plain layout raises
        ``TypeError: Accessing a group is done with bytes or str, not
        numpy.float64``, which is what this exists to prevent.

        Applies to every `sorting` field, `templates` and `spike_positions`
        included. Found in WP0; see results/WP0_data_validation.md Result 4.
        """
        d = self._S['sorting'][field]
        if d.dtype == object:
            return np.asarray(self._deref(d[0, probe]))
        if probe != 0:
            raise IndexError(
                f'sorting.{field} is stored single-probe but probe={probe} '
                f'was requested ({self.sess_name} has {len(self.probe_labels)} '
                f'probe(s))')
        return np.asarray(d)

    def unit_depths(self, probe: int):
        """
        Per-unit depth (um) = median of that unit's spike depths.

        `sorting.spike_positions` is a 2 x n_spikes array (row 1 = depth) and
        `sorting.spike_clusters` gives the cluster id per spike — ordinary HDF5
        datasets, so no MCOS decoding is needed. Both are read through
        `sorting_field` because their per-probe wrapping differs between
        two-probe and single-probe sessions.

        Returns (depths, cluster_ids), both length n_units.
        """
        pos = self.sorting_field('spike_positions', probe)
        clu = self.sorting_field('spike_clusters', probe).ravel()
        depth = pos[:, 1] if pos.shape[0] == clu.size else pos[1, :]
        uniq = np.unique(clu)
        # bincount-style grouping: sorting once beats an O(n_units x n_spikes)
        # boolean scan, which on a 20 M-spike probe is minutes rather than ms.
        order = np.argsort(clu, kind='stable')
        clu_s, depth_s = clu[order], depth[order]
        lo = np.searchsorted(clu_s, uniq, side='left')
        hi = np.searchsorted(clu_s, uniq, side='right')
        return (np.array([np.median(depth_s[a:b]) for a, b in zip(lo, hi)]),
                uniq)

    def trial_spiketimes(self, block: str, probe: int, trial: int) -> list:
        """
        Per-unit spike times (ms, relative to START_TRIAL) for one trial/probe.

        Returns a list of length n_units; entry i is a 1-D array for unit i.
        """
        tbl = self.read_table(block)
        cell = tbl[f'spiketimes_{probe + 1}'][trial]      # (1, n_units) of refs
        refs = np.asarray(cell).ravel()
        out = []
        for r in refs:
            try:
                d = self._deref(r)
                # A silent unit is stored as MATLAB's empty placeholder, whose
                # payload is the array's dimensions, not spike times.
                v = np.empty(0) if _is_empty_cell(d) else np.asarray(d).ravel()
            except (KeyError, ValueError):
                v = np.empty(0)
            out.append(v.astype(float))
        return out

    def trial_saccades(self, block: str, trial: int) -> np.ndarray:
        """
        Detected saccades for one trial as (n_saccades, 2) — [onset, offset] ms
        relative to START_TRIAL.

        The `saccades` column is a variable-length cell of 2-element arrays, one
        per detected saccade. On pursuit trials these are the CATCH-UP saccades
        made while tracking the ramp: a mean of 4.7 per trial inside the analysis
        window on kendra_scrappy_0136a, versus 2.8 on the saccade task. They are
        large events in a saccade-related area and, apart from the one primary
        movement, nothing in the design matrix accounts for them.

        Returns an empty (0, 2) array when the trial has none.
        """
        tbl = self.read_table(block)
        cell = tbl['saccades'][trial]
        refs = np.asarray(cell).ravel()
        out = []
        for r in refs:
            try:
                d = self._deref(r)
                if _is_empty_cell(d):
                    continue
                a = np.asarray(d, float).ravel()
            except (KeyError, ValueError, TypeError):
                continue
            if a.size >= 2:
                out.append(a[:2])
        return np.asarray(out, float) if out else np.empty((0, 2))


def open_session(path) -> NHPMatFile:
    """Convenience opener."""
    return NHPMatFile(path)
