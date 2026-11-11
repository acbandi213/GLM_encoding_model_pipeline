"""
invariants.py
=============
The S5 balancing invariants, as code (WP4c). Two of v1's false results came
from two-arm comparisons whose arms were balanced differently:

  * equalize_contexts off in an ablation harness -> adding pursuit trials made
    the pooled fit pursuit-dominated and reported dFDE_pur = +0.0183 for a
    true +0.0029 — an artifact shaped exactly like the hoped-for result;
  * subpop_coupling_gain_loo defaulted to balance_key='choice' while the base
    FDE being subtracted used 'trial_type' — differencing two fits scored on
    different test sets.

`fit_glm_single_session`'s DEFAULTS are the dangerous ones (balance_key=
'choice', equalize_contexts=False). Every production fit passes these
overrides, and every comparison harness must call `assert_production_balancing`
on BOTH arms before fitting. Do not rely on remembering.
"""

from __future__ import annotations

from .. import config as C

#: The exact overrides the production fit passes (v1 nhp/train.py::fit,
#: carried to v2 unchanged). Pinned by tests/test_design.py so drift in either
#: this dict or the fit call-site fails a test instead of a science claim.
PRODUCTION_FIT_KWARGS = dict(
    contexts=[C.CTX_SACCADE, C.CTX_PURSUIT],
    balance_key='trial_type',
    balance_levels=None,
    equalize_contexts=True,
    seed=42,
    split_random_state=0,
)


class BalancingViolation(RuntimeError):
    """A two-arm comparison was about to run with mismatched balancing."""


def assert_production_balancing(fit_kwargs, where=''):
    """
    Raise unless `fit_kwargs` matches PRODUCTION_FIT_KWARGS on every
    balancing-relevant key. Call on BOTH arms of any comparison.
    """
    problems = []
    for key, expected in PRODUCTION_FIT_KWARGS.items():
        got = fit_kwargs.get(key, '<MISSING>')
        if got != expected:
            problems.append(f'{key}: expected {expected!r}, got {got!r}')
    if problems:
        raise BalancingViolation(
            f'{where or "fit"}: balancing kwargs deviate from production — '
            + '; '.join(problems)
            + '. If the deviation is deliberate, change '
              'PRODUCTION_FIT_KWARGS in fef/fit/invariants.py with a dated '
              'comment; never override silently.')


def assert_no_trial_subsetting(trial_step=None, subset=None, where=''):
    """
    S5: never run a two-arm comparison on a --trial-step / subset sample.
    Both arms hit the same balancing cap and the effect vanishes.
    """
    if trial_step not in (None, 1) or subset is not None:
        raise BalancingViolation(
            f'{where or "comparison"}: trial subsetting '
            f'(trial_step={trial_step!r}, subset={subset!r}) is forbidden in '
            f'two-arm comparisons (STANDING_RULES S5).')
