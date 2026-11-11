"""
fef.io — session file reading and conversion.

Ported VERBATIM from v1 (`nhp/matio.py`, `nhp/convert.py`) in WP0; the only
edit is the import repoint to `fef.config`. Both files encode failures that were
expensive to find once (the MCOS table-pairing witnesses, the `MATLAB_empty`
cell fix), and STANDING_RULES / docs/03_PORT_AUDIT.md forbid rewriting them.
"""

from .matio import NHPMatFile, open_session          # noqa: F401
from .convert import NHPSessionConverter             # noqa: F401
