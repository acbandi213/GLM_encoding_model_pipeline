"""
session_registry.py
===================
Single source of truth for the session list and shared lesion-group names.

All GLM runner scripts (GLM_all_context.py, GLM_uni.py, GLM_CG.py,
GLM_uni_coupling.py, GLM_CG_coupling.py) import PPC_AVC and
BASE_LESION_GROUPS from here so that adding a new session only requires
editing this one file.
"""

# ── Session registry ──────────────────────────────────────────────────────────
# Keys   : mouse_ID strings
# Values : list of date strings (YYYY-MM-DD)

# PPC_AVC = {
#     'IS-2-1L': ['2025-03-10', '2025-03-11', '2025-03-12', '2025-03-13',
#                 '2025-03-20', '2025-03-26', '2025-04-02', '2025-04-04',
#                 '2025-04-08', '2025-04-09', '2025-05-05',
#                 '2025-05-28', '2025-05-30', '2025-11-20'],
#     'IS-3-1R': ['2024-12-23', '2024-12-24'],
#     'JP-2-1L': ['2025-08-19', '2025-08-26', '2025-09-16', '2025-09-17',
#                 '2025-09-19', '2025-09-22'],
#     'KA-1-00': ['2025-12-16', '2026-01-08'],
#     'KA-2-00': ['2026-01-08', '2026-01-13', '2026-01-14'],
#     'KO-5-1L': ['2026-03-05', '2026-03-06', '2026-03-12', '2026-03-13', 
#                 '2026-03-24', '2026-03-30', '2026-04-06', '2026-04-27'],
#     'KO-6-00': ['2026-03-12', '2026-03-23', '2026-03-25', '2026-03-30', '2026-04-06'],
# }

# ── Predictor groups that are always lesioned ─────────────────────────────────
# Coupling groups (network_pc1 … network_pcN) are appended by each runner
# that uses coupling.
BASE_LESION_GROUPS = [
    'position_only',
    'audio_context',
    'audio_stim',
    'visual_context',
    'visual_stim',
    'velocity',
    'choice_spatial',
    'turn_onset',
    'outcome',
]
