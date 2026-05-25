"""
session_registry_multisensory.py
=================================
Session registry for multisensory cross-task analysis.

This registry defines three datasets:
1. PPC_AVC_PRIMING: Original AVC priming task sessions
2. PPC_VACANT: Multisensory conflict task sessions (VACANT)
3. CROSS_TASK_PAIRS: Matched pairs for transfer analysis

Task mapping information:
------------------------
PPC_AVC_PRIMING sessions use 'avc' context mapping:
  - trial_type 1,2: context 0 (congruent)
  - trial_type 3,4: context 1 (visual)
  - trial_type 5,6: context 2 (audio)

PPC_VACANT sessions use '2loc' context mapping:
  - trial_type 1,2: context 0 (congruent audio-visual L/R)
  - trial_type 3,4: context 1 (visual relevant L/R)
  - trial_type 5,6: context 2 (audio relevant L/R)

The _1 suffix in dates indicates the multisensory/VACANT session that
was collected on the same day as the corresponding priming session.
"""

# ── Priming sessions (AVC task) ───────────────────────────────────────────────
# Keys   : mouse_ID strings
# Values : list of date strings (YYYY-MM-DD)
PPC_AVC_PRIMING = {
    'IS-2-1L': ['2025-04-14', '2025-04-15', '2025-04-22', '2025-12-12'],
    'KO-5-1L': ['2026-04-01', '2026-04-02', '2026-04-08', '2026-04-28', 
                '2026-04-29', '2026-04-30']
    #'KO-6-00': ['2026-04-02', '2026-04-07']
}

# ── Multisensory sessions (VACANT conflict task) ──────────────────────────────
# Keys   : mouse_ID strings
# Values : list of date strings with _1 suffix (YYYY-MM-DD_1)
PPC_VACANT = {
    'IS-2-1L': ['2025-04-14_1', '2025-04-15_1', '2025-04-22_1', '2025-12-12_1'],
    'KO-5-1L': ['2026-04-01_1', '2026-04-02_1', '2026-04-08_1', '2026-04-28_1', 
                '2026-04-29_1', '2026-04-30_1']
    #'KO-6-00': ['2026-04-02_1', '2026-04-07_1']
}

# ── Cross-task paired sessions ────────────────────────────────────────────────
# Keys   : mouse_ID strings
# Values : list of (priming_date, multisensory_date) tuples
CROSS_TASK_PAIRS = {
    'IS-2-1L': [
        ('2025-04-14', '2025-04-14_1'),
        ('2025-04-15', '2025-04-15_1'),
        ('2025-04-22', '2025-04-22_1'),
        ('2025-12-12', '2025-12-12_1'),
    ],
    'KO-5-1L': [
        ('2026-04-01', '2026-04-01_1'),
        ('2026-04-02', '2026-04-02_1'),
        ('2026-04-08', '2026-04-08_1'),
        ('2026-04-28', '2026-04-28_1'),
        ('2026-04-29', '2026-04-29_1'),
        ('2026-04-30', '2026-04-30_1'),
    ]
    # 'KO-6-00': [
    #     ('2026-04-02', '2026-04-02_1'),
    #     ('2026-04-07', '2026-04-07_1'),
    # ],
}

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

# ── Helper functions ──────────────────────────────────────────────────────────

def get_all_sessions():
    """Return combined dictionary of all priming and multisensory sessions."""
    combined = {}
    for mouse_ID in set(list(PPC_AVC_PRIMING.keys()) + list(PPC_VACANT.keys())):
        priming = PPC_AVC_PRIMING.get(mouse_ID, [])
        vacant = PPC_VACANT.get(mouse_ID, [])
        combined[mouse_ID] = priming + vacant
    return combined

def count_sessions():
    """Return counts for each session type."""
    n_priming = sum(len(dates) for dates in PPC_AVC_PRIMING.values())
    n_vacant = sum(len(dates) for dates in PPC_VACANT.values())
    n_pairs = sum(len(pairs) for pairs in CROSS_TASK_PAIRS.values())
    return {
        'priming': n_priming,
        'vacant': n_vacant,
        'pairs': n_pairs,
        'mice': len(PPC_AVC_PRIMING),
    }
