#!/usr/bin/env python
"""
check_model_status.py
=====================
Check which GLM models have been run for each session in the registries.

Usage:
    python config/check_model_status.py
    python config/check_model_status.py --dataset avc
    python config/check_model_status.py --dataset all --verbose
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add parent directory to path
_SCRIPT_DIR = Path(__file__).parent.parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config.session_registry import PPC_AVC
from config.session_registry_multisensory import (
    PPC_AVC_PRIMING,
    PPC_VACANT,
)

BASE_DATA_PATH = '/Volumes/Akhil Data/Akhil/ProcessedData'

# Model file patterns and their result directories
# Each model saves to its own directory: {MODEL_NAME}_results/
MODEL_FILE_PATTERNS = {
    'avc': {
        'GLM_all_context': {
            'dir': 'GLM_all_context_results',
            'files': ['glm_all_ctx_full.pkl', 'glm_all_ctx_lesion_all.pkl']
        },
        'GLM_all_context_coupling': {
            'dir': 'GLM_all_context_coupling_results',
            'files': ['glm_all_ctx_cpl_full.pkl', 'glm_all_ctx_cpl_lesion_all.pkl']
        },
        'GLM_uni': {
            'dir': 'GLM_uni_results',
            'files': ['glm_uni_full.pkl', 'glm_uni_lesion_all.pkl']
        },
        'GLM_uni_coupling': {
            'dir': 'GLM_uni_coupling_results',
            'files': ['glm_uni_cpl_full.pkl', 'glm_uni_cpl_lesion_all.pkl']
        },
        'GLM_CG': {
            'dir': 'GLM_CG_results',
            'files': ['glm_cg_full.pkl', 'glm_cg_lesion_all.pkl']
        },
        'GLM_CG_coupling': {
            'dir': 'GLM_CG_coupling_results',
            'files': ['glm_cg_cpl_full.pkl', 'glm_cg_cpl_lesion_all.pkl']
        },
    },
    'avc_priming': {
        'GLM_priming_uni': {
            'dir': 'GLM_priming_uni_results',
            'files': ['glm_priming_uni_full.pkl', 'glm_priming_uni_lesion_all.pkl']
        },
        'GLM_priming_uni_coupling': {
            'dir': 'GLM_priming_uni_coupling_results',
            'files': ['glm_priming_uni_cpl_full.pkl', 'glm_priming_uni_cpl_lesion_all.pkl']
        },
    },
    'vacant': {
        'GLM_vacant_all': {
            'dir': 'GLM_vacant_all_results',
            'files': ['glm_vacant_all_full.pkl', 'glm_vacant_all_lesion_all.pkl']
        },
        'GLM_vacant_all_coupling': {
            'dir': 'GLM_vacant_all_coupling_results',
            'files': ['glm_vacant_all_cpl_full.pkl', 'glm_vacant_all_cpl_lesion_all.pkl']
        },
        'GLM_vacant_conflict_only': {
            'dir': 'GLM_vacant_conflict_only_results',
            'files': ['glm_vacant_conflict_only_full.pkl', 'glm_vacant_conflict_only_lesion_all.pkl']
        },
        'GLM_vacant_conflict_only_coupling': {
            'dir': 'GLM_vacant_conflict_only_coupling_results',
            'files': ['glm_vacant_conflict_only_cpl_full.pkl', 'glm_vacant_conflict_only_cpl_lesion_all.pkl']
        },
    },
    'cross_task': {
        'GLM_cross_task': {
            'dir': 'GLM_cross_task_results',
            'files': ['glm_cross_task_full.pkl', 'glm_cross_task_lesion_all.pkl']
        },
        'GLM_cross_task_coupling': {
            'dir': 'GLM_cross_task_coupling_results',
            'files': ['glm_cross_task_coupling_full.pkl', 'glm_cross_task_coupling_lesion_all.pkl']
        },
    },
}


def check_session_status(mouse_id, date, dataset_type='avc', verbose=False):
    """
    Check completion status for a single session.
    
    Args:
        mouse_id: Mouse identifier
        date: Session date
        dataset_type: Type of dataset ('avc', 'avc_priming', 'vacant')
        verbose: Print detailed model files
    
    Returns dict with:
        - has_dict: bool (glm_data_dict.pkl.gz exists)
        - has_stacked: bool (glm_data_stacked.pkl exists)
        - completed_models: dict of {model_name: bool} for each expected model
        - model_files: dict of {model_name: list of files} per model
    """
    session_dir = Path(BASE_DATA_PATH) / mouse_id / date
    
    status = {
        'has_dict': (session_dir / 'glm_data_dict.pkl.gz').exists(),
        'has_stacked': (session_dir / 'glm_data_stacked.pkl').exists(),
        'completed_models': {},
        'model_files': {},
    }
    
    # Check each model's specific results directory
    if dataset_type in MODEL_FILE_PATTERNS:
        for model_name, model_info in MODEL_FILE_PATTERNS[dataset_type].items():
            model_dir = session_dir / model_info['dir']
            required_files = model_info['files']
            
            if model_dir.exists():
                existing_files = {f.name for f in model_dir.glob('*.pkl') 
                                 if not f.name.startswith('.')}
                status['model_files'][model_name] = sorted(
                    [f.replace('.pkl', '') for f in existing_files]
                )
                
                # Model is complete if it has both required files
                status['completed_models'][model_name] = all(
                    f in existing_files for f in required_files
                )
            else:
                status['completed_models'][model_name] = False
                status['model_files'][model_name] = []
    
    if verbose:
        for model_name, files in status['model_files'].items():
            if files:
                print(f"      [{model_name}] {', '.join(files)}")
    
    return status


def get_status_icon(status):
    """Return status icon based on completion."""
    has_any_model = any(status.get('completed_models', {}).values())
    
    if has_any_model:
        return '✓'
    elif status['has_stacked']:
        return '⚠'  # Data ready but models not run
    elif status['has_dict']:
        return '◐'  # Dict exists but not stacked
    else:
        return '✗'  # Missing prerequisite data


def print_dataset_summary(dataset_name, registry, dataset_type, verbose=False):
    """Print summary for a dataset."""
    print("\n" + "=" * 80)
    print(f"{dataset_name.upper()} Dataset")
    print("=" * 80)
    
    total_sessions = sum(len(dates) for dates in registry.values())
    stats = {
        'complete': 0,
        'has_data': 0,
        'has_dict': 0,
        'missing': 0,
    }
    
    # Track model completion across all sessions
    if dataset_type in MODEL_FILE_PATTERNS:
        expected_models = list(MODEL_FILE_PATTERNS[dataset_type].keys())
        model_completion = {m: 0 for m in expected_models}
    else:
        model_completion = {}
    
    session_statuses = []
    
    for mouse_id, dates in sorted(registry.items()):
        print(f"\n{mouse_id} ({len(dates)} sessions):")
        for date in sorted(dates):
            status = check_session_status(mouse_id, date, dataset_type, verbose)
            icon = get_status_icon(status)
            
            # Update stats
            has_any_model = any(status.get('completed_models', {}).values())
            if has_any_model:
                stats['complete'] += 1
                # Update per-model completion counts
                for model_name, is_complete in status['completed_models'].items():
                    if is_complete:
                        model_completion[model_name] += 1
            elif status['has_stacked']:
                stats['has_data'] += 1
            elif status['has_dict']:
                stats['has_dict'] += 1
            else:
                stats['missing'] += 1
            
            # Build status string
            status_parts = []
            if not status['has_dict']:
                status_parts.append('no dict')
            elif not status['has_stacked']:
                status_parts.append('no stacked')
            elif not has_any_model:
                status_parts.append('no models')
            else:
                n_complete = sum(status['completed_models'].values())
                n_total = len(status['completed_models'])
                status_parts.append(f"{n_complete}/{n_total} models")
            
            status_str = ' | '.join(status_parts)
            print(f"  {icon}  {date:20s} — {status_str}")
            
            session_statuses.append((mouse_id, date, status))
    
    # Print summary
    print("\n" + "-" * 80)
    print("SUMMARY:")
    print(f"  Total sessions:     {total_sessions}")
    print(f"  ✓ Complete:         {stats['complete']} ({stats['complete']/total_sessions*100:.1f}%)")
    print(f"  ⚠ Data ready:       {stats['has_data']} (stacked data exists, models not run)")
    print(f"  ◐ Partially ready:  {stats['has_dict']} (dict exists, needs stacking)")
    print(f"  ✗ Missing data:     {stats['missing']} (needs prepare_glm_data.py)")
    
    # Print per-model completion
    if model_completion:
        print(f"\n  Model Completion:")
        for model_name, count in sorted(model_completion.items()):
            pct = count / total_sessions * 100
            print(f"    {model_name:35s}  {count}/{total_sessions} ({pct:.1f}%)")
    
    print("-" * 80)
    
    return session_statuses


def print_quick_actions(dataset_name, registry, dataset_type):
    """Print suggested commands to complete missing steps."""
    print(f"\n📋 Quick Actions for {dataset_name}:")
    
    missing_dict = []
    missing_stacked = []
    missing_models = []
    
    for mouse_id, dates in registry.items():
        for date in dates:
            status = check_session_status(mouse_id, date, dataset_type)
            if not status['has_dict']:
                missing_dict.append((mouse_id, date))
            elif not status['has_stacked']:
                missing_stacked.append((mouse_id, date))
            elif not any(status.get('completed_models', {}).values()):
                missing_models.append((mouse_id, date))
    
    if missing_dict:
        task_map = {'avc': 'avc', 'avc_priming': 'priming', 'vacant': 'vacant'}
        task = task_map.get(dataset_name.lower(), 'all')
        print(f"  1. Build dict files ({len(missing_dict)} sessions):")
        print(f"     python core/prepare_glm_data.py --task {task} --base-path \"{BASE_DATA_PATH}\"")
    
    if missing_stacked:
        task = dataset_name.lower().replace(' ', '_')
        print(f"  2. Build stacked files ({len(missing_stacked)} sessions):")
        print(f"     python core/build_stacked_data.py --task {task}")
    
    if missing_models:
        print(f"  3. Run models ({len(missing_models)} sessions):")
        if 'priming' in dataset_name.lower():
            print(f"     python runners/run_multisensory_models.py --analysis priming --build-stacked")
        elif 'vacant' in dataset_name.lower():
            print(f"     python runners/run_multisensory_models.py --analysis vacant --build-stacked")
        elif 'avc' in dataset_name.lower():
            print(f"     python runners/run_unisensory_models.py --build-stacked")


def main():
    parser = argparse.ArgumentParser(
        description='Check GLM model completion status for all datasets.'
    )
    parser.add_argument(
        '--dataset', type=str, default='all',
        choices=['all', 'avc', 'avc_priming', 'vacant'],
        help='Which dataset to check (default: all)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show detailed model files for each session'
    )
    parser.add_argument(
        '--actions', action='store_true',
        help='Show suggested commands to complete missing steps'
    )
    
    args = parser.parse_args()
    
    datasets = []
    if args.dataset in ['all', 'avc']:
        datasets.append(('AVC (Original)', PPC_AVC, 'avc'))
    if args.dataset in ['all', 'avc_priming']:
        datasets.append(('AVC Priming', PPC_AVC_PRIMING, 'avc_priming'))
    if args.dataset in ['all', 'vacant']:
        datasets.append(('VACANT', PPC_VACANT, 'vacant'))
    
    print("\n" + "=" * 80)
    print("GLM MODEL STATUS CHECK")
    print("=" * 80)
    print(f"Base path: {BASE_DATA_PATH}")
    print("\nLegend:")
    print("  ✓  Complete (at least one model finished)")
    print("  ⚠  Data ready (stacked data exists, models not run)")
    print("  ◐  Partially ready (dict exists, needs stacking)")
    print("  ✗  Missing data (needs prepare_glm_data.py)")
    
    for dataset_name, registry, dataset_type in datasets:
        print_dataset_summary(dataset_name, registry, dataset_type, args.verbose)
        if args.actions:
            print_quick_actions(dataset_name, registry, dataset_type)
    
    print("\n" + "=" * 80)
    print("Status check complete!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()