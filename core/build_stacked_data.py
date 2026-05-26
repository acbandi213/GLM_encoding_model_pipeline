#!/usr/bin/env python
"""
Build glm_data_stacked.pkl files for various session types.

Usage:
    python core/build_stacked_data.py --task avc
    python core/build_stacked_data.py --task avc_priming
    python core/build_stacked_data.py --task vacant
    python core/build_stacked_data.py --task vacant --metric dff
    python core/build_stacked_data.py --task avc --mouse-id IS-2-1L
"""

import argparse
import os
import sys
import gc

# Add GLM_encoding_model to path (parent directory of core/)
GLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GLM_DIR not in sys.path:
    sys.path.insert(0, GLM_DIR)

from core.session_pipeline import (
    BASE_DATA_PATH,
    extract_data,
    generate_predictors,
    stack_design_matrices,
    save_glm_data,
)
from config.session_registry import PPC_AVC
from config.session_registry_multisensory import (
    PPC_AVC_PRIMING,
    PPC_VACANT,
)

def build_stacked_for_one_session(mouse_ID, date, metric='deconv'):
    """Build glm_data_stacked.pkl for a single session."""
    # For VACANT sessions, keep the _1 suffix in the filesystem path
    # (it's part of the actual directory name)
    session_dir = os.path.join(BASE_DATA_PATH, mouse_ID, date)
    dict_path = os.path.join(session_dir, 'glm_data_dict.pkl.gz')
    stacked_path = os.path.join(session_dir, 'glm_data_stacked.pkl')
    
    # Debug: print the actual path we're checking
    print(f"  Checking: {stacked_path}")
    
    # Check if stacked already exists
    if os.path.exists(stacked_path):
        print(f"  ✓ {mouse_ID}/{date} — already has glm_data_stacked.pkl")
        return True
    
    # Check if dict exists
    if not os.path.exists(dict_path):
        print(f"  ✗ {mouse_ID}/{date} — glm_data_dict.pkl.gz not found!")
        return False
    
    try:
        print(f"  Building {mouse_ID}/{date}...", end=' ', flush=True)
        
        # Load the dict data using extract_data function
        glm_data_dict = extract_data(mouse_ID, date)
        
        # Generate predictors
        design_matrices, predictor_names, predictor_info = generate_predictors(
            glm_data_dict
        )
        
        # Stack into design matrix
        X, Y, metadata, group_indices = stack_design_matrices(
            glm_data_dict,
            design_matrices,
            predictor_info,
            neural_metric=metric
        )
        
        # Save stacked data
        save_glm_data(
            X, Y, design_matrices, predictor_names, predictor_info,
            metadata, group_indices, mouse_ID, date
        )
        
        print("✓ Done!")
        gc.collect()
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_session_registry(task_type):
    """Return the appropriate session registry based on task type."""
    if task_type == 'avc':
        return PPC_AVC, 'AVC (original dataset)'
    elif task_type == 'avc_priming':
        return PPC_AVC_PRIMING, 'AVC Priming'
    elif task_type == 'vacant':
        return PPC_VACANT, 'VACANT'
    else:
        raise ValueError(f"Unknown task type: {task_type}. "
                        f"Choose from: avc, avc_priming, vacant")


def main():
    parser = argparse.ArgumentParser(
        description='Build glm_data_stacked.pkl files for specified sessions.'
    )
    parser.add_argument(
        '--task', type=str, required=True,
        choices=['avc', 'avc_priming', 'vacant'],
        help='Task type: avc (original) | avc_priming | vacant'
    )
    parser.add_argument(
        '--metric', type=str, default='deconv',
        help='Neural metric: deconv | dff | z_dff (default: deconv)'
    )
    parser.add_argument(
        '--mouse-id', type=str,
        help='Process only this mouse ID (optional)'
    )
    parser.add_argument(
        '--date', type=str,
        help='Process only this date (optional, requires --mouse-id)'
    )
    args = parser.parse_args()
    
    # Validate date argument
    if args.date and not args.mouse_id:
        parser.error("--date requires --mouse-id")
    
    # Get the appropriate session registry
    session_registry, task_name = get_session_registry(args.task)
    
    # Filter by mouse_id if specified
    if args.mouse_id:
        if args.mouse_id not in session_registry:
            print(f"ERROR: Mouse ID '{args.mouse_id}' not found in {task_name} registry")
            return
        session_registry = {args.mouse_id: session_registry[args.mouse_id]}
        
        # Filter by date if specified
        if args.date:
            if args.date not in session_registry[args.mouse_id]:
                print(f"ERROR: Date '{args.date}' not found for {args.mouse_id}")
                return
            session_registry = {args.mouse_id: [args.date]}
    
    print("=" * 70)
    print(f"Building glm_data_stacked.pkl for {task_name} sessions")
    print(f"Metric: {args.metric}")
    print("=" * 70)
    
    total = 0
    success = 0
    failed = 0
    
    for mouse_ID, dates in session_registry.items():
        print(f"\n{mouse_ID}:")
        for date in dates:
            total += 1
            if build_stacked_for_one_session(mouse_ID, date, metric=args.metric):
                success += 1
            else:
                failed += 1
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {success}/{total} successful, {failed} failed")
    print("=" * 70)


if __name__ == '__main__':
    main()
