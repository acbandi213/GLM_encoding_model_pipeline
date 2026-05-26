#!/usr/bin/env python
"""
Verification script to ensure all imports are updated after migration.
Run this to check for any remaining old-style imports.
"""

import os
import re
import sys
from pathlib import Path

GLM_ROOT = Path(__file__).parent
print(f"Checking: {GLM_ROOT}\n")

# Patterns to look for (old-style imports that should be fixed)
OLD_PATTERNS = [
    (r'^from session_pipeline import', 'Should be: from core.session_pipeline import'),
    (r'^from session_registry import', 'Should be: from config.session_registry import'),
    (r'^from session_registry_multisensory import', 'Should be: from config.session_registry_multisensory import'),
    (r'^from glm_class import', 'Should be: from core.glm_class import'),
    (r'^from predictor_builder import', 'Should be: from core.predictor_builder import'),
    (r'^from datasets import', 'Should be: from core.datasets import'),
    (r'^import glm_class$', 'Should be: import core.glm_class as glm_class'),
    (r'^import session_pipeline$', 'Should be: import core.session_pipeline as session_pipeline'),
]

# Directories to check
DIRS_TO_CHECK = ['core', 'config', 'models', 'runners', 'analysis', 'testing']

issues_found = []

# Check Python files
for dir_name in DIRS_TO_CHECK:
    dir_path = GLM_ROOT / dir_name
    if not dir_path.exists():
        continue
    
    for py_file in dir_path.rglob('*.py'):
        try:
            with open(py_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    for pattern, suggestion in OLD_PATTERNS:
                        if re.match(pattern, line.strip()):
                            # Ignore if it's already the new style
                            if 'from core.' in line or 'from config.' in line or 'as glm_class' in line:
                                continue
                            issues_found.append({
                                'file': str(py_file.relative_to(GLM_ROOT)),
                                'line': line_num,
                                'content': line.strip(),
                                'suggestion': suggestion
                            })
        except Exception as e:
            print(f"Error reading {py_file}: {e}")

# Check Jupyter notebooks
notebook_dir = GLM_ROOT / 'analysis' / 'notebooks'
if notebook_dir.exists():
    for nb_file in notebook_dir.glob('*.ipynb'):
        try:
            with open(nb_file, 'r') as f:
                content = f.read()
                for pattern, suggestion in OLD_PATTERNS:
                    if re.search(pattern, content, re.MULTILINE):
                        # Check if it's already fixed
                        if 'from core.' in content or 'from config.' in content:
                            continue
                        issues_found.append({
                            'file': str(nb_file.relative_to(GLM_ROOT)),
                            'line': '?',
                            'content': f'Old import pattern found: {pattern}',
                            'suggestion': suggestion
                        })
                        break  # Only report once per notebook
        except Exception as e:
            print(f"Error reading {nb_file}: {e}")

# Report results
print("=" * 70)
print("MIGRATION VERIFICATION RESULTS")
print("=" * 70)
print()

if not issues_found:
    print("✅ SUCCESS! No old-style imports found.")
    print("   All files have been properly updated to use the new import structure.")
    print()
    print("   You can now safely:")
    print("   1. Run your models")
    print("   2. Use your notebooks")
    print("   3. Delete the 'old/' folder if you want")
    sys.exit(0)
else:
    print(f"⚠️  Found {len(issues_found)} file(s) with old-style imports:\n")
    
    for issue in issues_found:
        print(f"File: {issue['file']}")
        print(f"Line: {issue['line']}")
        print(f"  Found: {issue['content']}")
        print(f"  {issue['suggestion']}")
        print()
    
    print("Please fix these imports before running your code.")
    sys.exit(1)
