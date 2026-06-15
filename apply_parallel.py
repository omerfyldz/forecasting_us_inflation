"""
Script to apply parallel processing to ALL rolling window functions.
This script automatically converts sequential for-loops to parallel execution.

Run: python apply_parallel.py
"""

import os
import re

def process_file(filepath):
    """Process a single file and add parallel processing."""
    
    if not os.path.exists(filepath):
        print(f"  SKIP: File not found")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already parallelized
    if 'Parallel(n_jobs=' in content:
        print(f"  SKIP: Already parallelized")
        return False
    
    original_content = content
    modified = False
    
    # Step 1: Add joblib import after other imports
    if 'from joblib import Parallel' not in content:
        if 'from utils import' in content:
            content = content.replace(
                'from utils import',
                'from joblib import Parallel, delayed\nfrom utils import'
            )
            modified = True
    
    # Step 2: Add N_JOBS constant after imports (before first function)
    if 'N_JOBS = ' not in content:
        # Find first "def " that starts a function
        match = re.search(r'\n\ndef (\w+)\(', content)
        if match:
            insert_pos = match.start()
            n_jobs_line = "\n\n# Number of parallel jobs (-1 = use all CPU cores)\nN_JOBS = -1"
            content = content[:insert_pos] + n_jobs_line + content[insert_pos:]
            modified = True
    
    # Step 3: Find and replace the for-loop pattern in rolling_window functions
    # This pattern matches the common structure across most files
    
    # Pattern for simple models (no extra data like feature_importances)
    simple_for_pattern = r'''(Y = np\.array\(Y\)\s*\n\s*)save_pred = np\.full\(\(nprev, 1\), np\.nan\)\s*\n\s*for i in range\(nprev, 0, -1\):\s*\n\s*# Window selection\s*\n\s*Y_window = Y\[\(nprev - i\):\(Y\.shape\[0\] - i\), :\]\s*\n\s*# Run (\w+) model\s*\n\s*result = (\w+)\(Y_window, indice, lag[^)]*\)\s*\n\s*idx = nprev - i\s*\n\s*save_pred\[idx, 0\] = result\['pred'\]\s*\n\s*print\(f"iteration \{idx \+ 1\}"\)'''
    
    match = re.search(simple_for_pattern, content)
    if match:
        prefix = match.group(1)
        model_name = match.group(2)
        func_name = match.group(3)
        
        replacement = f'''{prefix}def process_single_iteration(i, Y, nprev, indice, lag):
        """Process a single iteration - designed for parallel execution."""
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        result = {func_name}(Y_window, indice, lag)
        idx = nprev - i
        return idx, result['pred']
    
    print(f"Running {{nprev}} {model_name} iterations in parallel (N_JOBS={{N_JOBS}})...")
    
    # Parallel execution
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_single_iteration)(i, Y, nprev, indice, lag)
        for i in range(nprev, 0, -1)
    )
    
    # Collect results
    save_pred = np.full((nprev, 1), np.nan)
    for idx, pred in results:
        save_pred[idx, 0] = pred'''
        
        content = re.sub(simple_for_pattern, replacement, content)
        modified = True
    
    # Pattern for RF with feature_importances
    rf_for_pattern = r'''(Y = np\.array\(Y\)\s*\n\s*)save_importance = \[\]\s*\n\s*save_pred = np\.full\(\(nprev, 1\), np\.nan\)\s*\n\s*for i in range\(nprev, 0, -1\):\s*\n\s*# Window selection\s*\n\s*Y_window = Y\[\(nprev - i\):\(Y\.shape\[0\] - i\), :\]\s*\n\s*# Run (\w+) model\s*\n\s*result = (\w+)\(Y_window, indice, lag\)\s*\n\s*idx = nprev - i\s*\n\s*save_pred\[idx, 0\] = result\['pred'\]\s*\n\s*save_importance\.append\(result\['model'\]\.feature_importances_\)\s*\n\s*print\(f"iteration \{idx \+ 1\}"\)'''
    
    match = re.search(rf_for_pattern, content)
    if match:
        prefix = match.group(1)
        model_name = match.group(2)
        func_name = match.group(3)
        
        replacement = f'''{prefix}def process_single_iteration(i, Y, nprev, indice, lag):
        """Process a single iteration - designed for parallel execution."""
        Y_window = Y[(nprev - i):(Y.shape[0] - i), :]
        result = {func_name}(Y_window, indice, lag)
        idx = nprev - i
        return idx, result['pred'], result['model'].feature_importances_
    
    print(f"Running {{nprev}} {model_name} iterations in parallel (N_JOBS={{N_JOBS}})...")
    
    # Parallel execution
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_single_iteration)(i, Y, nprev, indice, lag)
        for i in range(nprev, 0, -1)
    )
    
    # Collect results
    save_pred = np.full((nprev, 1), np.nan)
    save_importance = [None] * nprev
    for idx, pred, importance in results:
        save_pred[idx, 0] = pred
        save_importance[idx] = importance'''
        
        content = re.sub(rf_for_pattern, replacement, content)
        modified = True
    
    if modified and content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  UPDATED!")
        return True
    elif modified:
        print(f"  PARTIAL: Added imports only (for-loop pattern didn't match)")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    else:
        print(f"  SKIP: No changes needed or pattern not matched")
        return False


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # All function files to process
    without_dummy_functions = [
        'func_ar.py', 'func_rf.py', 'func_xgb.py', 'func_nn.py', 
        'func_boosting.py', 'func_bag.py', 'func_csr.py', 'func_fact.py',
        'func_tfact.py', 'func_scad.py', 'func_jn.py', 'func_rfols.py',
        'func_adalassorf.py', 'func_polilasso.py', 'func_lasso.py', 'func_lstm.py'
    ]
    
    with_dummy_functions = [
        'func_ar.py', 'func_rf.py', 'func_xgb.py', 'func_nn.py',
        'func_boosting.py', 'func_bag.py', 'func_csr.py', 'func_fact.py',
        'func_tfact.py', 'func_scad.py', 'func_jn.py', 'func_rfols.py',
        'func_adalassorf.py', 'func_polilasso.py', 'func_lasso.py', 'func_lstm.py',
        'func_flasso.py', 'func_rflasso.py'
    ]
    
    print("=" * 70)
    print("APPLYING PARALLEL PROCESSING TO ALL FUNCTION FILES")
    print("=" * 70)
    
    updated_count = 0
    
    # Process without_dummy
    print("\n--- Without Dummy Functions ---")
    for filename in without_dummy_functions:
        filepath = os.path.join(base_dir, 'without_dummy', 'functions', filename)
        print(f"\n{filename}:", end=" ")
        if process_file(filepath):
            updated_count += 1
    
    # Process with_dummy
    print("\n\n--- With Dummy Functions ---")
    for filename in with_dummy_functions:
        filepath = os.path.join(base_dir, 'with_dummy', 'functions', filename)
        print(f"\n{filename}:", end=" ")
        if process_file(filepath):
            updated_count += 1
    
    print("\n\n" + "=" * 70)
    print(f"COMPLETE: Updated {updated_count} files")
    print("=" * 70)
    print("\nExpected speedup: 3-8x depending on your CPU cores")


if __name__ == "__main__":
    main()
