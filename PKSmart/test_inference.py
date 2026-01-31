
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from app.inference import run_cohort_simulation

print("Testing run_cohort_simulation...")
try:
    results = run_cohort_simulation(
        cl_ml_min_kg=1.5,
        vd_l_kg=1.2,
        mw_g_mol=350.0,
        fup=0.1,
        pop="EUR",
        n_subjects=10 # Small N for speed
    )
    
    if isinstance(results, dict) and "error" in results:
        print(f"FAILED: {results['error']}")
    elif isinstance(results, list):
        print(f"SUCCESS: Generated {len(results)} subjects.")
        print(f"Sample Result: {results[0]}")
    else:
        print(f"UNKNOWN RESULT: {type(results)}")
        
except Exception as e:
    print(f"EXCEPTION: {e}")
