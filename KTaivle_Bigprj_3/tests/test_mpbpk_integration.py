"""Test mPBPK + CYP2D6 Integration"""
import sys
sys.path.insert(0, '.')

from src.models.mpbpk_engine import (
    simulate_population_cohort, 
    summarize_cohort_results,
    DrugParams, 
    TargetParams
)

# Test drug parameters
drug = DrugParams(KD_nM=1.0, dose_mg=100)
target = TargetParams(baseline_nM=10)

print('Running EUR population cohort simulation (n=20)...')
results = simulate_population_cohort(drug, target, population='EUR', n_subjects=20)

print('\n=== Sample Results ===')
for r in results[:5]:
    print(f'{r["diplotype"]:12} | {r["phenotype"]:4} | TO={r["TO_trough"]:.1f}% | CL_mult={r["cl_multiplier"]}')

print('\n=== Summary by Phenotype ===')
summary = summarize_cohort_results(results)
for pheno, stats in summary.items():
    if pheno != 'overall':
        print(f'{pheno}: n={stats["count"]}, TO={stats["TO_trough_mean"]:.1f}%, success={stats["success_rate"]:.0f}%')
print(f'\nOverall: n={summary["overall"]["total"]}, success={summary["overall"]["success_rate"]:.0f}%')
