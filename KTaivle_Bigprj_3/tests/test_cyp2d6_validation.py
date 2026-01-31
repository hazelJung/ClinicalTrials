"""
CYP2D6 Virtual Cohort Validation Script

Validates:
1. Phenotype distribution against literature values
2. Sample size stability
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.parse_cyp2d6 import PharmVarCYP2D6Parser

# Literature reference values for CYP2D6 phenotype distributions
# Sources: CPIC Guidelines, Gaedigk et al. (2017), Bradford (2002)
LITERATURE_REFERENCE = {
    'EUR': {'PM': (0.05, 0.10), 'IM': (0.10, 0.20), 'NM': (0.70, 0.85), 'UM': (0.01, 0.05)},
    'EAS': {'PM': (0.00, 0.02), 'IM': (0.35, 0.55), 'NM': (0.45, 0.65), 'UM': (0.00, 0.02)},
    'AFR': {'PM': (0.01, 0.05), 'IM': (0.20, 0.35), 'NM': (0.60, 0.75), 'UM': (0.02, 0.10)},
    'AMR': {'PM': (0.02, 0.08), 'IM': (0.15, 0.30), 'NM': (0.65, 0.80), 'UM': (0.01, 0.05)},
    'SAS': {'PM': (0.01, 0.05), 'IM': (0.15, 0.30), 'NM': (0.65, 0.80), 'UM': (0.01, 0.05)},
}


def validate_phenotype_distribution(parser: PharmVarCYP2D6Parser, n_samples: int = 10000):
    """Validate phenotype distributions against literature."""
    results = {}
    
    for pop in parser.POPULATIONS:
        dist = parser.get_phenotype_distribution(pop, n_samples=n_samples)
        lit_ref = LITERATURE_REFERENCE.get(pop, {})
        
        pop_result = {
            'distribution': dist,
            'validation': {},
            'pass': True
        }
        
        for pheno in ['PM', 'IM', 'NM', 'UM']:
            value = dist.get(pheno, 0)
            ref_range = lit_ref.get(pheno, (0, 1))
            in_range = ref_range[0] <= value <= ref_range[1]
            
            pop_result['validation'][pheno] = {
                'value': value,
                'reference_range': ref_range,
                'in_range': in_range
            }
            
            if not in_range:
                pop_result['pass'] = False
        
        results[pop] = pop_result
    
    return results


def test_sample_size_stability(parser: PharmVarCYP2D6Parser, population: str = 'EUR'):
    """Test distribution stability across sample sizes."""
    sample_sizes = [100, 500, 1000, 5000, 10000]
    results = {}
    
    for n in sample_sizes:
        dist = parser.get_phenotype_distribution(population, n_samples=n)
        results[n] = dist
    
    # Calculate coefficient of variation across sample sizes
    import numpy as np
    phenotypes = ['PM', 'IM', 'NM', 'UM']
    stability = {}
    
    for pheno in phenotypes:
        values = [results[n][pheno] for n in sample_sizes]
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = (std_val / mean_val * 100) if mean_val > 0 else 0
        stability[pheno] = {
            'mean': mean_val,
            'std': std_val,
            'cv_percent': cv,
            'stable': cv < 20  # CV < 20% considered stable
        }
    
    return {
        'sample_sizes': sample_sizes,
        'distributions': results,
        'stability': stability
    }


def main():
    print("=" * 60)
    print("CYP2D6 Virtual Cohort Validation")
    print("=" * 60)
    
    # Initialize parser
    parser = PharmVarCYP2D6Parser()
    
    # 1. Validate phenotype distributions
    print("\n[1] Phenotype Distribution Validation")
    print("-" * 40)
    
    validation_results = validate_phenotype_distribution(parser, n_samples=10000)
    
    for pop, result in validation_results.items():
        status = "✅ PASS" if result['pass'] else "❌ FAIL"
        print(f"\n{parser.POPULATION_NAMES[pop]} ({pop}): {status}")
        
        for pheno, val_info in result['validation'].items():
            mark = "✓" if val_info['in_range'] else "✗"
            print(f"  {pheno}: {val_info['value']*100:.1f}% "
                  f"(ref: {val_info['reference_range'][0]*100:.0f}-{val_info['reference_range'][1]*100:.0f}%) [{mark}]")
    
    # 2. Sample size stability test
    print("\n" + "=" * 60)
    print("[2] Sample Size Stability Test (EUR)")
    print("-" * 40)
    
    stability_results = test_sample_size_stability(parser, 'EUR')
    
    print("\nDistribution by sample size:")
    print(f"{'N':>8} | {'PM':>8} | {'IM':>8} | {'NM':>8} | {'UM':>8}")
    print("-" * 50)
    for n in stability_results['sample_sizes']:
        dist = stability_results['distributions'][n]
        print(f"{n:>8} | {dist['PM']*100:>7.1f}% | {dist['IM']*100:>7.1f}% | "
              f"{dist['NM']*100:>7.1f}% | {dist['UM']*100:>7.1f}%")
    
    print("\nStability Analysis (CV%):")
    for pheno, stab in stability_results['stability'].items():
        status = "Stable" if stab['stable'] else "Variable"
        print(f"  {pheno}: CV={stab['cv_percent']:.1f}% - {status}")
    
    # Save results as JSON for report generation
    output = {
        'validation': {},
        'stability': stability_results
    }
    
    for pop, result in validation_results.items():
        output['validation'][pop] = {
            'distribution': result['distribution'],
            'pass': result['pass'],
            'details': result['validation']
        }
    
    output_path = Path(__file__).parent / 'validation_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=float)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    all_pass = all(r['pass'] for r in validation_results.values())
    print(f"Phenotype Distribution: {'✅ ALL PASS' if all_pass else '❌ SOME FAILED'}")
    all_stable = all(s['stable'] for s in stability_results['stability'].values())
    print(f"Sample Size Stability: {'✅ STABLE' if all_stable else '⚠️ VARIABLE'}")


if __name__ == "__main__":
    main()
