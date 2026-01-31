"""
Real Drug Validation - Liraglutide & Semaglutide
=================================================
Tests the trained mPBPK-ML model with real drug parameters.

Drug Data Sources:
- Liraglutide (Victoza): FDA label, DrugBank
- Semaglutide (Ozempic/Wegovy): FDA label, DrugBank

Note: These are GLP-1 receptor agonists (peptides), not antibodies.
The model was trained on antibody-like parameters, so predictions
are for comparison/validation purposes.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from pathlib import Path
from src.models.mpbpk_ml import mPBPKClassifier, MLConfig

# =============================================================================
# Real Drug Parameters (from literature/FDA labels)
# =============================================================================

REAL_DRUGS = {
    'Liraglutide': {
        # FDA-approved GLP-1 agonist (Victoza, Saxenda)
        'KD_nM': 0.5,           # High affinity to GLP-1R (~0.1-1 nM)
        'dose_mg_kg': 0.02,     # 1.2-1.8 mg/day for ~70kg = ~0.02 mg/kg
        'MW_kDa': 3.75,         # 3751 Da (peptide, not antibody)
        'halflife_hr': 13,      # Terminal half-life ~13 hours
        'T0_nM': 1.0,           # GLP-1R expression level (estimated)
        'charge': 0,            # Neutral
        'clinical_status': 'FDA Approved (2010)',
        'indication': 'Type 2 Diabetes, Obesity',
    },
    'Semaglutide': {
        # FDA-approved GLP-1 agonist (Ozempic, Wegovy, Rybelsus)
        'KD_nM': 0.4,           # High affinity to GLP-1R
        'dose_mg_kg': 0.035,    # 2.4 mg/week = ~0.035 mg/kg/day equivalent
        'MW_kDa': 4.11,         # 4113 Da (peptide)
        'halflife_hr': 168,     # ~7 days (weekly dosing)
        'T0_nM': 1.0,           # GLP-1R expression level (estimated)
        'charge': 0,            # Neutral
        'clinical_status': 'FDA Approved (2017)',
        'indication': 'Type 2 Diabetes, Obesity, CV risk reduction',
    },
}

# Patient scenarios to test
PATIENT_SCENARIOS = [
    {'population': 'EUR', 'phenotype': 'NM', 'activity_score': 2.0, 'cl_multiplier': 1.0},
    {'population': 'EAS', 'phenotype': 'IM', 'activity_score': 1.0, 'cl_multiplier': 0.7},
    {'population': 'AFR', 'phenotype': 'PM', 'activity_score': 0.0, 'cl_multiplier': 0.3},
    {'population': 'EUR', 'phenotype': 'UM', 'activity_score': 3.0, 'cl_multiplier': 1.8},
]

def create_feature_vector(drug_params: dict, patient: dict, feature_names: list) -> np.ndarray:
    """Create feature vector for prediction."""
    # Combine drug and patient parameters
    params = {
        'log_KD': np.log10(drug_params['KD_nM']),
        'log_dose': np.log10(drug_params['dose_mg_kg']),
        'charge': drug_params['charge'],
        'log_MW': np.log10(drug_params['MW_kDa']),
        'log_T0': np.log10(drug_params['T0_nM']),
        'log_halflife': np.log10(drug_params['halflife_hr']),
        **patient
    }
    
    features = []
    for feat in feature_names:
        if feat in params:
            features.append(params[feat])
        elif feat.startswith('pop_'):
            pop = feat.replace('pop_', '')
            features.append(1 if params.get('population') == pop else 0)
        elif feat == 'phenotype_encoded':
            pheno_map = {'PM': 0, 'IM': 1, 'NM': 2, 'UM': 3}
            features.append(pheno_map.get(params.get('phenotype', 'NM'), 2))
        else:
            features.append(0)
    
    return np.array(features).reshape(1, -1)

def run_validation():
    print("=" * 70)
    print("  REAL DRUG VALIDATION: Liraglutide & Semaglutide")
    print("=" * 70)
    
    # Train classifier
    print("\n1. Training model on virtual data...")
    classifier = mPBPKClassifier(MLConfig(use_smote=True))
    classifier.train()
    feature_names = classifier.feature_names
    
    print("\n2. Testing real drugs...")
    print("=" * 70)
    
    results = []
    
    for drug_name, drug_params in REAL_DRUGS.items():
        print(f"\n📊 {drug_name}")
        print(f"   Clinical Status: {drug_params['clinical_status']}")
        print(f"   Indication: {drug_params['indication']}")
        print(f"   KD: {drug_params['KD_nM']} nM | MW: {drug_params['MW_kDa']} kDa | t½: {drug_params['halflife_hr']} hr")
        print("-" * 50)
        
        for patient in PATIENT_SCENARIOS:
            X = create_feature_vector(drug_params, patient, feature_names)
            prob = classifier.rf_model.predict_proba(X)[0][1]
            pred = classifier.rf_model.predict(X)[0]
            
            pop = patient['population']
            pheno = patient['phenotype']
            result = 'SUCCESS' if pred else 'FAIL'
            
            print(f"   {pop}/{pheno}: P(success) = {prob:.3f} → {result}")
            
            results.append({
                'drug': drug_name,
                'population': pop,
                'phenotype': pheno,
                'probability': prob,
                'prediction': result,
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)
    
    print("""
    ⚠️  IMPORTANT NOTES:
    
    1. This model was trained on ANTIBODY parameters (MW ~150 kDa)
       Liraglutide/Semaglutide are PEPTIDES (MW ~4 kDa)
       
    2. The model may show lower success predictions because:
       - Peptide doses are much lower than typical antibody doses
       - MW is outside training distribution
       
    3. Clinical reality: Both drugs are FDA-approved and highly effective
       - Model predictions should be interpreted with caution for non-antibodies
    """)
    
    # Average predictions
    for drug_name in REAL_DRUGS.keys():
        drug_results = [r for r in results if r['drug'] == drug_name]
        avg_prob = np.mean([r['probability'] for r in drug_results])
        success_rate = sum(1 for r in drug_results if r['prediction'] == 'SUCCESS') / len(drug_results)
        print(f"   {drug_name}: Avg P(success) = {avg_prob:.3f}, Success rate = {success_rate*100:.0f}%")
    
    # Save results
    output_path = Path('data/processed/real_drug_validation_results.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("REAL DRUG VALIDATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        for r in results:
            f.write(f"{r['drug']} | {r['population']}/{r['phenotype']}: P={r['probability']:.3f} → {r['prediction']}\n")
    
    print(f"\n✅ Results saved to: {output_path}")

if __name__ == "__main__":
    run_validation()
