
"""
PKSmart PBPK & Virtual Cohort Runner
====================================
Runs PK prediction followed by PBPK/Cohort Simulation.
Separated from main pksmart_predict.py script.
"""

import argparse
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from pksmart.features import generate_features
from pksmart.models import PKSmartPipeline
from pksmart.monte_carlo_simulator import MonteCarloSimulator

def main():
    parser = argparse.ArgumentParser(description="PKSmart PBPK & Virtual Cohort Study")
    parser.add_argument("--smiles", type=str, required=False, help="SMILES string (optional if MW/CL/Vd provided)")
    parser.add_argument("--dose", type=float, default=100.0, help="Dose in mg")
    parser.add_argument("--pop", type=str, default="EUR", help="Population code (EUR, EAS, AFR, etc.)")
    parser.add_argument("--n_subjects", type=int, default=1000, help="Number of virtual subjects")
    parser.add_argument("--model_dir", type=str, default=".", help="Path to models directory")
    
    # Direct PK Input Overrides
    parser.add_argument("--cl", type=float, help="Override/Input Clearance (mL/min/kg)")
    parser.add_argument("--vd", type=float, help="Override/Input Vd (L/kg)")
    parser.add_argument("--mw", type=float, help="Override/Input Molecular Weight (g/mol)")
    parser.add_argument("--fu", type=float, help="Override/Input Fraction Unbound (0-1)")

    # Demographic Overrides
    parser.add_argument("--gender", type=str, default="BOTH", choices=["M", "F", "BOTH"], help="Gender filter: M, F, or BOTH (default)")
    parser.add_argument("--female_ratio", type=float, default=0.5, help="Ratio of females when gender is BOTH (0.0 - 1.0, default 0.5)")
    parser.add_argument("--age_min", type=int, default=18, help="Minimum Age (default: 18)")
    parser.add_argument("--age_max", type=int, default=80, help="Maximum Age (default: 80)")
    
    args = parser.parse_args()
    
    cl_pred = 0.0
    vd_pred = 0.0
    
    # Logic for PK Parameters (CL, Vd)
    if args.cl is not None and args.vd is not None:
        print(f"\n[1/2] Using provided PK parameters (Skipping Prediction)...")
        cl_pred = args.cl
        vd_pred = args.vd
    elif args.smiles:
        # Run Prediction Pipeline
        # 1. Feature Generation
        print(f"\n[1/3] Generating Features for: {args.smiles}")
        features_df = generate_features([args.smiles])
        if features_df is None or features_df.empty:
            print("Error: Feature generation failed.")
            return
    
        # 2. PK Prediction
        print("\n[2/3] Predicting PK Parameters...")
        if not os.path.exists(args.model_dir):
            if os.path.exists("./models"):
                args.model_dir = "."
            else:
                print(f"Error: Model directory not found: {args.model_dir}")
                return
                
        pipeline = PKSmartPipeline(os.path.join(args.model_dir, "models"))
        pk_results = pipeline.run_pipeline(features_df)
        
        try:
            cl_pred = pk_results["human_CL_mL_min_kg (Linear)"].values[0]
            vd_pred = pk_results["human_VDss_L_kg (Linear)"].values[0]
        except KeyError as e:
            print(f"Error: PK prediction missing columns: {e}")
            return
    else:
        print("Error: Either provide (--cl AND --vd) OR provide --smiles")
        return
            
    print(f"  Input CL: {cl_pred:.2f} mL/min/kg")
    print(f"  Input Vd: {vd_pred:.2f} L/kg")
    if args.fu:
        print(f"  Input Fu: {args.fu:.3f}")
    
    # 3. PBPK / Virtual Cohort Simulation
    print(f"\n[3/3] Running Virtual Cohort Simulation ({args.pop}, n={args.n_subjects})...")
    print(f"  Demographics: Gender={args.gender} (Female Ratio={args.female_ratio if args.gender == 'BOTH' else 'N/A'}), Age={args.age_min}-{args.age_max}")
    
    # Unit Conversions
    # Note: We rely on the engine to handle demographic-specific weight scaling if we pass per-kg values? 
    # Current engine takes total L/day. We convert using standard 70kg as 'typical' reference.
    patient_weight = 70.0 
    
    cl_L_day = cl_pred * patient_weight * 1.44
    vd_L = vd_pred * patient_weight
    
    print(f"  Converted CL (Typical 70kg): {cl_L_day:.2f} L/day")
    print(f"  Converted Vd (Typical 70kg): {vd_L:.2f} L Total")
    
    # Calculate MW
    mw_kda = 0.400 # default
    if args.mw:
        mw_kda = args.mw / 1000.0
        print(f"  Using provided MW: {args.mw} g/mol")
    elif args.smiles:
        mol = Chem.MolFromSmiles(args.smiles)
        if mol:
            mw_kda = Descriptors.MolWt(mol) / 1000.0
            print(f"  Calculated MW from SMILES: {mw_kda*1000:.1f} g/mol")
    else:
        print("Error: Either --mw OR --smiles is required for Molecular Weight.")
        return
    
    from pksmart.mpbpk_engine import DrugParams, TargetParams, simulate_population_cohort
    
    drug = DrugParams(dose_mg=args.dose, MW_kDa=mw_kda)
    target = TargetParams() # Default target
    
    # Run Simulation
    try:
        results = simulate_population_cohort(drug, target, args.pop, args.n_subjects,
                                             predicted_CL_L_day=cl_L_day, 
                                             predicted_Vd_L=vd_L,
                                             predicted_Fu=args.fu,
                                             gender_mode=args.gender,
                                             age_min=args.age_min,
                                             age_max=args.age_max,
                                             female_ratio=args.female_ratio)
    except Exception as e:
        print(f"Error during simulation: {e}")
        return

    # Analyze Results
    if not results:
        print("No simulation results returned.")
        return
        
    cmax_values = [r.get('C_max', 0) for r in results]
    success_count = sum(1 for r in results if r.get('success', False))
    
    # Convert Cmax to uM if engine returns nM, or assume uM.
    cmax_uM = [c / 1000.0 for c in cmax_values]
    
    mean_cmax = np.mean(cmax_uM)
    p95_cmax = np.percentile(cmax_uM, 95)
    
    high_exp_threshold = mean_cmax * 2.0
    pct_high = sum(1 for c in cmax_uM if c > high_exp_threshold) / len(cmax_uM) * 100
    
    print("\n" + "="*50)
    print("VIRTUAL COHORT RESULTS (mPBPK Engine)")
    print("="*50)
    print(f"Population: {args.pop}")
    print(f"Mean Cmax: {mean_cmax:.3f} uM")
    print(f"Cmax (95th %): {p95_cmax:.3f} uM")
    print(f"Target Success Rate: {success_count / len(results) * 100:.1f}%")
    print(f"High Exposure (>2x Mean): {pct_high:.1f}%")
            
    # Phenotype summary
    phenos = [r.get('phenotype', 'Unknown') for r in results]
    from collections import Counter
    pheno_counts = Counter(phenos)
    
    print("\nPhenotype Distribution:")
    for p, count in pheno_counts.items():
        print(f"  {p}: {count/len(results)*100:.1f}%")
            
    print("="*50)

if __name__ == "__main__":
    main()
