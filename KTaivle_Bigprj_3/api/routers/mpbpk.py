from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import mPBPK Engine
# Import mPBPK Engine
try:
    from src.models.mpbpk_engine import (
        DrugParams, PatientParams, TargetParams, PKConstants, mPBPKEngine
    )
    from src.data.parse_cyp2d6 import PharmVarCYP2D6Parser
except ImportError as e:
    print(f"Import Error: {e}")
    pass

router = APIRouter(
    prefix="/api/mpbpk",
    tags=["mpbpk"],
)

class SimulationRequest(BaseModel):
    # Drug
    drug_name: str
    kd: float
    dose: float
    mw: float
    charge: int
    # Target
    t0: float
    halflife: float
    # Cohort
    ethnicities: List[str]
    population_size: int
    male_ratio: int
    weight_mean: float
    weight_sd: float

# Singleton Parser
_cyp_parser = None

def get_parser():
    global _cyp_parser
    if _cyp_parser is None:
        # Assuming data is in data/raw relative to project root
        data_path = PROJECT_ROOT / "data" / "raw"
        _cyp_parser = PharmVarCYP2D6Parser(data_dir=data_path)
        # Preload data if needed method exists, but __init__ likely does it or lazy loads
    return _cyp_parser

@router.post("/simulate")
async def run_simulation(req: SimulationRequest):
    """
    Run mPBPK simulation on a custom cohort.
    """
    try:
        cyp_parser = get_parser()
        
        all_results = []
        pk_curves = []
        
        # Generate Cohort
        total_n = req.population_size
        n_per_pop = max(1, total_n // max(1, len(req.ethnicities)))
        
        for ethnicity in req.ethnicities:
            # 1. Get Genotypes
            cohort_df = cyp_parser.simulate_population(ethnicity, n_per_pop)
            
            # 2. Add Phenotypes (Gender, Weight)
            # Gender: 0=F, 1=M
            cohort_df['gender'] = np.random.choice(
                [1, 0], 
                size=len(cohort_df), 
                p=[req.male_ratio/100, 1 - req.male_ratio/100]
            )
            # Weight: Normal distribution (clip 40-150)
            weights = np.random.normal(req.weight_mean, req.weight_sd, len(cohort_df))
            cohort_df['weight'] = np.clip(weights, 40, 150)
            
            # 3. Simulate each patient
            for _, patient in cohort_df.iterrows():
                # Params
                d_params = DrugParams(
                    KD_nM=req.kd,
                    dose_mg=req.dose * patient['weight'], # mg/kg -> mg
                    charge=req.charge,
                    MW_kDa=req.mw
                )
                
                p_params = PatientParams(
                    weight_kg=patient['weight'],
                    gender=int(patient['gender']),
                    ethnicity=ethnicity,
                    cyp2d6_genotype=patient['diplotype']
                )
                
                t_params = TargetParams(
                    baseline_nM=req.t0,
                    halflife_hr=req.halflife
                )
                
                # Engine
                engine = mPBPKEngine(d_params, p_params, t_params)
                res = engine.simulate()
                
                # Store core metrics
                all_results.append({
                    'ethnicity': ethnicity,
                    'gender': 'Male' if patient['gender']==1 else 'Female',
                    'weight': patient['weight'],
                    'TO_trough': res['TO_trough'],
                    'C_max': res['C_max'],
                    'AUC': res['AUC']
                })
                
                # Store sample PK curve (1 per ethnicity)
                if len(pk_curves) < len(req.ethnicities) * 2: # Keep few curves
                    pk_curves.append({
                        'time': res['time'].tolist(),
                        'concentration': res['C_plasma'].tolist(),
                        'TO': res['TO_percent'].tolist(),
                        'ethnicity': ethnicity
                    })

        # Aggregation
        df = pd.DataFrame(all_results)
        
        summary = {
            'toTrough': float(df['TO_trough'].mean()),
            'cMax': float(df['C_max'].mean()),
            'auc': float(df['AUC'].mean()),
            'successRate': float((df['TO_trough'] >= 90).mean() * 100),
            'toxicityRisk': float(((df['C_max'] > 2000) | (df['AUC'] > 20000)).mean() * 100)
        }
        
        cohort_breakdown = []
        for eth, group in df.groupby('ethnicity'):
            cohort_breakdown.append({
                'ethnicity': eth,
                'n': len(group),
                'toMean': float(group['TO_trough'].mean()),
                'passPct': float((group['TO_trough'] >= 90).mean() * 100)
            })
            
        # Format PK Data for Chart
        if pk_curves:
            representative = pk_curves[0]
            step = 10
            chart_data = []
            for i in range(0, len(representative['time']), step):
                chart_data.append({
                    'time': round(representative['time'][i], 1),
                    'concentration': round(representative['concentration'][i], 1),
                    'TO': round(representative['TO'][i], 1)
                })
        else:
            chart_data = []

        return {
            'summary': summary,
            'cohortBreakdown': cohort_breakdown,
            'pkData': chart_data
        }

    except Exception as e:
        import traceback
        error_msg = str(e)
        with open("d:\\KTaivle\\Big_Project\\backend_debug.log", "a") as f:
            f.write(f"\n[ERROR] {error_msg}\n")
            traceback.print_exc(file=f)
        return {"error": error_msg}
