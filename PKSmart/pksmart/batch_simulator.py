"""
Batch Simulator
===============
Generates training data for mPBPK-ML classifier.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Adjusted Import
try:
    from pksmart.mpbpk_engine import (
        DrugParams, PatientParams, TargetParams, PKConstants,
        mPBPKEngine, get_cyp2d6_parser
    )
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from mpbpk_engine import (
        DrugParams, PatientParams, TargetParams, PKConstants,
        mPBPKEngine, get_cyp2d6_parser
    )

@dataclass
class SimulationConfig:
    kd_range: Tuple[float, float] = (0.01, 20)
    dose_range: Tuple[float, float] = (0.5, 20)
    charges: List[int] = None
    mw_range: Tuple[float, float] = (140, 160)
    t0_range: Tuple[float, float] = (1, 100)
    halflife_range: Tuple[float, float] = (1, 1000)
    populations: List[str] = None
    patients_per_pop: int = 20
    
    def __post_init__(self):
        if self.charges is None: self.charges = [-5, 0, 5]
        if self.populations is None: self.populations = ['EUR', 'EAS', 'AFR', 'AMR', 'SAS']

class DrugCandidateGenerator:
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
    
    def _log_uniform(self, low, high, size=1):
        return 10 ** np.random.uniform(np.log10(low), np.log10(high), size)
        
    def generate(self, n=100) -> pd.DataFrame:
        candidates = []
        for i in range(n):
            candidates.append({
                'drug_id': f'DRUG_{i:04d}',
                'KD_nM': self._log_uniform(*self.config.kd_range)[0],
                'dose_mg_kg': self._log_uniform(*self.config.dose_range)[0],
                'charge': np.random.choice(self.config.charges),
                'MW_kDa': np.random.uniform(*self.config.mw_range),
                'T0_nM': self._log_uniform(*self.config.t0_range)[0],
                'halflife_hr': self._log_uniform(*self.config.halflife_range)[0]
            })
        return pd.DataFrame(candidates)

class SimulationMatrix:
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.drug_generator = DrugCandidateGenerator(config)
        self.cyp2d6_parser = get_cyp2d6_parser()
        
    def run(self, n_drugs=100, n_patients_per_pop=None, verbose=True):
        if n_patients_per_pop is None: n_patients_per_pop = self.config.patients_per_pop
        
        drugs_df = self.drug_generator.generate(n_drugs)
        if verbose: print(f"Generated {len(drugs_df)} drugs.")
        
        pk_const = PKConstants()
        results = []
        
        for idx, drug_row in drugs_df.iterrows():
            if verbose and idx % 10 == 0: print(f"Processing drug {idx}...")
            
            for pop in self.config.populations:
                if self.cyp2d6_parser:
                    cohort = self.cyp2d6_parser.simulate_population(pop, n_patients_per_pop)
                else:
                    cohort = pd.DataFrame({'diplotype': ['*1/*1']*n_patients_per_pop, 'phenotype': ['NM']*n_patients_per_pop})
                    
                for _, p_row in cohort.iterrows():
                    drug = DrugParams(
                        KD_nM=drug_row['KD_nM'],
                        dose_mg=drug_row['dose_mg_kg']*70,
                        charge=int(drug_row['charge']),
                        MW_kDa=drug_row['MW_kDa']
                    )
                    patient = PatientParams(
                         cyp2d6_genotype=p_row.get('diplotype', '*1/*1'),
                         ethnicity=pop
                    )
                    target = TargetParams(
                        baseline_nM=drug_row['T0_nM'],
                        halflife_hr=drug_row['halflife_hr']
                    )
                    
                    try:
                        eng = mPBPKEngine(drug, patient, target, pk_const)
                        res = eng.simulate()
                        
                        results.append({
                            'drug_id': drug_row['drug_id'],
                            'population': pop,
                            'phenotype': p_row.get('phenotype', 'NM'),
                            'success': res.get('success', False),
                            'TO_trough': res.get('TO_trough', 0)
                        })
                    except Exception:
                        continue
                        
        return pd.DataFrame(results)

if __name__ == "__main__":
    sim = SimulationMatrix()
    df = sim.run(n_drugs=1, n_patients_per_pop=2)
    print(df.head())
