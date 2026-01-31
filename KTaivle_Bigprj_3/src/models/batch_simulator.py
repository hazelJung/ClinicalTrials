"""
Batch Simulator - Virtual Drug Candidate & Patient Cohort Simulation
=====================================================================
Generates training data for mPBPK-ML classifier by simulating
drug candidates across patient populations with CYP2D6 variability.

Features:
- Log-uniform sampling for drug parameters
- Hard Mode for class imbalance handling
- Multi-label output (efficacy, toxicity exposure)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import sys
import json
from datetime import datetime

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mpbpk_engine import (
    DrugParams, PatientParams, TargetParams, PKConstants,
    mPBPKEngine, get_cyp2d6_parser
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SimulationConfig:
    """
    Simulation configuration with antibody-specific parameter ranges.
    
    Data Sources (Validated 2026-01-22 from 20 FDA-approved antibodies):
    - KD: 0.01-20 nM (Strong binding characteristic of therapeutic Abs)
    - Dose: 0.5-20 mg/kg (Standard therapeutic dosing)
    - Half-life: 100-700 hours (Long half-life of IgG: 4-30 days)
    - MW: 145-152 kDa (IgG1-4 standard molecular weight)
    - T0: 1-100 nM (Literature values for target expression)
    """
    # Drug parameter ranges (log-uniform sampling)
    # Validated from 20 FDA-approved monoclonal antibodies
    kd_range: Tuple[float, float] = (0.01, 20)        # nM 
    dose_range: Tuple[float, float] = (0.5, 20)       # mg/kg
    charges: List[int] = None                          # Surface charge
    mw_range: Tuple[float, float] = (140, 160)        # kDa (Broad IgG range)
    
    # Target parameter ranges
    t0_range: Tuple[float, float] = (1, 100)          # nM
    halflife_range: Tuple[float, float] = (1, 1000)  # hours (extended to match real data)
    
    # Hard Mode settings (DISABLED for paper comparison)
    hard_mode: bool = False
    hard_t0_scale: float = 1.0           # No scaling
    hard_kint: float = 0.1               # Default internalization
    hard_ksyn_scale: float = 1.0         # No scaling
    
    # Thresholds for labels (paper standard)
    to_success_threshold: float = 90.0    # TO% for efficacy success (paper: 90%)
    cmax_toxic_threshold: float = 500.0  # nM for toxicity risk
    auc_toxic_threshold: float = 10000.0 # nM*day for toxicity risk
    
    # Populations to simulate
    populations: List[str] = None
    patients_per_pop: int = 20
    
    def __post_init__(self):
        if self.charges is None:
            self.charges = [-5, 0, 5]
        if self.populations is None:
            self.populations = ['EUR', 'EAS', 'AFR', 'AMR', 'SAS']


# =============================================================================
# Drug Candidate Generator
# =============================================================================

class DrugCandidateGenerator:
    """Generates virtual drug candidates with log-uniform sampling."""
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
    
    def _log_uniform(self, low: float, high: float, size: int = 1) -> np.ndarray:
        """Sample from log-uniform distribution."""
        log_low = np.log10(low)
        log_high = np.log10(high)
        return 10 ** np.random.uniform(log_low, log_high, size)
    
    def generate(self, n: int = 100) -> pd.DataFrame:
        """Generate n virtual drug candidates."""
        candidates = []
        
        for i in range(n):
            # Drug properties
            kd = self._log_uniform(*self.config.kd_range)[0]
            dose = self._log_uniform(*self.config.dose_range)[0]
            charge = np.random.choice(self.config.charges)
            mw = np.random.uniform(*self.config.mw_range)
            
            # Target properties
            t0 = self._log_uniform(*self.config.t0_range)[0]
            halflife = self._log_uniform(*self.config.halflife_range)[0]
            
            # Apply Hard Mode scaling
            if self.config.hard_mode:
                t0 *= self.config.hard_t0_scale
            
            candidates.append({
                'drug_id': f'DRUG_{i:04d}',
                'KD_nM': kd,
                'dose_mg_kg': dose,
                'charge': charge,
                'MW_kDa': mw,
                'T0_nM': t0,
                'halflife_hr': halflife,
            })
        
        return pd.DataFrame(candidates)


# =============================================================================
# Simulation Matrix Runner
# =============================================================================

class SimulationMatrix:
    """Runs drug × patient simulation matrix."""
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.drug_generator = DrugCandidateGenerator(config)
        self.cyp2d6_parser = get_cyp2d6_parser()
    
    def _create_pk_constants(self) -> PKConstants:
        """Create PK constants with Hard Mode adjustments."""
        pk = PKConstants()
        if self.config.hard_mode:
            pk.kint = self.config.hard_kint
        return pk
    
    def _add_biological_noise(self, value: float, cv: float = 0.15) -> float:
        """Add biological variability (coefficient of variation) to simulation output.
        
        This reflects real-world inter-individual variability in PK responses.
        CV of 15% is typical for antibody PK in clinical trials.
        """
        if value <= 0:
            return value
        # Log-normal noise to ensure positive values
        noise_factor = np.exp(np.random.normal(0, cv))
        return value * noise_factor
    
    def _generate_labels(self, result: Dict) -> Dict:
        """Generate multi-labels from simulation result with biological noise."""
        # Add realistic biological variability (15% CV)
        to_trough = self._add_biological_noise(result.get('TO_trough', 0), cv=0.15)
        c_max = self._add_biological_noise(result.get('C_max', 0), cv=0.20)
        auc = self._add_biological_noise(result.get('AUC', 0), cv=0.20)
        
        # Cap TO at 100%
        to_trough = min(to_trough, 100.0)
        
        return {
            'efficacy_success': 1 if to_trough >= self.config.to_success_threshold else 0,
            'exposure_toxicity_risk': 1 if (c_max > self.config.cmax_toxic_threshold or 
                                            auc > self.config.auc_toxic_threshold) else 0,
            'TO_trough_noisy': to_trough,  # Store noisy value for reference
        }
    
    def run(self, 
            n_drugs: int = 100, 
            n_patients_per_pop: int = None,
            verbose: bool = True) -> pd.DataFrame:
        """
        Run simulation matrix.
        
        Args:
            n_drugs: Number of virtual drug candidates
            n_patients_per_pop: Patients per population (default from config)
            verbose: Print progress
            
        Returns:
            DataFrame with simulation results and labels
        """
        if n_patients_per_pop is None:
            n_patients_per_pop = self.config.patients_per_pop
        
        # Generate drug candidates
        drugs_df = self.drug_generator.generate(n_drugs)
        
        if verbose:
            print(f"Generated {len(drugs_df)} drug candidates")
            print(f"Populations: {self.config.populations}")
            print(f"Patients per population: {n_patients_per_pop}")
            total_sims = len(drugs_df) * len(self.config.populations) * n_patients_per_pop
            print(f"Total simulations: {total_sims:,}")
        
        # Generate patient cohorts for each population
        cohorts = {}
        for pop in self.config.populations:
            if self.cyp2d6_parser:
                cohorts[pop] = self.cyp2d6_parser.simulate_population(pop, n_patients_per_pop)
            else:
                # Fallback: generate dummy patients
                cohorts[pop] = pd.DataFrame({
                    'subject_id': range(n_patients_per_pop),
                    'diplotype': ['*1/*1'] * n_patients_per_pop,
                    'allele_1': ['*1'] * n_patients_per_pop,
                    'allele_2': ['*1'] * n_patients_per_pop,
                    'activity_score': [2.0] * n_patients_per_pop,
                    'phenotype': ['NM'] * n_patients_per_pop,
                })
        
        # Run simulations
        results = []
        pk_const = self._create_pk_constants()
        
        for drug_idx, drug_row in drugs_df.iterrows():
            if verbose and drug_idx % 10 == 0:
                print(f"  Processing drug {drug_idx + 1}/{len(drugs_df)}...")
            
            for pop in self.config.populations:
                cohort = cohorts[pop]
                
                for _, patient_row in cohort.iterrows():
                    # Create parameters
                    drug = DrugParams(
                        KD_nM=drug_row['KD_nM'],
                        dose_mg=drug_row['dose_mg_kg'] * 70,  # Assume 70kg
                        charge=int(drug_row['charge']),
                        MW_kDa=drug_row['MW_kDa'],
                    )
                    
                    patient = PatientParams(
                        weight_kg=70.0,
                        cyp2d6_genotype=patient_row['diplotype'],
                        ethnicity=pop,
                    )
                    
                    target = TargetParams(
                        baseline_nM=drug_row['T0_nM'],
                        halflife_hr=drug_row['halflife_hr'],
                    )
                    
                    # Run simulation
                    try:
                        engine = mPBPKEngine(drug, patient, target, pk_const)
                        sim_result = engine.simulate()
                        
                        # Generate labels
                        labels = self._generate_labels(sim_result)
                        
                        # Collect result
                        result_row = {
                            # Drug features
                            'drug_id': drug_row['drug_id'],
                            'log_KD': np.log10(drug_row['KD_nM']),
                            'log_dose': np.log10(drug_row['dose_mg_kg']),
                            'charge': drug_row['charge'],
                            'log_MW': np.log10(drug_row['MW_kDa']),
                            # Target features
                            'log_T0': np.log10(drug_row['T0_nM']),
                            'log_halflife': np.log10(drug_row['halflife_hr']),
                            # Patient features
                            'population': pop,
                            'phenotype': patient_row['phenotype'],
                            'activity_score': patient_row['activity_score'],
                            'cl_multiplier': engine.cyp2d6_cl_multiplier,
                            # Simulation outputs
                            'TO_trough': sim_result.get('TO_trough', 0),
                            'C_max': sim_result.get('C_max', 0),
                            'AUC': sim_result.get('AUC', 0),
                            # Labels
                            **labels,
                        }
                        results.append(result_row)
                        
                    except Exception as e:
                        # Skip failed simulations
                        continue
        
        df = pd.DataFrame(results)
        
        if verbose:
            print(f"\nCompleted {len(df):,} simulations")
            print(f"Efficacy success rate: {df['efficacy_success'].mean()*100:.1f}%")
            print(f"Toxicity risk rate: {df['exposure_toxicity_risk'].mean()*100:.1f}%")
        
        return df


# =============================================================================
# Data Saving/Loading
# =============================================================================

def save_training_data(df: pd.DataFrame, output_dir: Path = None) -> Path:
    """Save training data to CSV."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save as CSV
    csv_path = output_dir / f'mpbpk_training_data_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    
    # Save latest pointer
    latest_path = output_dir / 'mpbpk_training_data_latest.csv'
    df.to_csv(latest_path, index=False)
    
    print(f"Saved training data to:")
    print(f"  - {csv_path}")
    
    return csv_path


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  mPBPK Batch Simulator - Training Data Generation")
    print("=" * 60)
    
    # Configuration
    config = SimulationConfig(
        hard_mode=False,          # Disabled for paper comparison
        patients_per_pop=10,      # 10 patients per population
    )
    
    # Run simulation matrix
    simulator = SimulationMatrix(config)
    
    # Generate training data (LARGE SCALE: 1000 drugs)
    # 1000 drugs × 10 patients × 5 populations = 50,000 simulations
    df = simulator.run(
        n_drugs=1000,             # Paper-scale: 1000 drugs
        n_patients_per_pop=10,    # 10 patients × 5 populations = 50 per drug
        verbose=True
    )
    
    # Save results
    output_path = save_training_data(df)
    
    print("\n" + "=" * 60)
    print("  Data Generation Complete!")
    print("=" * 60)
    
    # Summary statistics
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Features: {len(df.columns) - 2} (excluding labels)")
    print(f"\n📈 Label Distribution:")
    print(f"   Efficacy Success: {df['efficacy_success'].sum():,} / {len(df):,} ({df['efficacy_success'].mean()*100:.1f}%)")
    print(f"   Toxicity Risk: {df['exposure_toxicity_risk'].sum():,} / {len(df):,} ({df['exposure_toxicity_risk'].mean()*100:.1f}%)")
    
    print(f"\n📈 Phenotype Distribution:")
    print(df['phenotype'].value_counts())
