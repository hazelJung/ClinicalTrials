"""
CYP2D6 Monte Carlo Simulator
============================
Simulates population PK variability based on CYP2D6 genotypes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

# Updated import for pksmart structure
try:
    from pksmart.parse_cyp2d6 import PharmVarCYP2D6Parser
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from parse_cyp2d6 import PharmVarCYP2D6Parser

# Import Type definitions from pksmart.types
try:
    from pksmart.types import MonteCarloResult
except ImportError:
    # Minimal definition if types.py not found during direct run
    @dataclass
    class MonteCarloResult:
        cmax_mean: float
        cmax_std: float
        cmax_p95: float
        sm_mean: float
        sm_p5: float
        pct_high_exposure: float
        phenotypes: Dict[str, float]
        
        def to_dict(self) -> Dict:
            return {
                "cmax_mean": self.cmax_mean,
                "sm_mean": self.sm_mean,
                "pct_high_exposure": self.pct_high_exposure
            }

@dataclass
class PopulationRisk:
    """Population Risk Assessment"""
    population: str
    high_risk_ratio: float
    mean_margin: float
    recommendation: str

class MonteCarloSimulator:
    """
    Simulates PK parameters (Cmax) across different CYP2D6 phenotypes.
    """
    
    # Clearance multipliers based on Activity Score (AS)
    # Ref: PharmGKB / CPIC
    CL_MULTIPLIERS = {
        'PM': 0.3,   # Poor Metabolizer: 30% Clearance
        'IM': 0.6,   # Intermediate: 60% Clearance
        'NM': 1.0,   # Normal: 100%
        'UM': 1.5    # Ultrarapid: 150% Clearance
    }
    
    def __init__(self):
        self.cyp2d6_parser = PharmVarCYP2D6Parser()
        self.cyp2d6_enabled = True
        
    def _get_fallback_cohort(self, population: str, n: int) -> pd.DataFrame:
        """Generate dummy cohort if parser fails."""
        return pd.DataFrame({
            'subject_id': range(n),
            'phenotype': ['NM'] * n,
            'activity_score': [2.0] * n
        })

    def simulate(self, 
                 base_cmax: float, 
                 population: str = 'EUR',
                 n: int = 1000,
                 ic50_uM: Optional[float] = None) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.
        
        Args:
            base_cmax: Predicted Cmax for a Normal Metabolizer (NM)
            population: Target population
            n: Number of subjects
            ic50_uM: IC50 for Safety Margin calculation
            
        Returns:
            MonteCarloResult
        """
        # 1. Generate Virtual Cohort
        try:
            cohort = self.cyp2d6_parser.simulate_population(population, n)
        except Exception:
            cohort = self._get_fallback_cohort(population, n)
            
        # 2. Assign PK Variability based on Phenotype
        # Cmax is approximately inversely proportional to Clearance for oral drugs (AUC = Dose/CL)
        # But for Cmax specifically, it depends on Ka and CL.
        # Simplified assumption: Cmax_indiv = Base_Cmax / (CL_mult^0.5) 
        # (Assuming Vd is constant, and CL change affects half-life and steady state accumulation)
        # OR: Cmax(steady_state) ~ 1/Cl. Let's use 1/CL_mult for Exposure (AUC) proxy.
        # Let's assume we simulate Exposure (AUC/Cmax) scaling.
        
        cohort['cl_mult'] = cohort['phenotype'].map(self.CL_MULTIPLIERS).fillna(1.0)
        
        # Add random intra-individual variability (CV 30%)
        # Log-normal distribution
        variability = np.random.lognormal(mean=0, sigma=0.3, size=n)
        
        # Scaling factor: Lower CL -> Higher Cmax
        # exposure_factor = (1.0 / cohort['cl_mult']) * variability
        cohort['exposure_factor'] = (1.0 / cohort['cl_mult']) * variability
        
        cohort['sim_cmax'] = base_cmax * cohort['exposure_factor']
        
        # 3. Calculate Safety Margins
        if ic50_uM:
            cohort['safety_margin'] = ic50_uM / cohort['sim_cmax']
        else:
            cohort['safety_margin'] = np.inf
            
        # 4. Statistics
        cmax_mean = float(cohort['sim_cmax'].mean())
        cmax_std = float(cohort['sim_cmax'].std())
        cmax_p95 = float(np.percentile(cohort['sim_cmax'], 95))
        
        sm_mean = float(cohort['safety_margin'].replace([np.inf, -np.inf], np.nan).mean())
        if np.isnan(sm_mean): sm_mean = 999.0
            
        sm_p5 = float(np.percentile(cohort['safety_margin'].replace([np.inf, -np.inf], 999.0), 5))
        
        # Count high exposure (> 2x base)
        high_exp_count = (cohort['sim_cmax'] > (base_cmax * 2)).sum()
        pct_high_exposure = float(high_exp_count / n * 100)
        
        phenotype_counts = cohort['phenotype'].value_counts(normalize=True).to_dict()
        
        return MonteCarloResult(
            cmax_mean=cmax_mean,
            cmax_std=cmax_std,
            cmax_p95=cmax_p95,
            sm_mean=sm_mean,
            sm_p5=sm_p5,
            pct_high_exposure=pct_high_exposure,
            phenotypes=phenotype_counts
        )

    def analyze_population_risk(self, base_cmax: float, ic50_uM: float, n: int = 2000) -> Dict[str, PopulationRisk]:
        """Analyze risks across all supported populations."""
        results = {}
        supported_pops = ['EUR', 'EAS', 'AFR', 'AMR', 'SAS']
        
        for pop in supported_pops:
            res = self.simulate(base_cmax, pop, n, ic50_uM)
            
            # Logic for risk recommendation
            rec = "Low Risk"
            risk_ratio = res.pct_high_exposure / 100.0
            
            if res.sm_p5 < 10:
                rec = "Exclude PM / Dose Adjustment"
            elif risk_ratio > 0.1:
                rec = "Monitor High Risk"
                
            results[pop] = PopulationRisk(
                population=pop,
                high_risk_ratio=risk_ratio,
                mean_margin=res.sm_mean,
                recommendation=rec
            )
            
        return results

if __name__ == "__main__":
    sim = MonteCarloSimulator()
    res = sim.simulate(base_cmax=1.0, population="EUR", n=100)
    print("Mean Cmax:", res.cmax_mean)
    print("Phenotypes:", res.phenotypes)
