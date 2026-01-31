"""
Recommendation Engine
=====================
Phase 5: Recommendation Engine Implementation
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum
import warnings

# Adjusted imports to use pksmart.types
try:
    from pksmart.types import DrugSafetyResult, OverallRisk, PoSCategory
    from pksmart.monte_carlo_simulator import MonteCarloSimulator, PopulationRisk
    from pksmart.types import RecommendationResult # If added to types, otherwise define here
except ImportError:
    # Local fallback
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from types import DrugSafetyResult, OverallRisk
    from monte_carlo_simulator import MonteCarloSimulator, PopulationRisk

warnings.filterwarnings("ignore")

class PoSCategory(Enum):
    HIGH = "High PoS (>80%)"
    MEDIUM = "Medium PoS (50-80%)"
    LOW = "Low PoS (<50%)"

@dataclass
class RecommendationResult:
    drug_name: str
    pos_probability: float
    pos_category: PoSCategory
    population_risks: Dict[str, PopulationRisk]
    target_patient_group: str
    excluded_groups: List[str]
    development_strategy: str

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  Clinical Recommendation: {self.drug_name}",
            "=" * 60,
            f"  PoS: {self.pos_probability * 100:.1f}% ({self.pos_category.value})",
            f"  Target Group: {self.target_patient_group}",
        ]
        if self.excluded_groups:
            lines.append(f"  Excluded: {', '.join(self.excluded_groups)}")
        lines.append(f"  Strategy: {self.development_strategy}")
        return "\n".join(lines)

class RecommendationEngine:
    def __init__(self):
        self.mc_simulator = MonteCarloSimulator()

    def analyze(self, safety_result: DrugSafetyResult) -> RecommendationResult:
        pos = self._calculate_pos(safety_result)
        
        base_cmax = 0.0
        ic50 = 0.0
        
        if safety_result.ivive_result:
            base_cmax = safety_result.ivive_result.Cmax_uM
        if safety_result.safety_result:
            ic50 = safety_result.safety_result.IC50_uM
            
        population_risks = {}
        if base_cmax > 0 and ic50 > 0:
            population_risks = self.mc_simulator.analyze_population_risk(base_cmax, ic50)
            
        target_group, excluded, strategy = self._derive_strategy(pos, population_risks, safety_result)
        
        return RecommendationResult(
            drug_name=safety_result.drug_name,
            pos_probability=pos,
            pos_category=self._get_pos_category(pos),
            population_risks=population_risks,
            target_patient_group=target_group,
            excluded_groups=excluded,
            development_strategy=strategy
        )

    def _calculate_pos(self, result: DrugSafetyResult) -> float:
        score = result.overall_score
        
        # QSAR Penalty
        if result.qsar_predictions:
            toxic_count = sum(1 for p in result.qsar_predictions.values() if p.prediction == 1)
            total = len(result.qsar_predictions)
            if total > 0:
                score -= (toxic_count / total) * 30.0
                
        # Tox Penalty
        if result.toxicophore_result and result.toxicophore_result.risk_score > 50:
            score -= 20.0
            
        final_score = max(0, min(100, score))
        if result.overall_risk == OverallRisk.CRITICAL:
            final_score = min(final_score, 30.0)
            
        return final_score / 100.0

    def _get_pos_category(self, pos: float) -> PoSCategory:
        if pos >= 0.8: return PoSCategory.HIGH
        elif pos >= 0.5: return PoSCategory.MEDIUM
        else: return PoSCategory.LOW

    def _derive_strategy(self, pos: float, pop_risks: Dict[str, PopulationRisk], result: DrugSafetyResult):
        target_group = "All Comers"
        excluded = []
        strategy = ""
        
        pm_risky = False
        if result.monte_carlo_result and result.monte_carlo_result.sm_p5 is not None:
             if result.monte_carlo_result.sm_p5 < 10:
                 pm_risky = True
                 
        if pm_risky:
            target_group = "Genotype Screened"
            excluded.append("CYP2D6 PM")
            strategy += "Genotyping required. "
        
        high_risk_pops = [p for p, r in pop_risks.items() if r.high_risk_ratio > 0.1]
        if high_risk_pops:
            strategy += f"Caution in {high_risk_pops}. "
            
        if pos >= 0.8: strategy += "Proceed to Phase 1."
        elif pos >= 0.5: strategy += "Proceed with caution."
        else: strategy = "STOP DEVELOPMENT."
        
        return target_group, excluded, strategy

if __name__ == "__main__":
    eng = RecommendationEngine()
    print("Recommendation Engine Initialized")
