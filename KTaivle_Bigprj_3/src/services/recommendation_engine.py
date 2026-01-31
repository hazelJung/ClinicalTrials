"""
Recommendation Engine
=====================
Phase 5: 추천 엔진 구현

DrugSafetyService의 결과를 바탕으로
1. PoS (Probability of Success) 계산
2. 최적 환자군 추천 (CYP2D6 표현형 기반)
3. 고위험군 자동 플래그

Author: AI-Driven Clinical Trial Platform
Date: 2026-01-27
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import warnings

# Import dependencies
try:
    from src.services.drug_safety_service import DrugSafetyResult, OverallRisk
    from src.models.monte_carlo_simulator import MonteCarloSimulator, PopulationRisk
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.services.drug_safety_service import DrugSafetyResult, OverallRisk
    from src.models.monte_carlo_simulator import MonteCarloSimulator, PopulationRisk

warnings.filterwarnings("ignore")


class PoSCategory(Enum):
    HIGH = "High PoS (>80%)"
    MEDIUM = "Medium PoS (50-80%)"
    LOW = "Low PoS (<50%)"


@dataclass
class RecommendationResult:
    """추천 엔진 결과"""

    drug_name: str

    # 1. PoS (성공 확률)
    pos_probability: float  # 0.0 - 1.0
    pos_category: PoSCategory

    # 2. 인구 집단 분석
    population_risks: Dict[str, PopulationRisk]

    # 3. 최적 환자군 전략
    target_patient_group: (
        str  # e.g., "All Comers", "Exclude PM", "Genotype Screen Required"
    )
    excluded_groups: List[str]  # e.g., ["CYP2D6 PM"]

    # 4. 개발 전략 권고
    development_strategy: str

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  Clinical Recommendation: {self.drug_name}",
            "=" * 60,
            f"  PoS (Probability of Success): {self.pos_probability * 100:.1f}%",
            f"  Category: {self.pos_category.value}",
            "",
            f"  >>> Target Patient Group: {self.target_patient_group}",
        ]

        if self.excluded_groups:
            lines.append(f"  >>> Excluded Groups: {', '.join(self.excluded_groups)}")

        lines.append("")
        lines.append("  Development Strategy:")
        lines.append(f"    {self.development_strategy}")

        lines.append("")
        lines.append("  Population Risk Analysis:")
        for pop, risk in self.population_risks.items():
            lines.append(
                f"    - {pop}: {risk.recommendation} (High Risk Ratio: {risk.high_risk_ratio * 100:.1f}%)"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


class RecommendationEngine:
    """
    임상 성공 확률 예측 및 환자군 추천 엔진
    """

    def __init__(self):
        self.mc_simulator = MonteCarloSimulator()

    def analyze(self, safety_result: DrugSafetyResult) -> RecommendationResult:
        """
        DrugSafetyResult를 분석하여 추천 결과 생성
        """
        # 1. PoS 계산
        pos = self._calculate_pos(safety_result)

        # 2. 인구 집단 분석 (Monte Carlo)
        # 필요한 정보 추출: Cmax, IC50
        base_cmax = 0.0
        ic50 = 0.0

        if safety_result.ivive_result:
            base_cmax = safety_result.ivive_result.Cmax_uM

        if safety_result.safety_result:
            ic50 = safety_result.safety_result.IC50_uM

        # Monte Carlo 실행 (이미 결과에 있을 수 있지만, 전체 인구 분석을 위해 재실행 가능)
        # DrugSafetyService는 보통 지정된 1개 인구만 돌림. 여기서 전체 분석.
        population_risks = {}
        if base_cmax > 0 and ic50 > 0:
            population_risks = self.mc_simulator.analyze_population_risk(
                base_cmax=base_cmax,
                ic50_uM=ic50,
                n=2000,  # 충분한 샘플 수
            )

        # 3. 최적 환자군 및 전략 도출
        target_group, excluded, strategy = self._derive_strategy(
            pos, population_risks, safety_result
        )

        return RecommendationResult(
            drug_name=safety_result.drug_name,
            pos_probability=pos,
            pos_category=self._get_pos_category(pos),
            population_risks=population_risks,
            target_patient_group=target_group,
            excluded_groups=excluded,
            development_strategy=strategy,
        )

    def _calculate_pos(self, result: DrugSafetyResult) -> float:
        """
        Probability of Success (Clinical Safety) 계산

        Factors:
        1. Safety Score (from DrugSafetyService): 50%
        2. Toxicophore Alerts: 20% (No alerts -> higher PoS)
        3. QSAR Consensus: 30% (No toxicity -> higher PoS)
        """
        # Base: Safety Score (already 0-100)
        base_score = result.overall_score

        # Penalties/Bonuses
        # QSAR Consistency
        qsar_penalty = 0.0
        if result.qsar_predictions:
            toxic_count = sum(
                1 for p in result.qsar_predictions.values() if p.prediction == 1
            )
            total = len(result.qsar_predictions)
            if total > 0:
                toxic_ratio = toxic_count / total
                qsar_penalty = toxic_ratio * 30.0  # Max 30% penalty

        # Toxicophore Severity
        tox_penalty = 0.0
        if result.toxicophore_result:
            if result.toxicophore_result.risk_score > 50:
                tox_penalty = 20.0
            elif result.toxicophore_result.risk_score > 20:
                tox_penalty = 10.0

        # Final PoS Calculation
        # Simple weighted model:
        # Start with base score (which includes SM, Tox, QSAR weighted)
        # Let's map 0-100 score to probability 0.0-1.0 with sigmoid-like curve or linear

        # 선형 매핑 사용 (0-100 -> 0-1)
        # 단, Critical Alerts 있으면 급격히 낮춤

        final_score = base_score

        if result.overall_risk == OverallRisk.CRITICAL:
            final_score = min(final_score, 30.0)  # Max 30% for critical risk

        return max(0.0, min(1.0, final_score / 100.0))

    def _get_pos_category(self, pos: float) -> PoSCategory:
        if pos >= 0.8:
            return PoSCategory.HIGH
        elif pos >= 0.5:
            return PoSCategory.MEDIUM
        else:
            return PoSCategory.LOW

    def _derive_strategy(
        self, pos: float, pop_risks: Dict[str, PopulationRisk], result: DrugSafetyResult
    ) -> tuple:
        """
        전략 도출 로직
        """
        target_group = "All Comers"
        excluded = []
        strategy = ""

        # 1. Check Population Risks
        high_risk_pops = []
        for pop, risk in pop_risks.items():
            if risk.high_risk_ratio > 0.1:  # 10% 이상이 위험
                high_risk_pops.append(pop)

        # Check CYP2D6 PM specific risk (using EUR data usually or aggregate)
        # 만약 모든 인구에서 PM 빈도가 높은 인구(EUR)가 더 위험하다면

        # PM Risk Check
        pm_risky = False
        # Monte Carlo result in safety_result might specifically look at PM
        if result.monte_carlo_result and result.monte_carlo_result.sm_p5 is not None:
            if result.monte_carlo_result.sm_p5 < 10:  # PM-like individuals are at risk
                pm_risky = True

        if pm_risky:
            target_group = "Genotype Screened Population"
            excluded.append("CYP2D6 Poor Metabolizers (PM)")
            strategy += "Mandatory CYP2D6 genotyping required. Exclude PMs. "

        elif high_risk_pops:
            target_group = (
                f"Exclude High Risk Populations ({', '.join(high_risk_pops)})"
            )
            strategy += f"Geographic/Ethnic restrictions recommended. "

        # 2. PoS based strategy
        if pos >= 0.8:
            strategy += "Proceed to Phase 1. Fast-track eligible."
        elif pos >= 0.5:
            strategy += (
                "Proceed with caution. Enhanced monitoring for cardiac/hepatic events."
            )
        else:
            strategy = "STOP DEVELOPMENT. Structural optimization required to improve Safety Margin."
            target_group = "None"

        return target_group, excluded, strategy


# =============================================================================
# Main (Test)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Recommendation Engine Test")
    print("=" * 60)

    # 1. Create a mock DrugSafetyResult for testing
    from src.services.drug_safety_service import DrugSafetyService

    # Load Service
    print(">>> Initializing Service...")
    service = DrugSafetyService(enable_qsar=False)  # Speed up test

    # Test Case 1: Safe Drug (Acetaminophen-like)
    print("\n>>> Case 1: Safe Drug")
    res1 = service.evaluate(
        smiles="CC(=O)Nc1ccc(O)cc1",
        name="Safe-Drug-A",
        IC50_uM=200.0,
        CLint_uL_min_mg=10.0,
        include_monte_carlo=True,
    )

    engine = RecommendationEngine()
    rec1 = engine.analyze(res1)
    print(rec1.summary())

    # Test Case 2: Risk Drug (Low Margin, PM Risk)
    print("\n>>> Case 2: CYP2D6 Risk Drug")
    # Low IC50 (Risk) + High Clearance (Metabolism dependent)
    res2 = service.evaluate(
        smiles="CC(C)(C)c1ccc(cc1)C(O)CCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4",  # Terfenadine
        name="Risk-Drug-B",
        IC50_uM=2.0,  # Dangerous
        CLint_uL_min_mg=100.0,  # High metabolism -> PM will have huge Cmax
        include_monte_carlo=True,
    )

    rec2 = engine.analyze(res2)
    print(rec2.summary())
