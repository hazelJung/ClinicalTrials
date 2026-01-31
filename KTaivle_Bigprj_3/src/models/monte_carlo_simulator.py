"""
Monte Carlo Simulator
=====================
Step 6: CYP2D6 약물유전체 기반 Monte Carlo 시뮬레이션

기존 parse_cyp2d6.py의 PharmVarCYP2D6Parser를 활용하여
1,000명 가상 코호트의 PK 변이를 시뮬레이션합니다.

Features:
- CYP2D6 유전형 기반 clearance 스케일링
- 인구 집단별 표현형 분포 반영
- Cmax, AUC 분포 시뮬레이션
- Safety Margin 분포 분석

Author: AI-Driven Clinical Trial Platform
Date: 2026-01-26
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# CYP2D6 모듈 import
try:
    from src.data.parse_cyp2d6 import PharmVarCYP2D6Parser

    CYP2D6_AVAILABLE = True
except ImportError:
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data.parse_cyp2d6 import PharmVarCYP2D6Parser

        CYP2D6_AVAILABLE = True
    except ImportError:
        CYP2D6_AVAILABLE = False


@dataclass
class MonteCarloResult:
    """Monte Carlo 시뮬레이션 결과"""

    # 기본 정보
    n_subjects: int
    population: str
    base_cmax_uM: float

    # Cmax 분포
    cmax_mean: float
    cmax_std: float
    cmax_median: float
    cmax_p5: float  # 5th percentile
    cmax_p95: float  # 95th percentile
    cmax_min: float
    cmax_max: float

    # Safety Margin 분포 (IC50 제공 시)
    sm_mean: Optional[float] = None
    sm_p5: Optional[float] = None
    sm_p95: Optional[float] = None

    # 표현형 분포
    phenotype_distribution: Dict[str, float] = field(default_factory=dict)

    # 위험 환자 비율
    pct_high_exposure: float = 0.0  # Cmax > 2x baseline
    pct_low_exposure: float = 0.0  # Cmax < 0.5x baseline

    # 개별 결과 (선택적)
    individual_results: Optional[pd.DataFrame] = None

    def to_dict(self) -> Dict:
        return {
            "n_subjects": self.n_subjects,
            "population": self.population,
            "base_cmax_uM": self.base_cmax_uM,
            "cmax_distribution": {
                "mean": self.cmax_mean,
                "std": self.cmax_std,
                "median": self.cmax_median,
                "P5": self.cmax_p5,
                "P95": self.cmax_p95,
                "min": self.cmax_min,
                "max": self.cmax_max,
            },
            "safety_margin_distribution": {
                "mean": self.sm_mean,
                "P5": self.sm_p5,
                "P95": self.sm_p95,
            }
            if self.sm_mean is not None
            else None,
            "phenotype_distribution": self.phenotype_distribution,
            "risk_analysis": {
                "pct_high_exposure": self.pct_high_exposure,
                "pct_low_exposure": self.pct_low_exposure,
            },
        }


@dataclass
class PopulationRisk:
    """인구 집단별 위험도 분석"""

    population: str
    pm_frequency: float  # Poor Metabolizer 빈도
    um_frequency: float  # Ultra-rapid Metabolizer 빈도
    high_risk_ratio: float  # 고위험 환자 비율
    recommendation: str  # 임상 권고


class MonteCarloSimulator:
    """
    CYP2D6 기반 Monte Carlo 시뮬레이터

    가상 환자 코호트를 생성하고 PK 변이를 시뮬레이션합니다.
    """

    # 지원 인구 집단
    POPULATIONS = ["EUR", "EAS", "AFR", "AMR", "SAS"]
    POPULATION_NAMES = {
        "EUR": "European",
        "EAS": "East Asian",
        "AFR": "African",
        "AMR": "Admixed American",
        "SAS": "South Asian",
    }

    def __init__(self, random_seed: int = 42):
        """
        Initialize Monte Carlo Simulator.

        Args:
            random_seed: 랜덤 시드
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # CYP2D6 파서 초기화
        if CYP2D6_AVAILABLE:
            try:
                self.cyp2d6_parser = PharmVarCYP2D6Parser()
                self.cyp2d6_enabled = True
            except Exception as e:
                print(f"Warning: CYP2D6 parser initialization failed: {e}")
                self.cyp2d6_enabled = False
        else:
            self.cyp2d6_enabled = False
            print("Warning: CYP2D6 module not available. Using fallback distributions.")

    def _get_fallback_cohort(self, population: str, n_subjects: int) -> pd.DataFrame:
        """
        CYP2D6 모듈 없을 때 fallback 코호트 생성.

        기본 표현형 분포 사용 (CPIC 기준)
        """
        # 기본 표현형 분포 (EUR 기준)
        phenotype_probs = {
            "EUR": {"PM": 0.07, "IM": 0.10, "NM": 0.80, "UM": 0.03},
            "EAS": {"PM": 0.01, "IM": 0.45, "NM": 0.53, "UM": 0.01},
            "AFR": {"PM": 0.03, "IM": 0.25, "NM": 0.65, "UM": 0.07},
            "AMR": {"PM": 0.05, "IM": 0.15, "NM": 0.77, "UM": 0.03},
            "SAS": {"PM": 0.04, "IM": 0.18, "NM": 0.75, "UM": 0.03},
        }

        # CL multiplier 매핑
        cl_multipliers = {
            "PM": 0.2,  # 20% of normal CL
            "IM": 0.5,  # 50% of normal CL
            "NM": 1.0,  # Normal
            "UM": 1.5,  # 150% of normal CL
        }

        probs = phenotype_probs.get(population, phenotype_probs["EUR"])
        phenotypes = list(probs.keys())
        weights = list(probs.values())

        subjects = []
        for i in range(n_subjects):
            phenotype = np.random.choice(phenotypes, p=weights)
            cl_mult = cl_multipliers[phenotype]

            subjects.append(
                {
                    "subject_id": i + 1,
                    "population": population,
                    "phenotype": phenotype,
                    "clearance_multiplier": cl_mult,
                    "activity_score": {"PM": 0, "IM": 0.5, "NM": 1.5, "UM": 3.0}[
                        phenotype
                    ],
                }
            )

        return pd.DataFrame(subjects)

    def simulate(
        self,
        base_cmax: float,
        population: str = "EUR",
        n: int = 1000,
        ic50_uM: Optional[float] = None,
        return_individual: bool = False,
    ) -> MonteCarloResult:
        """
        CYP2D6 기반 Monte Carlo 시뮬레이션.

        Cmax는 clearance에 반비례:
        Cmax_adjusted = base_cmax / clearance_multiplier

        Args:
            base_cmax: 기준 Cmax (NM 기준, μM)
            population: 인구 코드 (EUR, EAS, AFR, AMR, SAS)
            n: 시뮬레이션 횟수 (피험자 수)
            ic50_uM: IC50 값 (Safety Margin 계산용)
            return_individual: 개별 결과 반환 여부

        Returns:
            MonteCarloResult
        """
        # 코호트 생성
        if self.cyp2d6_enabled:
            cohort = self.cyp2d6_parser.simulate_population(population, n)
        else:
            cohort = self._get_fallback_cohort(population, n)

        # Cmax 계산 (CL이 낮으면 Cmax 증가)
        cmax_values = []
        for _, patient in cohort.iterrows():
            cl_mult = patient["clearance_multiplier"]
            # Cmax ∝ 1/CL
            adjusted_cmax = base_cmax / cl_mult if cl_mult > 0 else base_cmax * 5
            cmax_values.append(adjusted_cmax)

        cohort["adjusted_cmax"] = cmax_values

        # Safety Margin 계산
        if ic50_uM is not None:
            cohort["safety_margin"] = ic50_uM / cohort["adjusted_cmax"]

        # 표현형 분포 계산
        phenotype_counts = cohort["phenotype"].value_counts(normalize=True)
        phenotype_dist = phenotype_counts.to_dict()

        # 위험 환자 비율
        pct_high = (cohort["adjusted_cmax"] > base_cmax * 2).mean() * 100
        pct_low = (cohort["adjusted_cmax"] < base_cmax * 0.5).mean() * 100

        # 결과 생성
        result = MonteCarloResult(
            n_subjects=n,
            population=population,
            base_cmax_uM=base_cmax,
            cmax_mean=np.mean(cmax_values),
            cmax_std=np.std(cmax_values),
            cmax_median=np.median(cmax_values),
            cmax_p5=np.percentile(cmax_values, 5),
            cmax_p95=np.percentile(cmax_values, 95),
            cmax_min=np.min(cmax_values),
            cmax_max=np.max(cmax_values),
            phenotype_distribution=phenotype_dist,
            pct_high_exposure=pct_high,
            pct_low_exposure=pct_low,
        )

        # Safety Margin 분포
        if ic50_uM is not None:
            sm_values = cohort["safety_margin"].values
            result.sm_mean = np.mean(sm_values)
            result.sm_p5 = np.percentile(sm_values, 5)
            result.sm_p95 = np.percentile(sm_values, 95)

        # 개별 결과
        if return_individual:
            result.individual_results = cohort

        return result

    def simulate_all_populations(
        self, base_cmax: float, n: int = 1000, ic50_uM: Optional[float] = None
    ) -> Dict[str, MonteCarloResult]:
        """
        모든 인구 집단에 대해 시뮬레이션.

        Args:
            base_cmax: 기준 Cmax (μM)
            n: 각 집단당 피험자 수
            ic50_uM: IC50 값

        Returns:
            인구별 MonteCarloResult 딕셔너리
        """
        results = {}
        for pop in self.POPULATIONS:
            results[pop] = self.simulate(base_cmax, pop, n, ic50_uM)
        return results

    def analyze_population_risk(
        self, base_cmax: float, ic50_uM: float, n: int = 5000
    ) -> Dict[str, PopulationRisk]:
        """
        인구 집단별 위험도 분석.

        Args:
            base_cmax: 기준 Cmax (μM)
            ic50_uM: IC50 값 (μM)
            n: 시뮬레이션 횟수

        Returns:
            인구별 PopulationRisk 딕셔너리
        """
        risks = {}

        for pop in self.POPULATIONS:
            result = self.simulate(base_cmax, pop, n, ic50_uM)

            pm_freq = result.phenotype_distribution.get("PM", 0)
            um_freq = result.phenotype_distribution.get("UM", 0)

            # SM < 10 인 비율 (고위험)
            if result.individual_results is not None:
                high_risk = (result.individual_results["safety_margin"] < 10).mean()
            else:
                # 재시뮬레이션
                sim_result = self.simulate(
                    base_cmax, pop, n, ic50_uM, return_individual=True
                )
                high_risk = (sim_result.individual_results["safety_margin"] < 10).mean()

            # 권고 사항 결정
            if high_risk > 0.10:
                recommendation = "High risk population. Consider dose reduction or alternative therapy."
            elif high_risk > 0.05:
                recommendation = (
                    "Moderate risk. Genotype screening recommended before treatment."
                )
            elif pm_freq > 0.05:
                recommendation = "Higher PM frequency. Consider CYP2D6 genotyping."
            else:
                recommendation = "Standard dosing appropriate for most patients."

            risks[pop] = PopulationRisk(
                population=self.POPULATION_NAMES.get(pop, pop),
                pm_frequency=pm_freq,
                um_frequency=um_freq,
                high_risk_ratio=high_risk,
                recommendation=recommendation,
            )

        return risks

    def get_dose_adjustment(
        self, phenotype: str, base_dose: float
    ) -> Tuple[float, str]:
        """
        표현형에 따른 용량 조절 권고.

        Args:
            phenotype: CYP2D6 표현형 (PM, IM, NM, UM)
            base_dose: 기준 용량 (mg)

        Returns:
            (조절된 용량, 권고 사항)
        """
        adjustments = {
            "PM": (0.5, "Reduce dose by 50% or consider alternative drug."),
            "IM": (0.75, "Consider 25% dose reduction."),
            "NM": (1.0, "Standard dose appropriate."),
            "UM": (
                1.5,
                "Standard dose may be inadequate. Consider efficacy monitoring.",
            ),
        }

        factor, recommendation = adjustments.get(phenotype, (1.0, "Unknown phenotype"))
        adjusted_dose = base_dose * factor

        return adjusted_dose, recommendation


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_monte_carlo(
    base_cmax: float,
    population: str = "EUR",
    n: int = 1000,
    ic50_uM: Optional[float] = None,
) -> Dict:
    """
    빠른 Monte Carlo 시뮬레이션.

    Args:
        base_cmax: 기준 Cmax (μM)
        population: 인구 코드
        n: 피험자 수
        ic50_uM: IC50 값

    Returns:
        결과 딕셔너리
    """
    simulator = MonteCarloSimulator()
    result = simulator.simulate(base_cmax, population, n, ic50_uM)
    return result.to_dict()


def get_cyp2d6_risk_summary(base_cmax: float, ic50_uM: float) -> Dict:
    """
    CYP2D6 관련 위험 요약.

    Args:
        base_cmax: 기준 Cmax
        ic50_uM: IC50 값

    Returns:
        위험 요약 딕셔너리
    """
    simulator = MonteCarloSimulator()
    risks = simulator.analyze_population_risk(base_cmax, ic50_uM)

    return {
        pop: {
            "population_name": risk.population,
            "pm_frequency": f"{risk.pm_frequency * 100:.1f}%",
            "high_risk_ratio": f"{risk.high_risk_ratio * 100:.1f}%",
            "recommendation": risk.recommendation,
        }
        for pop, risk in risks.items()
    }


# =============================================================================
# Main (Test)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Monte Carlo Simulator Test")
    print("=" * 60)

    simulator = MonteCarloSimulator()

    # Test parameters
    base_cmax = 2.0  # μM
    ic50 = 50.0  # μM

    print(f"\n>>> Test Parameters:")
    print(f"    Base Cmax: {base_cmax} μM")
    print(f"    hERG IC50: {ic50} μM")
    print(
        f"    CYP2D6 Module: {'Available' if simulator.cyp2d6_enabled else 'Fallback mode'}"
    )

    # Simulate for EUR population
    print(f"\n>>> Simulating EUR population (n=1000)...")
    result = simulator.simulate(
        base_cmax=base_cmax,
        population="EUR",
        n=1000,
        ic50_uM=ic50,
        return_individual=True,
    )

    print(f"\n>>> Results:")
    print(f"    Cmax Distribution:")
    print(f"      Mean: {result.cmax_mean:.2f} μM")
    print(f"      Median: {result.cmax_median:.2f} μM")
    print(f"      P5-P95: {result.cmax_p5:.2f} - {result.cmax_p95:.2f} μM")

    print(f"\n    Safety Margin Distribution:")
    print(f"      Mean: {result.sm_mean:.1f}")
    print(f"      P5 (worst): {result.sm_p5:.1f}")
    print(f"      P95: {result.sm_p95:.1f}")

    print(f"\n    Phenotype Distribution:")
    for pheno, freq in result.phenotype_distribution.items():
        print(f"      {pheno}: {freq * 100:.1f}%")

    print(f"\n    Risk Analysis:")
    print(f"      High exposure (>2x baseline): {result.pct_high_exposure:.1f}%")
    print(f"      Low exposure (<0.5x baseline): {result.pct_low_exposure:.1f}%")

    # Compare populations
    print("\n" + "-" * 60)
    print(">>> Population Comparison:")

    for pop in ["EUR", "EAS", "AFR"]:
        pop_result = simulator.simulate(base_cmax, pop, 1000, ic50)
        pm_pct = pop_result.phenotype_distribution.get("PM", 0) * 100
        print(
            f"    {pop}: PM={pm_pct:.1f}%, Cmax P95={pop_result.cmax_p95:.2f} μM, SM P5={pop_result.sm_p5:.1f}"
        )

    print("\n" + "=" * 60)
    print("  Monte Carlo Simulator Test Complete!")
    print("=" * 60)
