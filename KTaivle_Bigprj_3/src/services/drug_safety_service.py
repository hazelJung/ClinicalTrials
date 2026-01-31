"""
Drug Safety Service - Unified API
==================================
Phase 4: 통합 약물 안전성 평가 서비스

모든 QSAR 모듈을 통합하여 단일 API로 약물 안전성을 평가합니다.

Pipeline:
1. QSAR Prediction (QSARPredictor) - 41개 기술자 기반 독성 예측
2. IVIVE Calculation (IVIVECalculator) - CLint → CLh, Cmax 예측
3. Safety Margin (SafetyMarginCalculator) - IC50/Cmax 계산
4. Toxicophore Analysis (ToxicophoreAnalyzer) - SMARTS 기반 스크리닝
5. AD Analysis (ApplicabilityDomain) - 예측 신뢰도 검증
6. Monte Carlo (MonteCarloSimulator) - CYP2D6 약물유전체 시뮬레이션

MW > 900 Da 인 경우 기존 mPBPK 엔진으로 분기합니다.

Author: AI-Driven Clinical Trial Platform
Date: 2026-01-26
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from enum import Enum
import warnings

warnings.filterwarnings("ignore")

# 모듈 imports
try:
    from src.models.ivive_calculator import IVIVECalculator, DrugInput, IVIVEResult
    from src.models.safety_calculator import (
        SafetyMarginCalculator,
        SafetyResult,
        RiskLevel,
    )
    from src.models.toxicophore_analyzer import ToxicophoreAnalyzer, ToxicophoreResult
    from src.models.qsar_predictor import QSARPredictor, QSARPrediction
    from src.models.monte_carlo_simulator import MonteCarloSimulator, MonteCarloResult
except ImportError:
    # Fallback for relative imports
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.ivive_calculator import IVIVECalculator, DrugInput, IVIVEResult
    from models.safety_calculator import SafetyMarginCalculator, SafetyResult, RiskLevel
    from models.toxicophore_analyzer import ToxicophoreAnalyzer, ToxicophoreResult
    from models.qsar_predictor import QSARPredictor, QSARPrediction
    from models.monte_carlo_simulator import MonteCarloSimulator, MonteCarloResult

# mPBPK for large molecules (optional)
try:
    from src.models.mpbpk_engine import mPBPKEngine

    MPBPK_AVAILABLE = True
except ImportError:
    MPBPK_AVAILABLE = False


class DrugType(Enum):
    """약물 유형"""

    SMALL_MOLECULE = "Small Molecule"
    ANTIBODY = "Antibody"
    UNKNOWN = "Unknown"


class OverallRisk(Enum):
    """종합 위험도"""

    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"
    CRITICAL = "Critical Risk"
    UNKNOWN = "Unknown"


@dataclass
class DrugSafetyInput:
    """약물 안전성 평가 입력"""

    # 필수
    smiles: str
    name: str = "Unknown Drug"

    # In vitro 데이터
    IC50_uM: float = 10.0  # hERG IC50 (μM)
    CLint_uL_min_mg: float = 10.0  # Intrinsic clearance
    fu: float = 0.1  # Fraction unbound

    # PK 파라미터
    dose_mg: float = 100.0  # Dose (mg)
    MW: float = 400.0  # Molecular weight (Da)

    # 옵션
    population: str = "EUR"  # 인구 집단 (CYP2D6 시뮬레이션용)
    include_monte_carlo: bool = True  # Monte Carlo 포함 여부
    n_monte_carlo: int = 1000  # Monte Carlo 피험자 수

    def to_drug_input(self) -> DrugInput:
        """IVIVE용 DrugInput 변환"""
        return DrugInput(
            smiles=self.smiles,
            name=self.name,
            IC50_uM=self.IC50_uM,
            CLint_uL_min_mg=self.CLint_uL_min_mg,
            fu=self.fu,
            dose_mg=self.dose_mg,
            MW=self.MW,
        )


@dataclass
class DrugSafetyResult:
    """약물 안전성 평가 결과"""

    # 기본 정보
    drug_name: str
    smiles: str
    drug_type: DrugType
    molecular_weight: float

    # 종합 평가
    overall_risk: OverallRisk
    overall_score: float  # 0-100 (낮을수록 위험)
    overall_recommendation: str

    # 개별 분석 결과
    ivive_result: Optional[IVIVEResult] = None
    safety_result: Optional[SafetyResult] = None
    toxicophore_result: Optional[ToxicophoreResult] = None
    qsar_predictions: Dict[str, QSARPrediction] = field(default_factory=dict)
    monte_carlo_result: Optional[MonteCarloResult] = None

    # 상세 분석
    critical_alerts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # 메타 정보
    analysis_components: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """딕셔너리 변환"""
        return {
            "drug_name": self.drug_name,
            "smiles": self.smiles,
            "drug_type": self.drug_type.value,
            "molecular_weight": self.molecular_weight,
            "overall_assessment": {
                "risk_level": self.overall_risk.value,
                "score": self.overall_score,
                "recommendation": self.overall_recommendation,
            },
            "ivive": self.ivive_result.to_dict() if self.ivive_result else None,
            "safety_margin": self.safety_result.to_dict()
            if self.safety_result
            else None,
            "toxicophore": self.toxicophore_result.to_dict()
            if self.toxicophore_result
            else None,
            "qsar_predictions": {
                k: v.to_dict() for k, v in self.qsar_predictions.items()
            }
            if self.qsar_predictions
            else None,
            "monte_carlo": self.monte_carlo_result.to_dict()
            if self.monte_carlo_result
            else None,
            "alerts": {
                "critical": self.critical_alerts,
                "warnings": self.warnings,
                "notes": self.notes,
            },
            "analysis_components": self.analysis_components,
        }

    def summary(self) -> str:
        """요약 문자열"""
        lines = [
            "=" * 60,
            f"  Drug Safety Assessment: {self.drug_name}",
            "=" * 60,
            f"  SMILES: {self.smiles[:50]}..."
            if len(self.smiles) > 50
            else f"  SMILES: {self.smiles}",
            f"  Type: {self.drug_type.value}",
            f"  MW: {self.molecular_weight:.1f} Da",
            "",
            f"  >>> OVERALL RISK: {self.overall_risk.value}",
            f"  >>> Score: {self.overall_score:.0f}/100",
            "",
            f"  Recommendation:",
            f"    {self.overall_recommendation}",
        ]

        if self.safety_result:
            lines.extend(
                [
                    "",
                    f"  Safety Margin: {self.safety_result.safety_margin:.1f}",
                ]
            )

        if self.critical_alerts:
            lines.append("")
            lines.append("  CRITICAL ALERTS:")
            for alert in self.critical_alerts:
                lines.append(f"    [!] {alert}")

        if self.warnings:
            lines.append("")
            lines.append("  Warnings:")
            for warning in self.warnings:
                lines.append(f"    [*] {warning}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


class DrugSafetyService:
    """
    통합 약물 안전성 평가 서비스

    모든 QSAR 모듈을 통합하여 단일 API로 약물 안전성을 평가합니다.

    Usage:
        service = DrugSafetyService()
        result = service.evaluate(
            smiles="CC(=O)Nc1ccc(O)cc1",
            name="Acetaminophen",
            IC50_uM=100.0,
            CLint_uL_min_mg=50.0
        )
        print(result.summary())
    """

    # 분자량 임계값: 이 이상은 항체로 간주
    MW_THRESHOLD = 900.0

    def __init__(
        self,
        qsar_model_dir: Optional[Path] = None,
        enable_qsar: bool = True,
        enable_monte_carlo: bool = True,
    ):
        """
        Initialize Drug Safety Service.

        Args:
            qsar_model_dir: QSAR 모델 디렉토리
            enable_qsar: QSAR 예측 활성화
            enable_monte_carlo: Monte Carlo 시뮬레이션 활성화
        """
        # 계산기 초기화
        self.ivive_calculator = IVIVECalculator()
        self.safety_calculator = SafetyMarginCalculator()

        # Toxicophore 분석기
        try:
            self.toxicophore_analyzer = ToxicophoreAnalyzer()
            self.toxicophore_enabled = True
        except Exception as e:
            print(f"Warning: ToxicophoreAnalyzer not available: {e}")
            self.toxicophore_enabled = False

        # QSAR 예측기
        self.enable_qsar = enable_qsar
        if enable_qsar:
            try:
                self.qsar_predictor = QSARPredictor(model_dir=qsar_model_dir)
                self.qsar_enabled = True
            except Exception as e:
                print(f"Warning: QSARPredictor not available: {e}")
                self.qsar_enabled = False
        else:
            self.qsar_enabled = False

        # Monte Carlo 시뮬레이터
        self.enable_monte_carlo = enable_monte_carlo
        if enable_monte_carlo:
            try:
                self.monte_carlo = MonteCarloSimulator()
                self.monte_carlo_enabled = True
            except Exception as e:
                print(f"Warning: MonteCarloSimulator not available: {e}")
                self.monte_carlo_enabled = False
        else:
            self.monte_carlo_enabled = False

    def _determine_drug_type(self, mw: float) -> DrugType:
        """분자량으로 약물 유형 결정"""
        if mw > self.MW_THRESHOLD:
            return DrugType.ANTIBODY
        else:
            return DrugType.SMALL_MOLECULE

    def _calculate_overall_score(self, result: DrugSafetyResult) -> float:
        """
        종합 점수 계산 (0-100, 높을수록 안전)

        가중치:
        - Safety Margin: 40%
        - Toxicophore: 25%
        - QSAR: 20%
        - Monte Carlo: 15%
        """
        score = 100.0
        weights_used = 0.0

        # Safety Margin 기여 (40%)
        if result.safety_result:
            sm = result.safety_result.safety_margin
            if sm >= 30:
                sm_score = 100
            elif sm >= 10:
                sm_score = 50 + (sm - 10) * 2.5
            else:
                sm_score = max(0, sm * 5)
            score = sm_score * 0.4
            weights_used += 0.4

        # Toxicophore 기여 (25%)
        if result.toxicophore_result:
            tox_score = 100 - result.toxicophore_result.risk_score
            score += tox_score * 0.25
            weights_used += 0.25

        # QSAR 기여 (20%)
        if result.qsar_predictions:
            toxic_count = sum(
                1 for p in result.qsar_predictions.values() if p.prediction == 1
            )
            total = len(result.qsar_predictions)
            if total > 0:
                qsar_score = 100 * (1 - toxic_count / total)
                score += qsar_score * 0.20
                weights_used += 0.20

        # Monte Carlo 기여 (15%)
        if result.monte_carlo_result and result.monte_carlo_result.sm_p5 is not None:
            mc_sm = result.monte_carlo_result.sm_p5
            if mc_sm >= 30:
                mc_score = 100
            elif mc_sm >= 10:
                mc_score = 50 + (mc_sm - 10) * 2.5
            else:
                mc_score = max(0, mc_sm * 5)
            score += mc_score * 0.15
            weights_used += 0.15

        # 가중치 정규화
        if weights_used > 0:
            score = score / weights_used * 1.0

        return min(100, max(0, score))

    def _determine_overall_risk(
        self, score: float, result: DrugSafetyResult
    ) -> OverallRisk:
        """종합 위험도 결정"""
        # Critical alerts 체크
        if result.critical_alerts:
            return OverallRisk.CRITICAL

        # 점수 기반
        if score >= 80:
            return OverallRisk.LOW
        elif score >= 60:
            return OverallRisk.MODERATE
        elif score >= 40:
            return OverallRisk.HIGH
        else:
            return OverallRisk.CRITICAL

    def _generate_recommendation(self, result: DrugSafetyResult) -> str:
        """권고 사항 생성"""
        risk = result.overall_risk

        if risk == OverallRisk.LOW:
            return "Drug candidate shows favorable safety profile. Proceed with standard development pathway."
        elif risk == OverallRisk.MODERATE:
            return "Moderate safety concerns identified. Additional in vitro/in vivo studies recommended before clinical progression."
        elif risk == OverallRisk.HIGH:
            return "Significant safety concerns. Consider structural optimization to address identified liabilities."
        else:  # CRITICAL
            return "Critical safety issues identified. Drug candidate not recommended for further development in current form."

    def evaluate(
        self,
        smiles: str,
        name: str = "Unknown Drug",
        IC50_uM: float = 10.0,
        CLint_uL_min_mg: float = 10.0,
        fu: float = 0.1,
        dose_mg: float = 100.0,
        MW: float = 400.0,
        population: str = "EUR",
        include_monte_carlo: bool = True,
        n_monte_carlo: int = 1000,
    ) -> DrugSafetyResult:
        """
        약물 안전성 종합 평가.

        Args:
            smiles: SMILES 구조
            name: 약물명
            IC50_uM: hERG IC50 (μM)
            CLint_uL_min_mg: 간 미세소체 청소율
            fu: 혈장 비결합률
            dose_mg: 투여량 (mg)
            MW: 분자량 (Da)
            population: 인구 집단 코드
            include_monte_carlo: Monte Carlo 포함 여부
            n_monte_carlo: Monte Carlo 피험자 수

        Returns:
            DrugSafetyResult
        """
        # 결과 객체 초기화
        drug_type = self._determine_drug_type(MW)
        result = DrugSafetyResult(
            drug_name=name,
            smiles=smiles,
            drug_type=drug_type,
            molecular_weight=MW,
            overall_risk=OverallRisk.UNKNOWN,
            overall_score=0.0,
            overall_recommendation="",
        )

        # 항체인 경우 mPBPK 경로 (현재는 알림만)
        if drug_type == DrugType.ANTIBODY:
            result.notes.append(
                f"Large molecule (MW={MW:.0f} Da) detected. Consider mPBPK analysis."
            )
            if MPBPK_AVAILABLE:
                result.analysis_components.append("mPBPK (suggested)")

        # Step 1: QSAR 예측 (분자 기술자 기반)
        if self.qsar_enabled and self.qsar_predictor.is_trained:
            try:
                predictions = self.qsar_predictor.predict_multiple_endpoints(smiles)
                result.qsar_predictions = predictions
                result.analysis_components.append("QSAR Prediction")

                # 독성 예측 경고
                for endpoint, pred in predictions.items():
                    if pred.prediction == 1 and pred.probability > 0.7:
                        result.warnings.append(
                            f"QSAR predicts toxicity for {endpoint} (prob: {pred.probability:.2f})"
                        )
            except Exception as e:
                result.notes.append(f"QSAR prediction skipped: {e}")

        # Step 2: IVIVE 계산
        try:
            drug_input = DrugInput(
                smiles=smiles,
                name=name,
                IC50_uM=IC50_uM,
                CLint_uL_min_mg=CLint_uL_min_mg,
                fu=fu,
                dose_mg=dose_mg,
                MW=MW,
            )
            ivive_result = self.ivive_calculator.calculate(drug_input)
            result.ivive_result = ivive_result
            result.analysis_components.append("IVIVE")

            # 반감기 경고
            if ivive_result.half_life_hr > 100:
                result.warnings.append(
                    f"Long half-life predicted ({ivive_result.half_life_hr:.0f} hr) - accumulation risk"
                )

        except Exception as e:
            result.notes.append(f"IVIVE calculation failed: {e}")
            ivive_result = None

        # Step 3: Safety Margin 계산
        if ivive_result:
            try:
                safety_result = self.safety_calculator.calculate_herg_safety(
                    herg_IC50_uM=IC50_uM, Cmax_uM=ivive_result.Cmax_uM
                )
                result.safety_result = safety_result
                result.analysis_components.append("Safety Margin")

                # Safety Margin 경고
                if safety_result.risk_level == RiskLevel.CONCERN:
                    result.critical_alerts.append(
                        f"hERG Safety Margin ({safety_result.safety_margin:.1f}) below threshold - cardiac risk"
                    )
                elif safety_result.risk_level == RiskLevel.MODERATE:
                    result.warnings.append(
                        f"Moderate hERG Safety Margin ({safety_result.safety_margin:.1f}) - further testing needed"
                    )

            except Exception as e:
                result.notes.append(f"Safety margin calculation failed: {e}")

        # Step 4: Toxicophore 분석
        if self.toxicophore_enabled:
            try:
                tox_result = self.toxicophore_analyzer.analyze(smiles)
                result.toxicophore_result = tox_result
                result.analysis_components.append("Toxicophore")

                # 구조적 경고
                for match in tox_result.matches:
                    if match.pattern.severity >= 4:
                        result.warnings.append(
                            f"Structural alert: {match.pattern.name} (severity: {match.pattern.severity}/5)"
                        )

                if tox_result.toxicity_risk == "High":
                    result.critical_alerts.append(
                        f"High toxicophore risk score ({tox_result.risk_score:.0f}/100)"
                    )

            except Exception as e:
                result.notes.append(f"Toxicophore analysis failed: {e}")

        # Step 5: Monte Carlo 시뮬레이션 (CYP2D6)
        if include_monte_carlo and self.monte_carlo_enabled and ivive_result:
            try:
                mc_result = self.monte_carlo.simulate(
                    base_cmax=ivive_result.Cmax_uM,
                    population=population,
                    n=n_monte_carlo,
                    ic50_uM=IC50_uM,
                )
                result.monte_carlo_result = mc_result
                result.analysis_components.append("Monte Carlo (CYP2D6)")

                # CYP2D6 관련 경고
                if mc_result.pct_high_exposure > 10:
                    result.warnings.append(
                        f"{mc_result.pct_high_exposure:.1f}% of {population} population shows high exposure"
                    )

                # PM에서 Safety Margin 문제
                if mc_result.sm_p5 is not None and mc_result.sm_p5 < 10:
                    result.critical_alerts.append(
                        f"CYP2D6 PM patients at cardiac risk (SM P5: {mc_result.sm_p5:.1f})"
                    )

            except Exception as e:
                result.notes.append(f"Monte Carlo simulation failed: {e}")

        # 종합 점수 및 위험도 계산
        result.overall_score = self._calculate_overall_score(result)
        result.overall_risk = self._determine_overall_risk(result.overall_score, result)
        result.overall_recommendation = self._generate_recommendation(result)

        return result

    def quick_screen(self, smiles: str, IC50_uM: float = 10.0) -> Dict:
        """
        빠른 스크리닝 (주요 지표만)

        Args:
            smiles: SMILES 구조
            IC50_uM: hERG IC50

        Returns:
            간략한 결과 딕셔너리
        """
        result = self.evaluate(
            smiles=smiles, IC50_uM=IC50_uM, include_monte_carlo=False
        )

        return {
            "smiles": smiles,
            "risk_level": result.overall_risk.value,
            "score": result.overall_score,
            "safety_margin": result.safety_result.safety_margin
            if result.safety_result
            else None,
            "toxicophore_alerts": result.toxicophore_result.total_alerts
            if result.toxicophore_result
            else 0,
            "critical_alerts": len(result.critical_alerts),
            "warnings": len(result.warnings),
        }

    def batch_evaluate(
        self, compounds: List[Dict], include_monte_carlo: bool = False
    ) -> List[DrugSafetyResult]:
        """
        배치 평가.

        Args:
            compounds: 화합물 딕셔너리 리스트 (smiles, name, IC50_uM 등)
            include_monte_carlo: Monte Carlo 포함 여부

        Returns:
            DrugSafetyResult 리스트
        """
        results = []
        for compound in compounds:
            result = self.evaluate(
                smiles=compound.get("smiles", ""),
                name=compound.get("name", "Unknown"),
                IC50_uM=compound.get("IC50_uM", 10.0),
                CLint_uL_min_mg=compound.get("CLint", 10.0),
                fu=compound.get("fu", 0.1),
                dose_mg=compound.get("dose_mg", 100.0),
                MW=compound.get("MW", 400.0),
                include_monte_carlo=include_monte_carlo,
            )
            results.append(result)

        return results


# =============================================================================
# Convenience Functions
# =============================================================================


def evaluate_drug_safety(
    smiles: str, IC50_uM: float = 10.0, CLint: float = 10.0, fu: float = 0.1
) -> Dict:
    """
    빠른 약물 안전성 평가.

    Args:
        smiles: SMILES 구조
        IC50_uM: hERG IC50
        CLint: 간 미세소체 청소율
        fu: 혈장 비결합률

    Returns:
        결과 딕셔너리
    """
    service = DrugSafetyService(enable_qsar=False)
    result = service.evaluate(
        smiles=smiles,
        IC50_uM=IC50_uM,
        CLint_uL_min_mg=CLint,
        fu=fu,
        include_monte_carlo=False,
    )
    return result.to_dict()


def get_safety_summary(smiles: str, IC50_uM: float = 10.0) -> str:
    """
    안전성 요약 문자열.

    Args:
        smiles: SMILES 구조
        IC50_uM: hERG IC50

    Returns:
        요약 문자열
    """
    service = DrugSafetyService(enable_qsar=False)
    result = service.evaluate(smiles=smiles, IC50_uM=IC50_uM, include_monte_carlo=False)
    return result.summary()


# =============================================================================
# Main (Test)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Drug Safety Service Test")
    print("=" * 60)

    # Initialize service with QSAR enabled
    print("\n>>> Initializing DrugSafetyService...")
    service = DrugSafetyService(enable_qsar=True)

    # Report loaded models
    if service.qsar_enabled and service.qsar_predictor.is_trained:
        print(f"    QSAR models loaded: {len(service.qsar_predictor.models)}")
        for endpoint in service.qsar_predictor.models.keys():
            print(f"      - {endpoint}")
    else:
        print("    QSAR models not available (enable_qsar=False)")

    # Test compounds
    test_compounds = [
        {
            "name": "Acetaminophen",
            "smiles": "CC(=O)Nc1ccc(O)cc1",
            "IC50_uM": 100.0,  # Safe hERG
            "CLint": 50.0,
            "fu": 0.8,
        },
        {
            "name": "Terfenadine (withdrawn)",
            "smiles": "CC(C)(C)c1ccc(cc1)C(O)CCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4",
            "IC50_uM": 0.5,  # Dangerous hERG
            "CLint": 100.0,
            "fu": 0.03,
        },
        {
            "name": "Nitrobenzene (toxic)",
            "smiles": "c1ccc(cc1)[N+](=O)[O-]",
            "IC50_uM": 50.0,
            "CLint": 20.0,
            "fu": 0.5,
        },
    ]

    for compound in test_compounds:
        print(f"\n{'=' * 60}")
        print(f">>> Evaluating: {compound['name']}")

        result = service.evaluate(
            smiles=compound["smiles"],
            name=compound["name"],
            IC50_uM=compound["IC50_uM"],
            CLint_uL_min_mg=compound["CLint"],
            fu=compound["fu"],
            include_monte_carlo=True,
            n_monte_carlo=500,
        )

        print(result.summary())

        print("\nAnalysis Components Used:")
        for comp in result.analysis_components:
            print(f"  - {comp}")

        # Show QSAR predictions if available
        if result.qsar_predictions:
            print(f"\nQSAR Predictions ({len(result.qsar_predictions)} endpoints):")
            toxic_count = sum(
                1 for p in result.qsar_predictions.values() if p.prediction == 1
            )
            print(f"  Toxic endpoints: {toxic_count}/{len(result.qsar_predictions)}")

    print("\n" + "=" * 60)
    print("  Drug Safety Service Test Complete!")
    print("=" * 60)
