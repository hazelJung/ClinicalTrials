"""
Safety Margin Calculator
========================
Step 3: IC50 / Cmax 기반 안전역 계산

Safety Margin (SM) = IC50 / Cmax
- SM >= 30: Safe (녹색)
- 10 <= SM < 30: Moderate Risk (황색)
- SM < 10: Concern (적색)

References:
- ICH S7B: Cardiac Safety Guidelines
- Redfern et al. (2003): hERG Safety Margins
- Fermini & Fossa (2003): Cardiac Safety Assessment

Author: AI-Driven Clinical Trial Platform
Date: 2026-01-26
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class RiskLevel(Enum):
    """위험도 수준"""

    SAFE = "Safe"
    MODERATE = "Moderate Risk"
    CONCERN = "High Concern"
    UNKNOWN = "Unknown"


@dataclass
class SafetyThresholds:
    """안전역 임계값 설정"""

    # hERG (심장 독성)
    herg_safe: float = 30.0  # SM >= 30: Safe
    herg_moderate: float = 10.0  # SM >= 10: Moderate

    # Hepatotoxicity (간독성)
    hepato_safe: float = 50.0
    hepato_moderate: float = 20.0

    # General toxicity
    general_safe: float = 30.0
    general_moderate: float = 10.0


@dataclass
class SafetyResult:
    """안전성 평가 결과"""

    # 기본 Safety Margin
    safety_margin: float  # IC50 / Cmax

    # 리스크 레벨
    risk_level: RiskLevel
    risk_color: str  # "green", "yellow", "red"

    # 상세 정보
    IC50_uM: float
    Cmax_uM: float

    # 추가 마진
    margin_to_safe: float  # Safe 기준까지 남은 마진
    therapeutic_index: float  # 치료 지수 (TI)

    # 권고 사항
    recommendation: str

    def to_dict(self) -> Dict:
        return {
            "safety_margin": self.safety_margin,
            "risk_level": self.risk_level.value,
            "risk_color": self.risk_color,
            "IC50_uM": self.IC50_uM,
            "Cmax_uM": self.Cmax_uM,
            "margin_to_safe": self.margin_to_safe,
            "therapeutic_index": self.therapeutic_index,
            "recommendation": self.recommendation,
        }


@dataclass
class ComprehensiveSafetyResult:
    """종합 안전성 평가 결과"""

    # 개별 Safety Margin
    herg_safety: SafetyResult
    hepato_safety: Optional[SafetyResult] = None

    # 종합 평가
    overall_risk: RiskLevel = RiskLevel.UNKNOWN
    overall_score: float = 0.0  # 0-100 점수

    # 세부 분석
    critical_findings: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "herg_safety": self.herg_safety.to_dict(),
            "hepato_safety": self.hepato_safety.to_dict()
            if self.hepato_safety
            else None,
            "overall_risk": self.overall_risk.value,
            "overall_score": self.overall_score,
            "critical_findings": self.critical_findings,
            "warnings": self.warnings,
        }


class SafetyMarginCalculator:
    """
    안전역 계산기

    Safety Margin = IC50 / Cmax

    평가 기준 (hERG):
    - SM >= 30: Safe - 임상 진행 권장
    - 10 <= SM < 30: Moderate - 추가 연구 필요
    - SM < 10: Concern - 임상 진행 위험
    """

    def __init__(self, thresholds: SafetyThresholds = None):
        """
        Initialize calculator.

        Args:
            thresholds: 안전역 임계값
        """
        self.thresholds = thresholds if thresholds else SafetyThresholds()

    def calculate_safety_margin(
        self, IC50_uM: float, Cmax_uM: float, endpoint: str = "herg"
    ) -> SafetyResult:
        """
        Safety Margin 계산.

        SM = IC50 / Cmax

        Args:
            IC50_uM: IC50 값 (μM)
            Cmax_uM: 최대 혈중 농도 (μM)
            endpoint: 독성 엔드포인트 ('herg', 'hepato', 'general')

        Returns:
            SafetyResult
        """
        # Safety Margin 계산
        if Cmax_uM <= 0 or IC50_uM <= 0:
            return SafetyResult(
                safety_margin=0.0,
                risk_level=RiskLevel.UNKNOWN,
                risk_color="gray",
                IC50_uM=IC50_uM,
                Cmax_uM=Cmax_uM,
                margin_to_safe=0.0,
                therapeutic_index=0.0,
                recommendation="Invalid input values",
            )

        SM = IC50_uM / Cmax_uM

        # 임계값 선택
        if endpoint == "herg":
            safe_threshold = self.thresholds.herg_safe
            moderate_threshold = self.thresholds.herg_moderate
        elif endpoint == "hepato":
            safe_threshold = self.thresholds.hepato_safe
            moderate_threshold = self.thresholds.hepato_moderate
        else:
            safe_threshold = self.thresholds.general_safe
            moderate_threshold = self.thresholds.general_moderate

        # 리스크 레벨 결정
        if SM >= safe_threshold:
            risk_level = RiskLevel.SAFE
            risk_color = "green"
            recommendation = "Low risk. Clinical progression recommended."
        elif SM >= moderate_threshold:
            risk_level = RiskLevel.MODERATE
            risk_color = "yellow"
            recommendation = (
                "Moderate risk. Additional cardiac safety studies recommended."
            )
        else:
            risk_level = RiskLevel.CONCERN
            risk_color = "red"
            recommendation = (
                "High risk. Consider structural optimization or dose reduction."
            )

        # 마진 계산
        margin_to_safe = safe_threshold - SM if SM < safe_threshold else 0.0

        # 치료 지수 (간단화: SM 기반)
        therapeutic_index = SM / moderate_threshold if moderate_threshold > 0 else 0.0

        return SafetyResult(
            safety_margin=SM,
            risk_level=risk_level,
            risk_color=risk_color,
            IC50_uM=IC50_uM,
            Cmax_uM=Cmax_uM,
            margin_to_safe=margin_to_safe,
            therapeutic_index=therapeutic_index,
            recommendation=recommendation,
        )

    def calculate_herg_safety(
        self, herg_IC50_uM: float, Cmax_uM: float
    ) -> SafetyResult:
        """
        hERG Safety Margin 계산 (심장 독성).

        Args:
            herg_IC50_uM: hERG IC50 (μM)
            Cmax_uM: Cmax (μM)

        Returns:
            SafetyResult
        """
        return self.calculate_safety_margin(herg_IC50_uM, Cmax_uM, endpoint="herg")

    def calculate_hepato_safety(
        self, hepato_IC50_uM: float, Cmax_uM: float
    ) -> SafetyResult:
        """
        간독성 Safety Margin 계산.

        Args:
            hepato_IC50_uM: Hepatotoxicity IC50 (μM)
            Cmax_uM: Cmax (μM)

        Returns:
            SafetyResult
        """
        return self.calculate_safety_margin(hepato_IC50_uM, Cmax_uM, endpoint="hepato")

    def comprehensive_assessment(
        self,
        Cmax_uM: float,
        herg_IC50_uM: float,
        hepato_IC50_uM: Optional[float] = None,
    ) -> ComprehensiveSafetyResult:
        """
        종합 안전성 평가.

        Args:
            Cmax_uM: 최대 혈중 농도 (μM)
            herg_IC50_uM: hERG IC50 (μM)
            hepato_IC50_uM: 간독성 IC50 (μM, 선택)

        Returns:
            ComprehensiveSafetyResult
        """
        # hERG 평가
        herg_result = self.calculate_herg_safety(herg_IC50_uM, Cmax_uM)

        # 간독성 평가 (선택)
        hepato_result = None
        if hepato_IC50_uM is not None:
            hepato_result = self.calculate_hepato_safety(hepato_IC50_uM, Cmax_uM)

        # 종합 리스크 결정
        critical_findings = []
        warnings = []

        # hERG 기반 결정
        if herg_result.risk_level == RiskLevel.CONCERN:
            overall_risk = RiskLevel.CONCERN
            critical_findings.append(
                f"hERG Safety Margin ({herg_result.safety_margin:.1f}) below threshold"
            )
        elif herg_result.risk_level == RiskLevel.MODERATE:
            overall_risk = RiskLevel.MODERATE
            warnings.append(
                f"hERG Safety Margin ({herg_result.safety_margin:.1f}) requires monitoring"
            )
        else:
            overall_risk = RiskLevel.SAFE

        # 간독성 추가 평가
        if hepato_result:
            if hepato_result.risk_level == RiskLevel.CONCERN:
                overall_risk = RiskLevel.CONCERN
                critical_findings.append(
                    f"Hepatotoxicity concern (SM: {hepato_result.safety_margin:.1f})"
                )
            elif hepato_result.risk_level == RiskLevel.MODERATE:
                if overall_risk == RiskLevel.SAFE:
                    overall_risk = RiskLevel.MODERATE
                warnings.append(
                    f"Hepatotoxicity requires attention (SM: {hepato_result.safety_margin:.1f})"
                )

        # 종합 점수 계산 (0-100)
        herg_score = min(100, herg_result.safety_margin * 3.33)  # SM=30 → 100점
        if hepato_result:
            hepato_score = min(100, hepato_result.safety_margin * 2)  # SM=50 → 100점
            overall_score = herg_score * 0.6 + hepato_score * 0.4  # 가중 평균
        else:
            overall_score = herg_score

        return ComprehensiveSafetyResult(
            herg_safety=herg_result,
            hepato_safety=hepato_result,
            overall_risk=overall_risk,
            overall_score=overall_score,
            critical_findings=critical_findings,
            warnings=warnings,
        )

    def calculate_required_ic50(self, Cmax_uM: float, target_SM: float = 30.0) -> float:
        """
        목표 Safety Margin 달성에 필요한 IC50 계산.

        Args:
            Cmax_uM: 현재 Cmax
            target_SM: 목표 Safety Margin

        Returns:
            필요한 IC50 (μM)
        """
        return Cmax_uM * target_SM

    def calculate_max_cmax(self, IC50_uM: float, target_SM: float = 30.0) -> float:
        """
        목표 Safety Margin을 유지하면서 허용 가능한 최대 Cmax.

        Args:
            IC50_uM: 현재 IC50
            target_SM: 목표 Safety Margin

        Returns:
            허용 가능한 최대 Cmax (μM)
        """
        return IC50_uM / target_SM if target_SM > 0 else 0.0


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_safety_check(IC50_uM: float, Cmax_uM: float) -> Dict:
    """
    빠른 Safety Margin 체크.

    Args:
        IC50_uM: IC50 값 (μM)
        Cmax_uM: Cmax 값 (μM)

    Returns:
        Safety result dictionary
    """
    calculator = SafetyMarginCalculator()
    result = calculator.calculate_herg_safety(IC50_uM, Cmax_uM)
    return result.to_dict()


def get_risk_category(safety_margin: float) -> str:
    """
    Safety Margin에서 리스크 카테고리 반환.

    Args:
        safety_margin: SM 값

    Returns:
        Risk category string
    """
    if safety_margin >= 30:
        return "SAFE"
    elif safety_margin >= 10:
        return "MODERATE"
    else:
        return "CONCERN"


# =============================================================================
# Main (Test)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Safety Margin Calculator Test")
    print("=" * 60)

    calculator = SafetyMarginCalculator()

    # Test cases
    test_cases = [
        ("Safe Drug", 100.0, 1.0),  # SM = 100
        ("Moderate Risk", 50.0, 2.5),  # SM = 20
        ("High Concern", 10.0, 2.0),  # SM = 5
    ]

    for name, ic50, cmax in test_cases:
        print(f"\n>>> {name}")
        print(f"    IC50: {ic50} μM, Cmax: {cmax} μM")

        result = calculator.calculate_herg_safety(ic50, cmax)

        print(f"    Safety Margin: {result.safety_margin:.1f}")
        print(f"    Risk Level: {result.risk_level.value} ({result.risk_color})")
        print(f"    Recommendation: {result.recommendation}")

    # Comprehensive assessment
    print("\n" + "-" * 60)
    print(">>> Comprehensive Safety Assessment")

    comp_result = calculator.comprehensive_assessment(
        Cmax_uM=2.0, herg_IC50_uM=50.0, hepato_IC50_uM=100.0
    )

    print(f"    Overall Risk: {comp_result.overall_risk.value}")
    print(f"    Overall Score: {comp_result.overall_score:.1f}/100")
    print(f"    hERG SM: {comp_result.herg_safety.safety_margin:.1f}")
    if comp_result.hepato_safety:
        print(f"    Hepato SM: {comp_result.hepato_safety.safety_margin:.1f}")
    if comp_result.warnings:
        print(f"    Warnings: {comp_result.warnings}")

    print("\n" + "=" * 60)
    print("  Safety Margin Calculator Test Complete!")
    print("=" * 60)
