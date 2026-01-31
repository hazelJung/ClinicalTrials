"""
Drug Data Schemas
=================
약물 입력 및 예측 결과를 위한 데이터 클래스 정의.

Author: AI-Driven Clinical Trial Platform
Date: 2026-01-26
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class DrugInput:
    """
    약물 입력 데이터 (통합 스키마)

    QSAR 및 IVIVE/PBPK 계산에 필요한 모든 입력 파라미터.
    """

    # 기본 정보
    smiles: str
    name: str = "Unknown Drug"

    # In vitro 데이터 (실험값 또는 예측값)
    IC50_uM: Optional[float] = None  # hERG or Toxicity IC50 (μM)
    CLint_uL_min_mg: float = 10.0  # Intrinsic Clearance (μL/min/mg protein)
    fu: float = 0.1  # Fraction unbound in plasma (0.0 ~ 1.0)
    fu_mic: float = 1.0  # Fraction unbound in microsomes

    # 물리화학적 특성
    MW: Optional[float] = None  # Molecular Weight (Da)
    logP: Optional[float] = None  # LogP

    # PK 파라미터 (IVIVE용 가정값)
    dose_mg: float = 100.0  # 투여량 (mg)
    bioavailability: float = 1.0  # 생체이용률 (F, 0.0 ~ 1.0)
    volume_of_distribution: float = 1.0  # 분포용적 (L/kg)

    def validate(self):
        """데이터 유효성 검증"""
        if not self.smiles:
            raise ValueError("SMILES string is required.")
        if self.fu < 0.0 or self.fu > 1.0:
            raise ValueError(
                f"Fraction unbound (fu) must be between 0.0 and 1.0, got {self.fu}"
            )
        if self.CLint_uL_min_mg < 0:
            raise ValueError(f"CLint must be non-negative, got {self.CLint_uL_min_mg}")


@dataclass
class DrugResult:
    """통합 안전성 평가 결과"""

    input_data: DrugInput

    # Step 1: QSAR Predictions
    qsar_predictions: Dict[str, float] = field(
        default_factory=dict
    )  # Label -> Probability/Class
    qsar_uncertainty: Dict[str, float] = field(
        default_factory=dict
    )  # Label -> Uncertainty

    # Step 2: IVIVE / PK
    ivive_result: Optional[Dict[str, Any]] = None  # Detailed IVIVE results
    predicted_cmax_nm: Optional[float] = None
    predicted_auc: Optional[float] = None
    predicted_half_life: Optional[float] = None

    # Step 3: Safety Margin
    safety_margin: Optional[float] = None
    safety_category: str = "Unknown"  # Safe, Moderate, Concern

    # Step 4: Toxicophores
    toxicophores_detected: List[str] = field(default_factory=list)

    # Step 5: Applicability Domain
    in_applicability_domain: bool = True
    ad_details: Dict[str, Any] = field(default_factory=dict)

    # Step 6: Population Analysis (Monte Carlo)
    population_stats: Dict[str, float] = field(
        default_factory=dict
    )  # mean, std, p5, p95

    # Final Decision
    approval_status: str = "Pending"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """결과를 딕셔너리로 변환 (API 응답용)"""
        return {
            "drug_name": self.input_data.name,
            "predictions": self.qsar_predictions,
            "pk_params": {
                "cmax_nm": self.predicted_cmax_nm,
                "auc": self.predicted_auc,
                "half_life": self.predicted_half_life,
            },
            "safety": {
                "margin": self.safety_margin,
                "category": self.safety_category,
                "toxicophores": self.toxicophores_detected,
            },
            "decision": {"status": self.approval_status, "warnings": self.warnings},
        }
