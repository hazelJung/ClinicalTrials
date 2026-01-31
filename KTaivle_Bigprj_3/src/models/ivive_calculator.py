"""
IVIVE Calculator - In Vitro to In Vivo Extrapolation
=====================================================
Step 2: Well-Stirred Model 기반 CLint -> CLh 변환 및 Cmax 예측

References:
- Obach (1999): Prediction of human clearance
- Houston (1994): In vitro-in vivo correlations for drug metabolism
- Riley et al. (2005): A unified model for hepatic clearance

Author: AI-Driven Clinical Trial Platform
Date: 2026-01-26
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

try:
    from .drug_schema import DrugInput
except ImportError:
    # Standalone execution support
    import sys
    import os

    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )
    from src.models.drug_schema import DrugInput


@dataclass
class PhysiologicalParams:
    """
    인체 생리학적 파라미터 (표준 70kg 성인 기준)

    References:
    - Davies & Morris (1993): Physiological parameters
    - ICRP Publication 89 (2002)
    """

    # 체중 (kg)
    body_weight: float = 70.0

    # 간 관련
    liver_weight: float = 1.8  # kg
    liver_blood_flow: float = 1.5  # L/min (약 21 mL/min/kg)
    hepatocyte_density: float = 1.2e8  # cells/g liver

    # 혈장 관련
    plasma_volume: float = 3.0  # L
    blood_volume: float = 5.0  # L
    cardiac_output: float = 5.0  # L/min

    # 비율
    hematocrit: float = 0.45  # 적혈구 비율

    @property
    def Qh(self) -> float:
        """간 혈류량 (L/hr)"""
        return self.liver_blood_flow * 60  # L/min -> L/hr

    @property
    def Qh_per_kg(self) -> float:
        """체중 당 간 혈류량 (L/hr/kg)"""
        return self.Qh / self.body_weight


@dataclass
class IVIVEResult:
    """IVIVE 계산 결과"""

    # 청소율 결과
    CLint_uL_min_mg: float  # 입력 CLint
    CLint_scaled: float  # 스케일링된 CLint (mL/min/kg)
    CLh: float  # 간 청소율 (L/hr)
    CLh_per_kg: float  # 체중당 간 청소율 (L/hr/kg)

    # 추출비
    extraction_ratio: float  # 간 추출비 (0-1)

    # PK 예측
    half_life_hr: float  # 반감기 (hr)
    Cmax_uM: float  # 최대 혈중 농도 (μM)
    AUC: float  # Area Under Curve (μM·hr)

    # 메타 정보
    model_used: str = "well_stirred"  # 사용된 모델

    def to_dict(self) -> Dict:
        return {
            "CLint_input": self.CLint_uL_min_mg,
            "CLint_scaled": self.CLint_scaled,
            "CLh": self.CLh,
            "CLh_per_kg": self.CLh_per_kg,
            "extraction_ratio": self.extraction_ratio,
            "half_life_hr": self.half_life_hr,
            "Cmax_uM": self.Cmax_uM,
            "AUC": self.AUC,
            "model": self.model_used,
        }


class IVIVECalculator:
    """
    IVIVE (In Vitro to In Vivo Extrapolation) 계산기

    Well-Stirred Model을 사용하여:
    1. CLint (in vitro) → CLh (in vivo) 변환
    2. Cmax 및 AUC 예측

    Well-Stirred Model:
        CLh = (Qh × fu × CLint_scaled) / (Qh + fu × CLint_scaled)

    where:
        Qh = 간 혈류량
        fu = 혈장 비결합률
        CLint_scaled = 스케일링된 내인성 청소율
    """

    # 스케일링 상수
    MPPGL = 45.0  # mg microsomal protein per gram liver
    HPGL = 1.2e8  # hepatocytes per gram liver (120 million)

    def __init__(self, physio: PhysiologicalParams = None):
        """
        Initialize IVIVE calculator.

        Args:
            physio: 생리학적 파라미터 (기본값: 70kg 성인)
        """
        self.physio = physio if physio else PhysiologicalParams()

    def scale_clint(self, clint_uL_min_mg: float, fu_mic: float = 1.0) -> float:
        """
        In vitro CLint를 in vivo 스케일로 변환.

        Scaling:
            CLint_scaled (mL/min/kg) = CLint (μL/min/mg)
                                       × MPPGL (mg/g liver)
                                       × Liver Weight (g) / Body Weight (kg)
                                       × (1/1000) [μL → mL]

        Args:
            clint_uL_min_mg: In vitro CLint (μL/min/mg microsomal protein)
            fu_mic: 미세소체 비결합률

        Returns:
            Scaled CLint (mL/min/kg body weight)
        """
        # μL/min/mg → mL/min/kg
        liver_g = self.physio.liver_weight * 1000  # kg → g

        clint_scaled = (
            clint_uL_min_mg
            * self.MPPGL
            * liver_g
            / self.physio.body_weight
            / 1000  # μL → mL
            / fu_mic
        )

        return clint_scaled

    def calculate_hepatic_clearance(
        self, clint_scaled: float, fu: float, model: str = "well_stirred"
    ) -> Tuple[float, float]:
        """
        간 청소율(CLh) 계산.

        Well-Stirred Model:
            CLh = (Qh × fu × CLint) / (Qh + fu × CLint)

        Args:
            clint_scaled: 스케일링된 CLint (mL/min/kg)
            fu: 혈장 비결합률
            model: 모델 유형 ('well_stirred', 'parallel_tube')

        Returns:
            (CLh in L/hr, Extraction Ratio)
        """
        # mL/min/kg → L/hr/kg
        clint_L_hr_kg = clint_scaled * 60 / 1000

        Qh = self.physio.Qh_per_kg  # L/hr/kg

        if model == "well_stirred":
            # Well-Stirred Model
            numerator = Qh * fu * clint_L_hr_kg
            denominator = Qh + fu * clint_L_hr_kg

            if denominator < 1e-10:
                CLh_per_kg = 0.0
            else:
                CLh_per_kg = numerator / denominator

        elif model == "parallel_tube":
            # Parallel Tube Model (Dispersion)
            fu_clint = fu * clint_L_hr_kg
            CLh_per_kg = Qh * (1 - np.exp(-fu_clint / Qh))

        else:
            raise ValueError(f"Unknown model: {model}")

        # Extraction Ratio
        extraction_ratio = CLh_per_kg / Qh if Qh > 0 else 0.0
        extraction_ratio = min(1.0, extraction_ratio)  # Cap at 1.0

        # Total CLh (L/hr)
        CLh = CLh_per_kg * self.physio.body_weight

        return CLh, extraction_ratio

    def predict_pk_parameters(
        self, drug: DrugInput, CLh: float
    ) -> Tuple[float, float, float]:
        """
        PK 파라미터 예측 (Half-life, Cmax, AUC).

        Args:
            drug: 약물 입력 데이터
            CLh: 간 청소율 (L/hr)

        Returns:
            (half_life_hr, Cmax_uM, AUC_uM_hr)
        """
        # 분포용적 (L)
        Vd = drug.volume_of_distribution * self.physio.body_weight

        # 제거 상수 (1/hr)
        if Vd > 0 and CLh > 0:
            ke = CLh / Vd
        else:
            ke = 0.01  # 기본값

        # 반감기 (hr)
        half_life = 0.693 / ke if ke > 0 else 100.0

        # MW check
        mw = drug.MW if drug.MW is not None else 400.0

        # Cmax 예측 (1-Compartment IV bolus 가정)
        # Cmax = (F × Dose) / Vd
        dose_umol = (drug.dose_mg * 1000) / mw  # mg → μmol

        # μmol / L = μM
        Cmax_uM = (drug.bioavailability * dose_umol) / Vd

        # AUC = Dose / CL = Cmax / ke
        if CLh > 0:
            AUC = (drug.bioavailability * dose_umol) / (CLh * 1000 / mw)
            # 간단히: AUC ≈ Cmax / ke
            AUC = Cmax_uM / ke if ke > 0 else Cmax_uM * 100
        else:
            AUC = Cmax_uM * 100

        return half_life, Cmax_uM, AUC

    def calculate(self, drug: DrugInput) -> IVIVEResult:
        """
        전체 IVIVE 계산 수행.

        Args:
            drug: 약물 입력 데이터

        Returns:
            IVIVEResult: IVIVE 결과
        """
        # 1. CLint 스케일링
        clint_scaled = self.scale_clint(drug.CLint_uL_min_mg, drug.fu_mic)

        # 2. 간 청소율 계산
        CLh, extraction_ratio = self.calculate_hepatic_clearance(
            clint_scaled, drug.fu, model="well_stirred"
        )

        # 3. PK 파라미터 예측
        half_life, Cmax, AUC = self.predict_pk_parameters(drug, CLh)

        return IVIVEResult(
            CLint_uL_min_mg=drug.CLint_uL_min_mg,
            CLint_scaled=clint_scaled,
            CLh=CLh,
            CLh_per_kg=CLh / self.physio.body_weight,
            extraction_ratio=extraction_ratio,
            half_life_hr=half_life,
            Cmax_uM=Cmax,
            AUC=AUC,
            model_used="well_stirred",
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_ivive(
    smiles: str,
    clint: float,
    fu: float = 0.1,
    dose_mg: float = 100.0,
    mw: float = 400.0,
) -> Dict:
    """
    빠른 IVIVE 계산.

    Args:
        smiles: SMILES 구조
        clint: CLint (μL/min/mg protein)
        fu: 혈장 비결합률
        dose_mg: 투여량 (mg)
        mw: 분자량 (Da)

    Returns:
        Dictionary with IVIVE results
    """
    drug = DrugInput(
        smiles=smiles, CLint_uL_min_mg=clint, fu=fu, dose_mg=dose_mg, MW=mw
    )

    calculator = IVIVECalculator()
    result = calculator.calculate(drug)

    return result.to_dict()


# =============================================================================
# Main (Test)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  IVIVE Calculator Test")
    print("=" * 60)

    # Test compound
    drug = DrugInput(
        smiles="CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
        name="Acetaminophen",
        IC50_uM=100.0,
        CLint_uL_min_mg=50.0,  # μL/min/mg
        fu=0.8,  # 80% unbound
        dose_mg=500.0,  # 500 mg dose
        MW=151.16,  # Da
    )

    print(f"\nTest Drug: {drug.name}")
    print(f"  SMILES: {drug.smiles}")
    print(f"  CLint: {drug.CLint_uL_min_mg} μL/min/mg")
    print(f"  fu: {drug.fu}")
    print(f"  Dose: {drug.dose_mg} mg")

    # Calculate IVIVE
    calculator = IVIVECalculator()
    result = calculator.calculate(drug)

    print("\n>>> IVIVE Results:")
    print(f"  CLint (scaled): {result.CLint_scaled:.2f} mL/min/kg")
    print(f"  CLh: {result.CLh:.2f} L/hr")
    print(f"  Extraction Ratio: {result.extraction_ratio:.2%}")
    print(f"  Half-life: {result.half_life_hr:.1f} hr")
    print(f"  Cmax: {result.Cmax_uM:.2f} μM")
    print(f"  AUC: {result.AUC:.2f} μM·hr")

    print("\n" + "=" * 60)
    print("  IVIVE Calculator Test Complete!")
    print("=" * 60)
