"""
Toxicophore Analyzer
====================
Step 4: SMARTS 기반 독성 작용기 스크리닝

6개 핵심 독성 작용기 (Toxicophores):
1. Quinone - 반응성 산소종 생성
2. Epoxide - DNA 알킬화
3. Nitroaromatic - 간독성
4. Halogenated Aromatic - 갑상선 독성
5. Michael Acceptor - 단백질 공유결합
6. Acyl Halide - 반응성 대사체

References:
- Enoch et al. (2011): A review of structural alerts for skin sensitization
- Stepan et al. (2011): Structural Alert Identification
- Kazius et al. (2005): Derivation and validation of toxicophores

Author: AI-Driven Clinical Trial Platform
Date: 2026-01-26
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class ToxicityType(Enum):
    """독성 유형"""

    HEPATOTOXICITY = "Hepatotoxicity"
    CARDIOTOXICITY = "Cardiotoxicity"
    GENOTOXICITY = "Genotoxicity"
    SKIN_SENSITIZATION = "Skin Sensitization"
    THYROID_TOXICITY = "Thyroid Toxicity"
    REACTIVE_METABOLITE = "Reactive Metabolite"
    GENERAL = "General Toxicity"


@dataclass
class ToxicophorePattern:
    """독성 작용기 패턴 정의"""

    name: str  # 패턴 이름
    smarts: str  # SMARTS 패턴
    toxicity_types: List[ToxicityType]  # 관련 독성 유형
    severity: int  # 심각도 (1-5)
    description: str  # 설명
    reference: str = ""  # 참조 문헌


@dataclass
class ToxicophoreMatch:
    """독성 작용기 매치 결과"""

    pattern: ToxicophorePattern
    atom_indices: List[Tuple[int, ...]]  # 매칭된 원자 인덱스
    count: int  # 매치 횟수


@dataclass
class ToxicophoreResult:
    """독성 작용기 분석 결과"""

    smiles: str
    is_valid: bool = True

    # 매치 결과
    matches: List[ToxicophoreMatch] = field(default_factory=list)
    total_alerts: int = 0

    # 요약
    toxicity_risk: str = "Low"  # "Low", "Medium", "High"
    risk_score: float = 0.0  # 0-100

    # 상세 정보
    found_patterns: List[str] = field(default_factory=list)
    toxicity_types_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "smiles": self.smiles,
            "is_valid": self.is_valid,
            "total_alerts": self.total_alerts,
            "toxicity_risk": self.toxicity_risk,
            "risk_score": self.risk_score,
            "found_patterns": self.found_patterns,
            "toxicity_types": self.toxicity_types_found,
            "recommendations": self.recommendations,
            "matches": [
                {
                    "pattern": m.pattern.name,
                    "count": m.count,
                    "severity": m.pattern.severity,
                }
                for m in self.matches
            ],
        }


class ToxicophoreAnalyzer:
    """
    독성 작용기 분석기

    SMARTS 패턴을 사용하여 분자 구조에서 독성 작용기를 탐지합니다.
    """

    # 6개 핵심 독성 작용기 + 추가 패턴
    TOXICOPHORE_PATTERNS = [
        # 1. Quinone - 반응성 산소종, 미토콘드리아 독성
        ToxicophorePattern(
            name="Quinone",
            smarts="[#6]1=[#6]-[#6](=[O])-[#6]=[#6]-[#6](=[O])-1",
            toxicity_types=[
                ToxicityType.HEPATOTOXICITY,
                ToxicityType.REACTIVE_METABOLITE,
            ],
            severity=4,
            description="Quinone structure - generates reactive oxygen species",
            reference="Monks et al., 1992",
        ),
        # 2. Epoxide - DNA 알킬화, 돌연변이 유발
        ToxicophorePattern(
            name="Epoxide",
            smarts="[#6]1[#8][#6]1",
            toxicity_types=[
                ToxicityType.GENOTOXICITY,
                ToxicityType.REACTIVE_METABOLITE,
            ],
            severity=5,
            description="Epoxide ring - DNA alkylation risk",
            reference="Guengerich, 2003",
        ),
        # 3. Nitroaromatic - 간독성, 발암성
        ToxicophorePattern(
            name="Nitroaromatic",
            smarts="[#6;R1]1[#6;R1][#6;R1][#6;R1][#6;R1][#6;R1]1[N+](=O)[O-]",
            toxicity_types=[ToxicityType.HEPATOTOXICITY, ToxicityType.GENOTOXICITY],
            severity=4,
            description="Nitroaromatic compound - reductive metabolism to reactive intermediates",
            reference="Boelsterli, 2002",
        ),
        # 단순화된 Nitro 패턴
        ToxicophorePattern(
            name="Nitro Group",
            smarts="[N+](=O)[O-]",
            toxicity_types=[ToxicityType.HEPATOTOXICITY],
            severity=3,
            description="Nitro group - potential hepatotoxicity",
            reference="Kazius, 2005",
        ),
        # 4. Halogenated Aromatic - 갑상선 독성, 간독성
        ToxicophorePattern(
            name="Polyhalogenated Aromatic",
            smarts="c1cc([F,Cl,Br,I])c([F,Cl,Br,I])cc1",
            toxicity_types=[ToxicityType.THYROID_TOXICITY, ToxicityType.HEPATOTOXICITY],
            severity=3,
            description="Polyhalogenated aromatic - thyroid disruption, bioaccumulation",
            reference="Grimm et al., 2015",
        ),
        # 5. Michael Acceptor - 단백질 공유결합, 피부 감작
        ToxicophorePattern(
            name="Michael Acceptor (alpha,beta-unsaturated carbonyl)",
            smarts="[#6]=[#6]-[#6](=[O])",
            toxicity_types=[
                ToxicityType.SKIN_SENSITIZATION,
                ToxicityType.REACTIVE_METABOLITE,
            ],
            severity=4,
            description="Michael acceptor - reacts with nucleophilic amino acids",
            reference="Enoch et al., 2011",
        ),
        # 6. Acyl Halide - 반응성 대사체
        ToxicophorePattern(
            name="Acyl Halide",
            smarts="[#6](=[O])[F,Cl,Br,I]",
            toxicity_types=[ToxicityType.REACTIVE_METABOLITE],
            severity=5,
            description="Acyl halide - highly reactive, acylating agent",
            reference="Thompson et al., 2012",
        ),
        # 추가 패턴
        # 7. Aromatic Amine - 발암성
        ToxicophorePattern(
            name="Aromatic Amine",
            smarts="c[NH2]",
            toxicity_types=[ToxicityType.GENOTOXICITY],
            severity=3,
            description="Primary aromatic amine - metabolic activation to carcinogens",
            reference="Benigni & Bossa, 2011",
        ),
        # 8. Hydrazine - 간독성
        ToxicophorePattern(
            name="Hydrazine",
            smarts="[#7]-[#7]",
            toxicity_types=[ToxicityType.HEPATOTOXICITY],
            severity=3,
            description="Hydrazine group - hepatotoxicity risk",
            reference="Timbrell et al., 1982",
        ),
        # 9. Aldehyde - 반응성
        ToxicophorePattern(
            name="Aldehyde",
            smarts="[#6;H1](=[O])",
            toxicity_types=[
                ToxicityType.REACTIVE_METABOLITE,
                ToxicityType.SKIN_SENSITIZATION,
            ],
            severity=3,
            description="Aldehyde group - protein adduct formation",
            reference="LoPachin & Gavin, 2014",
        ),
        # 10. Thiophene - CYP450 매개 간독성
        ToxicophorePattern(
            name="Thiophene",
            smarts="c1ccsc1",
            toxicity_types=[ToxicityType.HEPATOTOXICITY],
            severity=2,
            description="Thiophene ring - metabolic activation by CYP450",
            reference="Dalvie et al., 2002",
        ),
    ]

    def __init__(self):
        """Initialize analyzer with compiled SMARTS patterns."""
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for ToxicophoreAnalyzer")

        self.patterns = []
        for pattern in self.TOXICOPHORE_PATTERNS:
            try:
                mol = Chem.MolFromSmarts(pattern.smarts)
                if mol is not None:
                    self.patterns.append((pattern, mol))
            except Exception as e:
                print(f"Warning: Failed to compile pattern {pattern.name}: {e}")

    def analyze(self, smiles: str) -> ToxicophoreResult:
        """
        SMILES 구조에서 독성 작용기 분석.

        Args:
            smiles: SMILES 문자열

        Returns:
            ToxicophoreResult
        """
        result = ToxicophoreResult(smiles=smiles)

        # SMILES 파싱
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result.is_valid = False
                result.toxicity_risk = "Unknown"
                result.recommendations.append("Invalid SMILES structure")
                return result
        except Exception:
            result.is_valid = False
            result.toxicity_risk = "Unknown"
            return result

        # 각 패턴 검사
        total_severity = 0

        for pattern, smarts_mol in self.patterns:
            matches = mol.GetSubstructMatches(smarts_mol)

            if matches:
                match_result = ToxicophoreMatch(
                    pattern=pattern, atom_indices=list(matches), count=len(matches)
                )
                result.matches.append(match_result)
                result.found_patterns.append(pattern.name)

                # 독성 유형 추가
                for tox_type in pattern.toxicity_types:
                    if tox_type.value not in result.toxicity_types_found:
                        result.toxicity_types_found.append(tox_type.value)

                total_severity += pattern.severity * len(matches)

        # 총 경고 수
        result.total_alerts = len(result.matches)

        # 리스크 점수 계산 (0-100)
        result.risk_score = min(100, total_severity * 10)

        # 리스크 수준 결정
        if result.risk_score >= 50:
            result.toxicity_risk = "High"
            result.recommendations.append(
                "Multiple structural alerts detected. Consider structural modification."
            )
        elif result.risk_score >= 20:
            result.toxicity_risk = "Medium"
            result.recommendations.append(
                "Some structural alerts found. Additional toxicity testing recommended."
            )
        else:
            result.toxicity_risk = "Low"
            if result.total_alerts == 0:
                result.recommendations.append("No structural alerts detected.")

        # 특정 패턴별 권고
        for match in result.matches:
            if match.pattern.severity >= 4:
                result.recommendations.append(
                    f"Alert: {match.pattern.name} - {match.pattern.description}"
                )

        return result

    def batch_analyze(self, smiles_list: List[str]) -> List[ToxicophoreResult]:
        """
        여러 SMILES 분석.

        Args:
            smiles_list: SMILES 문자열 리스트

        Returns:
            ToxicophoreResult 리스트
        """
        return [self.analyze(smiles) for smiles in smiles_list]

    def get_pattern_info(self, pattern_name: str) -> Optional[ToxicophorePattern]:
        """패턴 정보 반환."""
        for pattern, _ in self.patterns:
            if pattern.name == pattern_name:
                return pattern
        return None

    def list_patterns(self) -> List[str]:
        """모든 패턴 이름 반환."""
        return [p.name for p, _ in self.patterns]

    def summarize_toxicity_risk(self, result: ToxicophoreResult) -> Dict:
        """
        독성 리스크 요약.

        Args:
            result: ToxicophoreResult

        Returns:
            요약 딕셔너리
        """
        severity_breakdown = {}
        for match in result.matches:
            severity_breakdown[match.pattern.name] = {
                "count": match.count,
                "severity": match.pattern.severity,
                "total_impact": match.count * match.pattern.severity,
            }

        return {
            "overall_risk": result.toxicity_risk,
            "risk_score": result.risk_score,
            "total_alerts": result.total_alerts,
            "severity_breakdown": severity_breakdown,
            "toxicity_types": result.toxicity_types_found,
            "high_severity_patterns": [
                m.pattern.name for m in result.matches if m.pattern.severity >= 4
            ],
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_toxicophore_check(smiles: str) -> Dict:
    """
    빠른 독성 작용기 체크.

    Args:
        smiles: SMILES 문자열

    Returns:
        결과 딕셔너리
    """
    if not RDKIT_AVAILABLE:
        return {"error": "RDKit not available"}

    analyzer = ToxicophoreAnalyzer()
    result = analyzer.analyze(smiles)
    return result.to_dict()


def has_structural_alerts(smiles: str) -> bool:
    """
    구조적 경고가 있는지 확인.

    Args:
        smiles: SMILES 문자열

    Returns:
        True if any structural alerts found
    """
    if not RDKIT_AVAILABLE:
        return False

    analyzer = ToxicophoreAnalyzer()
    result = analyzer.analyze(smiles)
    return result.total_alerts > 0


# =============================================================================
# Main (Test)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Toxicophore Analyzer Test")
    print("=" * 60)

    if not RDKIT_AVAILABLE:
        print("Error: RDKit not available. Install with: pip install rdkit-pypi")
        exit(1)

    analyzer = ToxicophoreAnalyzer()

    # 테스트 화합물
    test_compounds = [
        # Safe compound (Aspirin)
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        # Nitroaromatic (Nitrobenzene)
        ("Nitrobenzene", "c1ccc(cc1)[N+](=O)[O-]"),
        # Epoxide containing
        ("Epoxide example", "C1OC1c2ccccc2"),
        # Michael acceptor (Acrolein)
        ("Acrolein", "C=CC=O"),
        # Polyhalogenated
        ("Dichlorobenzene", "c1ccc(Cl)c(Cl)c1"),
    ]

    for name, smiles in test_compounds:
        print(f"\n>>> {name}")
        print(f"    SMILES: {smiles}")

        result = analyzer.analyze(smiles)

        print(f"    Risk Level: {result.toxicity_risk}")
        print(f"    Risk Score: {result.risk_score:.0f}/100")
        print(f"    Total Alerts: {result.total_alerts}")

        if result.found_patterns:
            print(f"    Patterns Found: {', '.join(result.found_patterns)}")
        if result.toxicity_types_found:
            print(f"    Toxicity Types: {', '.join(result.toxicity_types_found)}")

    print("\n" + "-" * 60)
    print(">>> Available Patterns:")
    for pattern_name in analyzer.list_patterns():
        print(f"    - {pattern_name}")

    print("\n" + "=" * 60)
    print("  Toxicophore Analyzer Test Complete!")
    print("=" * 60)
