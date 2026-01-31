"""
PKSmart Shared Types
====================
Consolidated data structures from various external modules (IVIVE, QSAR, Toxicophore, Safety)
to support Recommendation Engine without full dependency overhead.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

# =============================================================================
# Enums
# =============================================================================

class DrugType(Enum):
    SMALL_MOLECULE = "Small Molecule"
    ANTIBODY = "Antibody"
    UNKNOWN = "Unknown"

class OverallRisk(Enum):
    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"
    CRITICAL = "Critical Risk"
    UNKNOWN = "Unknown"

class RiskLevel(Enum):
    SAFE = "Safe"
    MODERATE = "Moderate Risk"
    CONCERN = "High Concern"
    UNKNOWN = "Unknown"

class ToxicityType(Enum):
    HEPATOTOXICITY = "Hepatotoxicity"
    CARDIOTOXICITY = "Cardiotoxicity"
    GENOTOXICITY = "Genotoxicity"
    SKIN_SENSITIZATION = "Skin Sensitization"
    THYROID_TOXICITY = "Thyroid Toxicity"
    REACTIVE_METABOLITE = "Reactive Metabolite"
    GENERAL = "General Toxicity"

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SafetyResult:
    """From safety_calculator.py"""
    safety_margin: float
    risk_level: RiskLevel
    risk_color: str
    IC50_uM: float
    Cmax_uM: float
    margin_to_safe: float
    therapeutic_index: float
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
class IVIVEResult:
    """From ivive_calculator.py"""
    CLint_uL_min_mg: float
    CLint_scaled: float
    CLh: float
    CLh_per_kg: float
    extraction_ratio: float
    half_life_hr: float
    Cmax_uM: float
    AUC: float
    model_used: str = "well_stirred"

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

@dataclass
class ToxicophorePattern:
    """From toxicophore_analyzer.py"""
    name: str
    smarts: str
    toxicity_types: List[ToxicityType]
    severity: int
    description: str
    reference: str = ""

@dataclass
class ToxicophoreMatch:
    """From toxicophore_analyzer.py"""
    pattern: ToxicophorePattern
    atom_indices: List[Tuple[int, ...]]
    count: int

@dataclass
class ToxicophoreResult:
    """From toxicophore_analyzer.py"""
    smiles: str
    is_valid: bool = True
    matches: List[ToxicophoreMatch] = field(default_factory=list)
    total_alerts: int = 0
    toxicity_risk: str = "Low"
    risk_score: float = 0.0
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
        }

@dataclass
class QSARPrediction:
    """From qsar_predictor.py"""
    smiles: str
    is_valid: bool = True
    prediction: int = 0
    probability: float = 0.0
    in_ad: bool = True
    leverage: float = 0.0
    molecular_weight: float = 0.0
    descriptors: Dict[str, float] = field(default_factory=dict)
    model_endpoint: str = "general"
    confidence: str = "High"

    def to_dict(self) -> Dict:
        return {
            "smiles": self.smiles,
            "is_valid": self.is_valid,
            "prediction": self.prediction,
            "probability": self.probability,
            "in_ad": self.in_ad,
            "leverage": self.leverage,
            "molecular_weight": self.molecular_weight,
            "model_endpoint": self.model_endpoint,
            "confidence": self.confidence,
        }

# Defined in monte_carlo_simulator.py usually, but included here for forward reference if needed
@dataclass
class MonteCarloResult:
    """From monte_carlo_simulator.py"""
    cmax_mean: float
    cmax_std: float
    cmax_p95: float
    sm_mean: float
    sm_p5: float
    pct_high_exposure: float
    phenotypes: Dict[str, float]
    # Detailed arrays omitted for lightweight struct
    
    def to_dict(self) -> Dict:
        return {
            "cmax_mean": self.cmax_mean,
            "cmax_p95": self.cmax_p95,
            "sm_mean": self.sm_mean,
            "sm_p5": self.sm_p5,
            "pct_high_exposure": self.pct_high_exposure,
            "phenotypes": self.phenotypes
        }

@dataclass
class DrugSafetyResult:
    """From drug_safety_service.py"""
    drug_name: str
    smiles: str
    drug_type: DrugType
    molecular_weight: float
    overall_risk: OverallRisk
    overall_score: float
    overall_recommendation: str
    ivive_result: Optional[IVIVEResult] = None
    safety_result: Optional[SafetyResult] = None
    toxicophore_result: Optional[ToxicophoreResult] = None
    qsar_predictions: Dict[str, QSARPrediction] = field(default_factory=dict)
    monte_carlo_result: Optional[MonteCarloResult] = None
    critical_alerts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    analysis_components: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
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
            "safety_margin": self.safety_result.to_dict() if self.safety_result else None,
            "toxicophore": self.toxicophore_result.to_dict() if self.toxicophore_result else None,
            "qsar_predictions": {k: v.to_dict() for k, v in self.qsar_predictions.items()} if self.qsar_predictions else None,
            "monte_carlo": self.monte_carlo_result.to_dict() if self.monte_carlo_result else None,
            "alerts": {
                "critical": self.critical_alerts,
                "warnings": self.warnings,
                "notes": self.notes,
            },
            "analysis_components": self.analysis_components,
        }
