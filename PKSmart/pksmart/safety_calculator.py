"""
Safety Margin Calculator
========================
Calculates SM = IC50 / Cmax and Risk Levels.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

class RiskLevel(Enum):
    SAFE = "Safe"
    MODERATE = "Moderate Risk"
    CONCERN = "High Concern"
    UNKNOWN = "Unknown"

@dataclass
class SafetyResult:
    safety_margin: float
    risk_level: RiskLevel
    risk_color: str
    IC50_uM: float
    Cmax_uM: float
    recommendation: str

class SafetyMarginCalculator:
    def calculate_herg_safety(self, herg_IC50_uM: float, Cmax_uM: float) -> SafetyResult:
        if Cmax_uM <= 0 or herg_IC50_uM <= 0:
            return SafetyResult(0, RiskLevel.UNKNOWN, "gray", herg_IC50_uM, Cmax_uM, "Invalid")
            
        SM = herg_IC50_uM / Cmax_uM
        
        if SM >= 30:
            return SafetyResult(SM, RiskLevel.SAFE, "green", herg_IC50_uM, Cmax_uM, "Low Risk")
        elif SM >= 10:
            return SafetyResult(SM, RiskLevel.MODERATE, "yellow", herg_IC50_uM, Cmax_uM, "Moderate Risk")
        else:
            return SafetyResult(SM, RiskLevel.CONCERN, "red", herg_IC50_uM, Cmax_uM, "High Risk")

if __name__ == "__main__":
    calc = SafetyMarginCalculator()
    res = calc.calculate_herg_safety(100, 1)
    print(res.risk_level)
