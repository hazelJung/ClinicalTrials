from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd
import sys
import base64
import io
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Libraries
try:
    import pubchempy as pcp
    from rdkit import Chem
    from rdkit.Chem import Draw
except ImportError as e:
    print(f"Warning: Missing dependencies: {e}")
    pcp = None
    Chem = None
    Draw = None

# Import QSAR Predictor
try:
    from src.models.qsar_predictor import QSARPredictor
except ImportError:
    print("Warning: Could not import QSARPredictor. Using mock mode if needed.")
    QSARPredictor = None

router = APIRouter(
    prefix="/api/qsar",
    tags=["qsar"],
    responses={404: {"description": "Not found"}},
)

# Initialize QSAR Engine (Singleton)
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None and QSARPredictor:
        _predictor = QSARPredictor(
            model_dir=PROJECT_ROOT / "models" / "qsar",
            auto_load=True
        )
    return _predictor

class PredictionRequest(BaseModel):
    smiles: str
    threshold: Optional[float] = 0.20

class EndpointResult(BaseModel):
    name: str
    prob: float
    positive: bool

class RiskAssessment(BaseModel):
    level: str  # LOW, MEDIUM, HIGH, CRITICAL
    color: str

class PredictionResponse(BaseModel):
    smiles: str
    drug_name: Optional[str] = None
    structure_image: Optional[str] = None  # Base64 SVG
    endpoints: List[EndpointResult]
    positiveCount: int
    risk: RiskAssessment
    threshold: float
    error: Optional[str] = None

@router.post("/predict", response_model=PredictionResponse)
async def predict_toxicity(request: PredictionRequest):
    """
    Predict toxicity using QSAR models.
    """
    predictor = get_predictor()
    
    if not predictor:
        return PredictionResponse(
            smiles=request.smiles,
            endpoints=[],
            positiveCount=0,
            risk=RiskAssessment(level="UNKNOWN", color="gray"),
            threshold=request.threshold,
            error="QSAR engine not initialized"
        )
    
    try:
        # 1. Identify Drug Name (PubChem)
        drug_name = "Unknown Compound"
        if pcp:
            try:
                compounds = pcp.get_compounds(request.smiles, namespace='smiles')
                if compounds:
                    drug_name = compounds[0].synonyms[0] if compounds[0].synonyms else compounds[0].iupac_name
            except Exception:
                pass # Fail silently for name lookup
        
        # 2. Generate Structure Image (RDKit)
        structure_image = None
        if Chem and Draw:
            try:
                mol = Chem.MolFromSmiles(request.smiles)
                if mol:
                    # Generate SVG
                    drawer = Draw.MolDraw2DSVG(300, 300)
                    drawer.DrawMolecule(mol)
                    drawer.FinishDrawing()
                    svg = drawer.GetDrawingText()
                    # Convert to base64 for safe transport (optional for SVG but good practice)
                    # For SVG we can just send the string, but let's base64 it to avoid JSON escaping issues
                    structure_image = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
            except Exception as e:
                print(f"Image generation failed: {e}")

        # 3. Run prediction
        results_dict = predictor.predict_multiple_endpoints(request.smiles)
        
        # Format results
        endpoints = []
        positive_count = 0
        
        for name, res in results_dict.items():
            # Apply custom threshold
            is_positive = res.probability >= request.threshold
            if is_positive:
                positive_count += 1
            
            endpoints.append(EndpointResult(
                name=name,
                prob=float(res.probability),
                positive=is_positive
            ))
            
        # Determine risk level
        if positive_count <= 1:
            risk = RiskAssessment(level="LOW", color="green")
        elif positive_count <= 3:
            risk = RiskAssessment(level="MEDIUM", color="yellow")
        elif positive_count <= 5:
            risk = RiskAssessment(level="HIGH", color="orange")
        else:
            risk = RiskAssessment(level="CRITICAL", color="red")
            
        return PredictionResponse(
            smiles=request.smiles,
            drug_name=drug_name,
            structure_image=structure_image,
            endpoints=endpoints,
            positiveCount=positive_count,
            risk=risk,
            threshold=request.threshold
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return PredictionResponse(
            smiles=request.smiles,
            endpoints=[],
            positiveCount=0,
            risk=RiskAssessment(level="ERROR", color="red"),
            threshold=request.threshold,
            error=str(e)
        )
