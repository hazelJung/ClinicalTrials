import os
import json
import sys
from app.inference import engine

smiles = "C1=COC(=C1)CNC2=CC(=C(C=C2C(=O)O)S(=O)(=O)N)Cl" # Furosemide
print(f"Testing SMILES: {smiles}")

try:
    result = engine.predict(smiles)
    # Filter for Tox21 and LD50 mostly as user was worried about those
    summary = {
        "smiles": result["smiles"],
        "ld50": result["ld50"],
        "pk_human_cl": result["pk"].get("human_CL_mL_min_kg_linear"),
        "tox21_high_risk": [t for t, r in result["tox21"].items() if r.get("risk") == "HIGH"]
    }
    print("--- Prediction Results ---")
    print(json.dumps(summary, indent=2))
except Exception as e:
    print(f"Error: {e}")
