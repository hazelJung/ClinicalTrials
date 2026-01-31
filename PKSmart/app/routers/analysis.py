from fastapi import APIRouter, Depends, HTTPException, status, Request, Form
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional
from .. import schemas, models, database, auth
from ..inference import engine as ai_engine
import json

router = APIRouter(tags=["Analysis"])

@router.post("/projects/{project_id}/predict")
async def run_analysis(
    request: Request,
    project_id: int,
    smiles: str = Form(...),
    db: Session = Depends(database.get_db)
):
    # Auth
    token = request.cookies.get("access_token")
    if not token: return RedirectResponse(url="/login")
    scheme, _, param = token.partition(" ")
    current_user = await auth.get_current_user(token=param, db=db)
    
    # Check Project Ownership
    project = db.query(models.Project).filter(models.Project.id == project_id, models.Project.owner_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Run Inference
    try:
        prediction_result = ai_engine.predict(smiles)
        
        if "error" in prediction_result:
             # Basic error handling - in prod pass error to UI
             print(f"Prediction Error: {prediction_result['error']}")
             return RedirectResponse(url=f"/projects/{project_id}", status_code=303)

        # Save to DB
        new_prediction = models.Prediction(
            project_id=project_id,
            smiles=smiles,
            result_json=prediction_result  # SQLAlchemy handles JSON serialization for SQLite (supported in modern versions/drivers) or we dump
        )
        
        # SQLite with SQLAlchemy sometimes needs explicit casting if not using JSON type engine support perfectly
        # But let's rely on SQLAlchemy 'JSON' type which usually works or creates TEXT
        
        db.add(new_prediction)
        db.commit()
        
    except Exception as e:
        print(f"Analysis Failed: {e}")
        
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)
