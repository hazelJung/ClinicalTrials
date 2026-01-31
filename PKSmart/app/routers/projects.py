from fastapi import APIRouter, Depends, HTTPException, status, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import Optional
from .. import schemas, models, database, auth

router = APIRouter(tags=["Projects"])
templates = Jinja2Templates(directory="app/templates")



@router.get("/projects/{project_id}/results")
async def view_project_results(
    request: Request,
    project_id: int,
    db: Session = Depends(database.get_db)
):
    token = request.cookies.get("access_token")
    if not token: return RedirectResponse(url="/login")
    
    try:
        scheme, _, param = token.partition(" ")
        current_user = await auth.get_current_user(token=param, db=db)
    except:
        return RedirectResponse(url="/login")
    
    project = db.query(models.Project).filter(models.Project.id == project_id, models.Project.owner_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get all cohorts for this project
    cohorts = db.query(models.Cohort).filter(models.Cohort.project_id == project_id).all()
    
    return templates.TemplateResponse("project_results.html", {
        "request": request,
        "user": current_user,
        "project": project,
        "cohorts": cohorts,
        "sidebar_mode": "compact", # Compact sidebar as requested
        "is_compact": True
    })

@router.post("/projects")
async def create_project(
    request: Request,
    title: str = Form(...),
    description: str = Form(None),
    db: Session = Depends(database.get_db)
):
    token = request.cookies.get("access_token")
    if not token: return RedirectResponse(url="/login")
    
    try:
        scheme, _, param = token.partition(" ")
        current_user = await auth.get_current_user(token=param, db=db)
    except:
        return RedirectResponse(url="/login")
    
    new_project = models.Project(title=title, description=description, owner_id=current_user.id)
    db.add(new_project)
    db.commit()
    
    return RedirectResponse(url="/dashboard", status_code=303)

@router.get("/projects/{project_id}")
async def view_project(
    request: Request,
    project_id: int,
    db: Session = Depends(database.get_db)
):
    token = request.cookies.get("access_token")
    if not token: return RedirectResponse(url="/login")
    
    try:
        scheme, _, param = token.partition(" ")
        current_user = await auth.get_current_user(token=param, db=db)
    except:
        return RedirectResponse(url="/login")
    
    project = db.query(models.Project).filter(models.Project.id == project_id, models.Project.owner_id == current_user.id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return templates.TemplateResponse("project_detail.html", {
        "request": request,
        "user": current_user,
        "project": project,
        "sidebar_mode": "compact",
        "is_compact": True
    })



@router.delete("/projects/{project_id}")
async def delete_project(
    request: Request,
    project_id: int,
    db: Session = Depends(database.get_db)
):
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        scheme, _, param = token.partition(" ")
        current_user = await auth.get_current_user(token=param, db=db)
    except:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Find project and ensure ownership
    project = db.query(models.Project).filter(
        models.Project.id == project_id, 
        models.Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(status_code=404, detail="Project not found or permission denied")
        
    db.delete(project)
    db.commit()
    
    return {"message": "Project deleted successfully"}
