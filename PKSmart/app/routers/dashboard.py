from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from .. import models, database, auth
from datetime import datetime

router = APIRouter(tags=["Dashboard"])
templates = Jinja2Templates(directory="app/templates")

# Mock Data (Moved to global scope for access by multiple routes)
MOCK_IND_REPORTS = [
    {
        "id": 101,
        "title": "IND Submission - Oncology Phase I",
        "date": datetime.now().strftime("%b %d, %Y"),
        "status": "Ready",
        "thumbnail_url": "https://via.placeholder.com/300x200/00487E/ffffff?text=Report+101",
        "description": "Comprehensive Investigational New Drug application for the novel oncology compound, including PK/PD modeling results and safety profile analysis."
    },
    {
        "id": 102,
        "title": "Toxicity Profile Analysis",
        "date": datetime.now().strftime("%b %d, %Y"),
        "status": "In Progress",
        "thumbnail_url": "https://via.placeholder.com/300x200/00487E/ffffff?text=Report+102",
        "description": "Detailed analysis of potential toxicity risks based on in-silico predictions and early-stage in-vitro data."
    },
    {
        "id": 103,
        "title": "FDA Safety Review Draft",
        "date": datetime.now().strftime("%b %d, %Y"),
        "status": "Draft",
        "thumbnail_url": "https://via.placeholder.com/300x200/00487E/ffffff?text=Report+103",
        "description": "Preliminary draft of the safety review section for FDA submission, adhering to the latest regulatory guidelines."
    },
    {
        "id": 104,
        "title": "Clinical Trial Protocol v2",
        "date": datetime.now().strftime("%b %d, %Y"),
        "status": "Review",
        "thumbnail_url": "https://via.placeholder.com/300x200/00487E/ffffff?text=Report+104",
        "description": "Updated clinical trial protocol incorporating feedback from the initial review board and optimization results."
    }
]

@router.get("/dashboard")
async def dashboard(
    request: Request,
    db: Session = Depends(database.get_db)
):
    # Auth Check
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login")
    
    try:
        scheme, _, param = token.partition(" ")
        current_user = await auth.get_current_user(token=param, db=db)
    except:
        return RedirectResponse(url="/login")

    # 1. PROJECTS Section (My Projects)
    projects = db.query(models.Project).filter(models.Project.owner_id == current_user.id).all()
    
    # 2. RESULTS Section (Grouped by Project)
    # Fetch projects that have at least one cohort (Simulation Result)
    # Using distinct to avoid duplicates if multiple cohorts exist
    results = db.query(models.Project).join(models.Cohort).filter(models.Project.owner_id == current_user.id).distinct().all()

    # 3. IND REPORT Section (Mock Data)
    ind_reports = MOCK_IND_REPORTS
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request, 
        "user": current_user, 
        "projects": projects,
        "results": results,
        "ind_reports": ind_reports,
        "sidebar_mode": "default" # Wide mode
    })

@router.get("/dashboard/reports/{report_id}")
async def view_report_detail(
    request: Request,
    report_id: int,
    db: Session = Depends(database.get_db)
):
    # Auth Check
    token = request.cookies.get("access_token")
    if not token: return RedirectResponse(url="/login")
    
    try:
        scheme, _, param = token.partition(" ")
        current_user = await auth.get_current_user(token=param, db=db)
    except:
        return RedirectResponse(url="/login")

    # Find the report from Mock Data
    report = next((item for item in MOCK_IND_REPORTS if item["id"] == report_id), None)
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
        
    return templates.TemplateResponse("ind_report_detail.html", {
        "request": request,
        "user": current_user,
        "report": report,
        "sidebar_mode": "compact"
    })

@router.get("/dashboard/share")
async def share_summary(
    request: Request,
    db: Session = Depends(database.get_db)
):
    token = request.cookies.get("access_token")
    if not token: return RedirectResponse(url="/login")
    
    try:
        scheme, _, param = token.partition(" ")
        current_user = await auth.get_current_user(token=param, db=db)
    except:
        return RedirectResponse(url="/login")
    
    # Get shared projects (showing all for demo)
    shared_items = db.query(models.Project).all()
    
    return templates.TemplateResponse("share_summary.html", {
        "request": request,
        "user": current_user,
        "shared_items": shared_items,
        "sidebar_mode": "default"
    })

@router.get("/dashboard/users")
async def user_management(
    request: Request,
    db: Session = Depends(database.get_db)
):
    token = request.cookies.get("access_token")
    if not token: return RedirectResponse(url="/login")
    
    try:
        scheme, _, param = token.partition(" ")
        current_user = await auth.get_current_user(token=param, db=db)
    except:
        return RedirectResponse(url="/login")
    
    # Get all users
    users = db.query(models.User).all()
    
    return templates.TemplateResponse("user_management.html", {
        "request": request,
        "user": current_user,
        "users": users,
        "sidebar_mode": "default"
    })
