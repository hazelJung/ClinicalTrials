
from fastapi import APIRouter, Depends, HTTPException, Request, Form
from sqlalchemy.orm import Session
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
import json

from .. import models, database, auth

router = APIRouter(
    prefix="/projects/{project_id}/cohorts",
    tags=["cohorts"],
    responses={404: {"description": "Not found"}},
)

templates = Jinja2Templates(directory="app/templates")

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/new")
async def create_cohort_form(request: Request, project_id: int, pred_id: int = None, db: Session = Depends(database.get_db)):
    # Auth Check (Cookie based)
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login")
    
    try:
        scheme, _, param = token.partition(" ")
        current_user = await auth.get_current_user(token=param, db=db)
    except:
        return RedirectResponse(url="/login")
        
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    # If pred_id is provided, fetch the prediction to pre-fill the form
    prediction = None
    if pred_id:
        prediction = db.query(models.Prediction).filter(models.Prediction.id == pred_id).first()
    
    # Fetch existing cohorts for the list view (Defer heavy JSON results)
    from sqlalchemy.orm import defer
    cohorts = db.query(models.Cohort).filter(models.Cohort.project_id == project_id)\
        .options(defer(models.Cohort.results_json))\
        .order_by(models.Cohort.created_at.desc())\
        .limit(20).all()
        
    return templates.TemplateResponse("create_cohort.html", {
        "request": request,
        "project": project,
        "user": current_user,
        "prediction": prediction,
        "cohorts": cohorts,
        "cohorts": cohorts,
        "sidebar_mode": "compact",
        "is_compact": True
    })

from .. import inference

@router.post("")
async def create_cohort(
    request: Request, 
    project_id: int, 
    pop: str = Form(...),
    n_subjects: int = Form(...),
    age_min: int = Form(...),
    age_max: int = Form(...),
    female_ratio: float = Form(...),
    cl: float = Form(...),
    vd: float = Form(...),
    mw: float = Form(...),
    fu: float = Form(...),
    phase: int = Form(1), # Default to Phase 1
    # Advanced Settings (New)
    enzyme: str = Form("CYP2D6"),
    pheno_pm: float = Form(5.0),
    pheno_im: float = Form(25.0),
    pheno_em: float = Form(65.0),
    pheno_um: float = Form(5.0),
    bmi_min: float = Form(18.5),
    bmi_max: float = Form(30.0),
    bmi_dist: str = Form("normal"),
    renal_normal: float = Form(100.0),
    renal_mild: float = Form(0.0),
    renal_mod: float = Form(0.0),
    renal_sev: float = Form(0.0),
    hepatic_grade: str = Form("normal"),
    concomitant_meds: str = Form(""),
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
        
    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    if not project: raise HTTPException(status_code=404, detail="Project not found")

    # 1. Run Simulation
    # Note: In production, this should be a background task (Celery/Redis Queue)
    # For prototype, we run it synchronously (might take 5-10s)
    
    print(f"Starting Simulation for Project {project_id}...")
    results = inference.run_cohort_simulation(
        cl_ml_min_kg=cl,
        vd_l_kg=vd,
        mw_g_mol=mw,
        fup=fu,
        pop=pop,
        n_subjects=n_subjects,
        age_min=age_min,
        age_max=age_max,
        female_ratio=female_ratio
    )
    
    if isinstance(results, dict) and "error" in results:
        # Handle error (flash message ideally, but simpler here)
        print(f"Simulation Failed: {results['error']}")
        # For now redirect back with error in query param?
        return RedirectResponse(url=f"/projects/{project_id}/cohorts/new?error=sim_failed", status_code=303)

    # 2. Save to DB
    # Generate a name
    count = db.query(models.Cohort).filter(models.Cohort.project_id == project_id).count()
    name = f"Cohort #{count + 1} ({pop}, n={n_subjects})"
    
    # Structure Advanced Settings
    advanced_settings = {
        "genetic": {
            "enzyme": enzyme,
            "distribution": {"PM": pheno_pm, "IM": pheno_im, "EM": pheno_em, "UM": pheno_um}
        },
        "body": {
            "bmi_range": [bmi_min, bmi_max],
            "distribution": bmi_dist
        },
        "organ": {
            "renal": {"normal": renal_normal, "mild": renal_mild, "mod": renal_mod, "sev": renal_sev},
            "hepatic": hepatic_grade
        },
        "ddi": {
            # Parse comma-separated string if present
            "meds": [m.strip() for m in concomitant_meds.split(",") if m.strip()]
        }
    }

    demographics = {
        "pop": pop, 
        "n": n_subjects, 
        "age": [age_min, age_max], 
        "female_ratio": female_ratio,
        "phase": phase,
        "advanced": advanced_settings
    }
    drug_params = {
        "cl": cl, "vd": vd, "mw": mw, "fu": fu
    }
    
    # We might want to compress results or just store full JSON (sqlite handles JSON)
    new_cohort = models.Cohort(
        project_id=project_id,
        name=name,
        description=f"Virtual cohort simulation for {pop}",
        demographics=demographics,
        drug_params=drug_params,
        results_json=results
    )
    
    db.add(new_cohort)
    db.commit()
    db.refresh(new_cohort)
    
    # Redirect to the new cohort detail page (Phase specific views will be handled in Phase 2)
    return RedirectResponse(url=f"/projects/{project_id}/cohorts/{new_cohort.id}", status_code=303)

from ..inference import run_cohort_simulation, run_dose_response_analysis

@router.get("/{cohort_id}")
async def get_cohort_detail(request: Request, project_id: int, cohort_id: int, db: Session = Depends(database.get_db)):
    # Auth Check
    token = request.cookies.get("access_token")
    if not token: return RedirectResponse(url="/login")
    try:
        scheme, _, param = token.partition(" ")
        current_user = await auth.get_current_user(token=param, db=db)
    except:
        return RedirectResponse(url="/login")

    project = db.query(models.Project).filter(models.Project.id == project_id).first()
    cohort = db.query(models.Cohort).filter(models.Cohort.id == cohort_id).first()
    
    if not cohort: raise HTTPException(status_code=404, detail="Cohort not found")
    
    # Calculate Dose Response Data
    dose_response_data = []
    if cohort.drug_params:
        try:
            # Check keys (create_cohort vs db model compatibility)
            # Default values if missing
            cl = cohort.drug_params.get("cl", 2.0)
            vd = cohort.drug_params.get("vd", 1.5)
            mw = cohort.drug_params.get("mw", 150.0)
            fu = cohort.drug_params.get("fu", 0.8)
            project_dose = 100.0 # Default/Placeholder if not stored (Project might have it? no, cohort sim specific)
            # Actually create_cohort.html didn't explicitly ask for Dose, it used 'Drug Params' from prediction or manual.
            # Mpbpk engine defaults dose to 100mg if not passed.
            # Let's use 100 as base or if we find it.
            
            dose_response_data = run_dose_response_analysis(
                cl_ml_min_kg=float(cl),
                vd_l_kg=float(vd),
                mw_g_mol=float(mw),
                fup=float(fu),
                base_dose_mg=100.0
            )
        except Exception as e:
            print(f"Dose Analysis Failed: {e}")

    return templates.TemplateResponse("cohort_detail.html", {
        "request": request,
        "project": project,
        "cohort": cohort,
        "user": current_user,
        "dose_response_data": dose_response_data,
        "dose_response_data": dose_response_data,
        "sidebar_mode": "compact",
        "is_compact": True
    })

