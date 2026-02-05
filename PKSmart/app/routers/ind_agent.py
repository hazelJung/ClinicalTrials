from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy.orm import Session
from app.services.ind_generator import INDGeneratorService
import os
import io
import base64
from datetime import date


from app import models, database

# Templates
templates = Jinja2Templates(directory="app/templates")

router = APIRouter(tags=["IND Agent"])

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# === Pydantic Models ===
class INDRequest(BaseModel):
    # === APPLICANT INFORMATION ===
    applicant_name: Optional[str] = Field("", description="Name of the applicant/sponsor")
    applicant_address: Optional[str] = Field("", description="Address of the applicant")
    applicant_phone: Optional[str] = Field("", description="Phone number of the applicant")
    application_date: Optional[str] = Field(None, description="Date of application submission")

    # === INVESTIGATOR INFORMATION ===
    pi_name: Optional[str] = Field("", description="Name of Principal Investigator")
    pi_credentials: Optional[str] = Field("", description="Credentials of PI")
    institution_name: Optional[str] = Field("", description="Name of research institution")
    institution_address: Optional[str] = Field("", description="Address of research institution")
    irb_name: Optional[str] = Field("", description="Name of IRB/Ethics Committee")

    # === DRUG CANDIDATE INFORMATION ===
    drug_name: str = Field(..., description="Name of the drug candidate")
    drug_class: Optional[str] = Field("Small Molecule", description="Class of drug")
    smiles: str = Field(..., description="SMILES string or sequence")
    indication: str = Field(..., description="Target disease indication")
    mechanism: Optional[str] = Field("", description="Mechanism of Action")
    dose_mg: float = Field(..., gt=0, description="Proposed dose in mg")
    dosing_regimen: Optional[str] = Field("QD", description="Dosing frequency")
    target_population: Optional[str] = Field("Adult patients", description="Target patient population")

    # === CLINICAL TRIAL INFORMATION ===
    clinical_phase: Optional[str] = Field("1", description="Clinical trial phase (1, 2, or 3)")
    expected_patients: Optional[str] = Field("30", description="Expected number of patients")
    study_duration: Optional[str] = Field("12 weeks", description="Planned study duration")

    # === PK DATA (from mPBPK model) ===
    cmax: float = Field(..., description="Predicted Cmax (ng/mL)")
    auc: float = Field(..., description="Predicted AUC (ng*h/mL)")
    t_half: float = Field(..., description="Terminal Half-life (hours)")
    to_trough: Optional[float] = Field(0.0, description="Target Occupancy at Trough (%)")
    vss: Optional[float] = Field(0.0, description="Volume of Distribution (L/kg)")
    
    # === ANIMAL PK (Hidden fields) ===
    rat_cl: Optional[float] = None
    rat_vss: Optional[float] = None
    rat_fup: Optional[float] = None
    dog_cl: Optional[float] = None
    dog_vss: Optional[float] = None
    dog_fup: Optional[float] = None
    monkey_cl: Optional[float] = None
    monkey_vss: Optional[float] = None
    monkey_fup: Optional[float] = None

    # === TOXICITY/SAFETY DATA (from QSAR model) ===
    qsar_results_formatted: str = Field(..., description="Formatted string of QSAR results")
    herg_margin: float = Field(..., description="hERG Safety Margin (x-fold)")
    hepato_risk: str = Field(..., description="Hepatotoxicity Risk (High/Medium/Low)")
    ames_result: Optional[str] = Field("Negative", description="Ames Test Result")
    overall_score: float = Field(..., ge=0, le=100, description="Overall Safety Score (0-100)")
    risk_category: Optional[str] = Field("Unknown", description="Risk Category")
    
    # Project reference for linking report to project
    project_id: Optional[int] = Field(None, description="Project ID to link the report to")


# === Page Routes ===

@router.get("/ind-generator", response_class=HTMLResponse)
async def ind_generator_page(
    request: Request,
    pred_id: Optional[int] = None,
    cohort_id: Optional[int] = None,
    project_id: Optional[int] = None,
    db: Session = Depends(get_db) # Using local get_db since we are in a router file not fully integrated with root dep yet? Or use database.get_db
):
    """Render the IND Generator page."""
    context = {"request": request, "sidebar_mode": "default", "project_id": project_id, "form_data": {}}
    
    # Fetch cohort data if cohort_id is provided
    cohort_data = None
    if cohort_id:
        cohort = db.query(models.Cohort).filter(models.Cohort.id == cohort_id).first()
        if cohort:
            demographics = cohort.demographics or {}
            # results_json can be a list (simulation timeseries) or dict (summary stats)
            # Handle both cases
            results_raw = cohort.results_json
            mean_to = None
            if isinstance(results_raw, dict):
                results = results_raw
                mean_to = results.get("mean_to")
            elif isinstance(results_raw, list) and len(results_raw) > 0:
                # Calculate average Target Occupancy from simulation results
                to_values = [r.get("mean_to", 0) for r in results_raw if r.get("mean_to") is not None]
                if to_values:
                    mean_to = sum(to_values) / len(to_values)
                results = {}
            else:
                results = {}
            
            # Handle age - stored as array [min, max] or individual keys
            age_data = demographics.get("age", [18, 65])
            if isinstance(age_data, list) and len(age_data) >= 2:
                age_min, age_max = age_data[0], age_data[1]
            else:
                age_min = demographics.get("age_min", 18)
                age_max = demographics.get("age_max", 65)
            
            cohort_data = {
                "name": cohort.name,
                "description": cohort.description,
                "population": demographics.get("pop", "Mixed"),
                "gender": demographics.get("gender", "BOTH"),
                "age_min": age_min,
                "age_max": age_max,
                # Handle n_subjects key - stored as 'n' in cohorts.py
                "n_subjects": demographics.get("n", demographics.get("n_subjects", 100)),
                "female_ratio": demographics.get("female_ratio", 0.5),
                # Simulation results
                "mean_cmax": results.get("mean_cmax"),
                "success_rate": results.get("success_rate"),
                "phenotype_distribution": results.get("phenotypes", {}),
                # Target Occupancy from simulation
                "mean_to": mean_to,
            }
            context["cohort_data"] = cohort_data
    
    if pred_id:
        # Fetch prediction data
        prediction = db.query(models.Prediction).filter(models.Prediction.id == pred_id).first()
        if prediction and prediction.result_json:
            result_json = prediction.result_json
            # Pre-calc Summary of QSAR
            qsar_data = result_json.get("tox21", {})
            pos_count = 0
            total_count = 0
            risk_details = []
            
            for k, v in qsar_data.items():
                prob = v if isinstance(v, float) else v.get("probability", 0)
                total_count += 1
                if prob >= 0.7:
                    pos_count += 1
                    risk_details.append(f"{k}: HIGH ({prob:.2f})")
                elif prob >= 0.3:
                     risk_details.append(f"{k}: MED ({prob:.2f})")
            
            qsar_summary = ", ".join(risk_details) if risk_details else "No significant risks identified."
            
            # Map Prediction to IND Form Data
            pk = result_json.get("pk", {})
            
            # Calculate Cmax and AUC if not directly available
            # Use 1-compartment PK model assumptions:
            # Assume standard dose of 100mg and 70kg body weight
            dose_mg = 100  # Standard dose assumption
            body_weight_kg = 70  # Standard body weight
            
            # Get CL and Vss
            cl_ml_min_kg = pk.get("human_CL_mL_min_kg_linear", 0)  # mL/min/kg
            vss_l_kg = pk.get("human_VDss_L_kg_linear", 0)  # L/kg
            
            # Calculate AUC: AUC = (Dose in mg) / (CL in L/h)
            # CL in L/h = CL(mL/min/kg) * 60 / 1000 * BodyWeight
            if cl_ml_min_kg and cl_ml_min_kg > 0:
                cl_l_h = cl_ml_min_kg * 60 / 1000 * body_weight_kg  # Convert to L/h
                auc_calc = (dose_mg * 1000) / cl_l_h  # ng·h/mL (dose in µg / L/h)
            else:
                auc_calc = None
            
            # Calculate Cmax: Cmax ≈ Dose / Vss (simplified 1-compartment)
            # Vss in L = Vss(L/kg) * BodyWeight
            if vss_l_kg and vss_l_kg > 0:
                vss_l = vss_l_kg * body_weight_kg  # Convert to L
                cmax_calc = (dose_mg * 1000) / vss_l  # ng/mL (dose in µg / L)
            else:
                cmax_calc = None
            
            # Use calculated values if direct values not available
            cmax_value = pk.get("human_Cmax_ng_mL_linear") or cmax_calc
            auc_value = pk.get("human_AUC_ng_h_mL_linear") or auc_calc
            
            form_data = {
                "smiles": prediction.smiles,
                "cmax": cmax_value,
                "auc": auc_value,
                "t_half": pk.get("human_thalf_linear"),
                "vss": pk.get("human_VDss_L_kg_linear"),
                "gh_ld50": result_json.get("ld50", {}).get("value"),
                "qsar_results_formatted": qsar_summary,
                "overall_score": 100 - (pos_count * 10),
                
                # Animal PK
                "rat_cl": pk.get("rat_CL_mL_min_kg_linear"),
                "rat_vss": pk.get("rat_VDss_L_kg_linear"),
                "rat_fup": (pk.get("rat_fup", 0) * 100) if pk.get("rat_fup") else None,
                
                "dog_cl": pk.get("dog_CL_mL_min_kg_linear"),
                "dog_vss": pk.get("dog_VDss_L_kg_linear"),
                "dog_fup": (pk.get("dog_fup", 0) * 100) if pk.get("dog_fup") else None,
                
                "monkey_cl": pk.get("monkey_CL_mL_min_kg_linear"),
                "monkey_vss": pk.get("monkey_VDss_L_kg_linear"),
                "monkey_fup": (pk.get("monkey_fup", 0) * 100) if pk.get("monkey_fup") else None,
            }
            
            # Add cohort data to form_data if available
            if cohort_data:
                form_data["cohort_name"] = cohort_data["name"]
                form_data["target_population"] = f"{cohort_data['population']} population, {cohort_data['gender']}, age {cohort_data['age_min']}-{cohort_data['age_max']}"
                form_data["expected_patients"] = str(cohort_data["n_subjects"])
                form_data["simulation_mean_cmax"] = cohort_data.get("mean_cmax")
                form_data["simulation_success_rate"] = cohort_data.get("success_rate")
                # Target Occupancy from cohort simulation (uses 'to_trough' key for template compatibility)
                form_data["to_trough"] = cohort_data.get("mean_to")
            
            context["form_data"] = form_data
    
    return templates.TemplateResponse("ind_generator.html", context)



# === API Routes ===

@router.post("/api/ind/generate")
async def generate_ind(
    request: INDRequest,
    db: Session = Depends(get_db)
):
    """Generate a comprehensive FDA IND application document."""
    service = INDGeneratorService()
    
    data = request.model_dump()
    
    if not data.get("application_date"):
        data["application_date"] = date.today().strftime("%Y-%m-%d")

    
    placeholder_fields = {
        "applicant_name": "[Applicant Name - To Be Filled]",
        "applicant_address": "[Applicant Address - To Be Filled]",
        "applicant_phone": "[Phone Number - To Be Filled]",
        "pi_name": "[Principal Investigator Name - To Be Filled]",
        "pi_credentials": "[Credentials - To Be Filled]",
        "institution_name": "[Research Institution - To Be Filled]",
        "institution_address": "[Institution Address - To Be Filled]",
        "irb_name": "[IRB Name - To Be Filled]",
        "mechanism": "[Mechanism of Action - To Be Determined]",
    }
    
    for field, placeholder in placeholder_fields.items():
        if not data.get(field) or data[field].strip() == "":
            data[field] = placeholder
    
    try:
        result = service.generate_ind_draft(data)
        
        # Save to DB
        new_report = models.INDReport(
            project_id=data.get("project_id"),
            title=f"IND Application: {data['drug_name']}",
            file_path=result["filename"], # relative or filename
            status="Completed",
            meta_data=data
        )
        db.add(new_report)
        db.commit()
        db.refresh(new_report)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/ind/download/{filename}")
async def download_ind(filename: str):
    """Download a generated IND document."""
    # Locate data directory relative to app root
    # Assumption: run from PKSmart root
    base_dir = os.getcwd()
    file_path = os.path.join(base_dir, "data", "generated_ind", filename)

    if not os.path.exists(file_path):
        # Try finding it relative to this file if cwd is different
        # This fallback mirrors ind_generator.py logic
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.dirname(current_file_dir)
        root_dir = os.path.dirname(app_dir)
        file_path_alt = os.path.join(root_dir, "data", "generated_ind", filename)
        
        if os.path.exists(file_path_alt):
            file_path = file_path_alt
        else:
            raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
    )


@router.get("/api/molecule/image")
async def get_molecule_image(smiles: str, width: int = 400, height: int = 300):
    """Generate a 2D structure image from SMILES string."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise HTTPException(status_code=400, detail="Invalid SMILES string")
        
        # Generate image
        img = Draw.MolToImage(mol, size=(width, height))
        
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return StreamingResponse(
            img_buffer,
            media_type="image/png",
            headers={"Cache-Control": "max-age=3600"}  # Cache for 1 hour
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="RDKit is not installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate image: {str(e)}")

