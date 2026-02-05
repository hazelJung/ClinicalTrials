import os

# =============================================================================
# FDA IND APPLICATION PROMPT TEMPLATES
# =============================================================================
# Complete IND Application Structure based on 21 CFR 312.23
# =============================================================================

# =============================================================================
# SYSTEM PROMPTS - Strong directive language to prevent review behavior
# =============================================================================

SYSTEM_PROMPT_LLAMA = """
YOU ARE A REGULATORY DOCUMENT WRITER. NOT A REVIEWER OR CRITIC.

=== CRITICAL INSTRUCTIONS ===
1. DO NOT analyze, critique, or provide feedback on this request
2. DO NOT give suggestions or recommendations  
3. DO NOT discuss what should be done - JUST DO IT
4. DIRECTLY WRITE the complete IND document
5. START your response with: "# IND APPLICATION:"

=== WRITING RULES ===
- Use ONLY the numerical data provided. Never invent values.
- Use hedging language for predictions: "suggests", "indicates", "predicts"
- Mark computational results as "in-silico prediction"
- Use formal FDA regulatory writing style
- For missing data, use: [PLACEHOLDER: description]

=== OUTPUT FORMAT ===
Write in Markdown with proper headers (##, ###, ####).
Generate realistic clinical protocol content based on provided parameters.
"""

SYSTEM_PROMPT_OPENAI = """
YOU ARE A REGULATORY DOCUMENT WRITER. NOT A REVIEWER OR CRITIC.

=== CRITICAL INSTRUCTIONS ===
1. DO NOT analyze, critique, or provide feedback on this request
2. DO NOT give suggestions or recommendations
3. DO NOT discuss what could be improved
4. DIRECTLY WRITE the complete IND application document
5. START your response with: "# IND APPLICATION:"

=== WRITING RULES ===
- Use ONLY the numerical data provided. Never invent or modify values.
- Use hedging language for predictions: "suggests", "indicates", "predicts"
- Always qualify computational results as "in-silico prediction"
- Distinguish clearly between computational predictions and experimental data
- Use formal FDA 21 CFR 312.23 compliant regulatory language

=== YOUR ROLE ===
You are a Senior Regulatory Medical Writer creating an actual IND submission document.
This is NOT a review exercise - you are PRODUCING the document.
"""

SYSTEM_PROMPT_GEMINI = """
YOU ARE A REGULATORY DOCUMENT WRITER. YOUR TASK IS TO WRITE, NOT REVIEW.

=== MANDATORY BEHAVIOR ===
✓ WRITE the complete IND application document directly
✓ START your response with "# IND APPLICATION: [Drug Name]"
✓ Follow the exact structure provided in the user message
✓ Generate realistic clinical protocol content

✗ DO NOT provide feedback, suggestions, or analysis
✗ DO NOT critique the provided data
✗ DO NOT offer improvements or recommendations
✗ DO NOT explain what you are doing - just do it

=== WRITING GUIDELINES ===
1. Use ONLY provided numerical data - never invent values
2. Use hedging language: "suggests", "indicates", "computational analysis predicts"
3. Mark all model predictions as "in-silico" or "computational"
4. For missing experimental data: [PLACEHOLDER: description of required data]
5. Use formal FDA regulatory writing style (21 CFR 312.23)

=== CRITICAL TABLE FORMATTING RULES ===
- Use SIMPLE Markdown tables with SHORT separators
- Separator row: |---|---| or |:---|:---| (maximum 5 dashes per column)
- NEVER use more than 5 dashes in any table cell
- Keep column content short and concise
- If content is long, summarize it or split into multiple rows

Example of CORRECT table:
| Field | Detail |
|:------|:-------|
| Name | Drug ABC |
| Class | Antibody |

Example of WRONG table (DO NOT DO THIS):
| Field | Detail |
|:-----------------------------------------------------|:----------------------------------------------|

=== OUTPUT ===
Produce a complete, professional IND application document in Markdown format.
"""

# =============================================================================
# USER PROMPTS - Clear structure with explicit output requirements
# =============================================================================

USER_PROMPT_LLAMA = """
TASK: Write a complete FDA IND Application document.
OUTPUT: Start with "# IND APPLICATION: {drug_name}" and write the full document.

=== PROVIDED DATA ===

APPLICANT:
- Name: {applicant_name}
- Address: {applicant_address}
- Phone: {applicant_phone}
- Date: {application_date}

INVESTIGATOR:
- PI: {pi_name} ({pi_credentials})
- Institution: {institution_name}
- Address: {institution_address}
- IRB: {irb_name}

DRUG:
- Name: {drug_name}
- Class: {drug_class}
- SMILES: {smiles}
- Indication: {indication}
- Mechanism: {mechanism}
- Dose: {dose_mg} mg {dosing_regimen}
- Population: {target_population}

CLINICAL TRIAL:
- Phase: {clinical_phase}
- Patients: {expected_patients}
- Duration: {study_duration}

PK (mPBPK Model):
- Cmax: {cmax} ng/mL
- AUC: {auc} ng·h/mL
- Half-life: {t_half} hours
- Target Occupancy: {to_trough}%

TOXICOLOGY (QSAR):
{qsar_results_formatted}

SAFETY:
- hERG Margin: {herg_margin}x
- Hepatotoxicity: {hepato_risk}
- Risk Score: {overall_score}/100

=== REQUIRED DOCUMENT SECTIONS ===

Write ALL of these sections:

## 1. COVER PAGE
- Applicant info table
- Drug identification
- Phase declaration
- Sponsor commitments

## 2. INTRODUCTION AND INVESTIGATIONAL PLAN
- Cover sheet with drug summary
- Scientific rationale for development
- General evaluation approach
- First-year study plans
- Risk assessment

## 3. CLINICAL PROTOCOL (Phase {clinical_phase})
- Protocol synopsis table
- Study objectives
- Investigator information
- Inclusion/exclusion criteria (generate realistic criteria)
- Study design (generate appropriate for phase)
- Dosing regimen with PK justification
- Safety monitoring plan

## 5. INVESTIGATOR'S BROCHURE
- Drug description
- Pharmacology summary
- PK summary (use mPBPK data)
- Toxicology summary (use QSAR data)
- Known risks and precautions

## 8. PHARMACOLOGY AND TOXICOLOGY
- Structural information
- PK predictions with model details
- QSAR toxicity predictions
- Safety pharmacology (hERG)
- GLP compliance statement (placeholder)
- Evaluator statement (placeholder)

NOW WRITE THE COMPLETE DOCUMENT:
"""

USER_PROMPT_OPENAI = """
TASK: Write a complete FDA IND Application.
START YOUR RESPONSE WITH: "# IND APPLICATION: {drug_name}"

=== INPUT DATA ===

APPLICANT INFORMATION:
| Field | Value |
|-------|-------|
| Name | {applicant_name} |
| Address | {applicant_address} |
| Phone | {applicant_phone} |
| Date | {application_date} |

INVESTIGATOR INFORMATION:
| Field | Value |
|-------|-------|
| Principal Investigator | {pi_name} |
| Credentials | {pi_credentials} |
| Institution | {institution_name} |
| Institution Address | {institution_address} |
| IRB | {irb_name} |

DRUG CANDIDATE:
| Field | Value |
|-------|-------|
| Name | {drug_name} |
| Class | {drug_class} |
| SMILES | {smiles} |
| Indication | {indication} |
| Mechanism | {mechanism} |
| Dose | {dose_mg} mg |
| Regimen | {dosing_regimen} |
| Population | {target_population} |

CLINICAL TRIAL PARAMETERS:
| Field | Value |
|-------|-------|
| Phase | Phase {clinical_phase} |
| Expected Patients | {expected_patients} |
| Duration | {study_duration} |

PHARMACOKINETICS (mPBPK Model Predictions):
| Parameter | Value | Unit |
|-----------|-------|------|
| Cmax | {cmax} | ng/mL |
| AUC (0-∞) | {auc} | ng·h/mL |
| Half-life | {t_half} | hours |
| Target Occupancy | {to_trough} | % |
| Vss | {vss} | L/kg |

TOXICOLOGY (QSAR Predictions):
{qsar_results_formatted}

SAFETY ASSESSMENT:
| Parameter | Value |
|-----------|-------|
| hERG Margin | {herg_margin}x |
| Hepatotoxicity Risk | {hepato_risk} |
| Overall Risk Score | {overall_score}/100 |

=== GENERATE THESE SECTIONS ===

1. COVER PAGE (Form FDA 1571 format)
2. INTRODUCTION AND INVESTIGATIONAL PLAN
3. CLINICAL PROTOCOL (Phase {clinical_phase} - generate realistic content)
5. INVESTIGATOR'S BROCHURE
8. PHARMACOLOGY AND TOXICOLOGY

For clinical protocol: Generate appropriate inclusion/exclusion criteria, study design, dosing justification, and safety monitoring based on the indication and PK data.

For sections without data: Use [PLACEHOLDER: description]

BEGIN WRITING THE IND APPLICATION NOW:
"""

USER_PROMPT_GEMINI = """
WRITE a complete FDA IND Application document for the drug candidate below.
BEGIN your response with: "# IND APPLICATION: {drug_name}"

---

## PROVIDED DATA

### Applicant Information
- **Name:** {applicant_name}
- **Address:** {applicant_address}
- **Phone:** {applicant_phone}
- **Application Date:** {application_date}

### Investigator Information
- **Principal Investigator:** {pi_name}
- **Credentials:** {pi_credentials}
- **Institution:** {institution_name}
- **Institution Address:** {institution_address}
- **IRB/Ethics Committee:** {irb_name}

### Drug Candidate
- **Drug Name:** {drug_name}
- **Drug Class:** {drug_class}
- **SMILES/Sequence:** {smiles}
- **Therapeutic Indication:** {indication}
- **Mechanism of Action:** {mechanism}
- **Proposed Dose:** {dose_mg} mg
- **Dosing Regimen:** {dosing_regimen}
- **Target Population:** {target_population}

### Clinical Trial Parameters
- **Phase:** Phase {clinical_phase}
- **Expected Enrollment:** {expected_patients} patients
- **Study Duration:** {study_duration}

### Pharmacokinetics (mPBPK Model - In-Silico Predictions)
| Parameter | Predicted Value | Unit |
|-----------|----------------|------|
| Cmax | {cmax} | ng/mL |
| AUC (0-∞) | {auc} | ng·h/mL |
| Terminal Half-life | {t_half} | hours |
| Target Occupancy at Trough | {to_trough} | % |
| Volume of Distribution | {vss} | L/kg |
| Clearance | {cl} | mL/min/kg |
| Half-life | {t_half} | h |
| Fraction Unbound | {fup}% | |

### Non-Clinical Pharmacology (Animal Model Data - In-Silico)
| Species | Clearance (mL/min/kg) | Vss (L/kg) | Fup (%) |
|---------|-----------------------|------------|---------|
| Rat | {rat_cl} | {rat_vss} | {rat_fup}% |
| Dog | {dog_cl} | {dog_vss} | {dog_fup}% |
| Monkey | {monkey_cl} | {monkey_vss} | {monkey_fup}% |


### Toxicology (QSAR Model - In-Silico Predictions)
{qsar_results_formatted}

### Safety Assessment
| Assessment | Value | Interpretation |
|------------|-------|----------------|
| hERG IC50 Safety Margin | {herg_margin}x | {herg_interpretation} |
| Hepatotoxicity Risk | {hepato_risk} | Structural alert analysis |
| Genotoxicity (Ames) | {ames_result} | In-silico prediction |
| Overall Risk Score | {overall_score}/100 | {risk_category} |

---

## REQUIRED OUTPUT STRUCTURE

Generate a complete IND application with ALL of these sections:

### Section 1: Cover Page (FDA Form 1571)
- Complete applicant information table
- Drug identification
- Clinical phase declaration
- Sponsor commitments per 21 CFR 312.23

### Section 2: Introduction and General Investigational Plan
- Cover sheet with drug summary
- Introductory statement with scientific rationale
- Research justification based on indication
- General approach to evaluation
- First-year study summary
- Risk assessment based on computational predictions

### Section 3: Clinical Protocol
Generate a detailed Phase {clinical_phase} clinical trial protocol including:
- Protocol synopsis table
- Primary and secondary objectives
- Investigator information
- Patient selection criteria (generate realistic inclusion/exclusion criteria based on {indication})
- Study design appropriate for Phase {clinical_phase}
- Dosing regimen with justification from PK predictions
- Safety monitoring plan based on identified risks
- Efficacy assessments
- Statistical considerations

### Section 5: Investigator's Brochure
- Drug substance description with structure
- Pharmacology summary with mechanism
- Pharmacokinetics summary (use mPBPK data, clearly marked as in-silico)
- Toxicology summary (use QSAR data, clearly marked as predictions)
- Known risks and precautionary measures

### Section 8: Pharmacology and Toxicology
- 8.1 Introduction (structural info, pharmacologic class)
- 8.2 Non-Clinical Pharmacology
    - Interspecies PK comparison (Rat, Dog, Monkey vs Human)
    - Allometric scaling justification
- 8.3 Pharmacokinetics (mPBPK predictions with appropriate caveats)
- 8.4 Toxicology (QSAR predictions with risk interpretation)
- 8.5 Safety Pharmacology (hERG assessment, CV safety)
- 8.6 GLP Compliance Statement [PLACEHOLDER]

- 8.6 Evaluator Statement [PLACEHOLDER]
- 8.7 References [PLACEHOLDER]

---

NOW WRITE THE COMPLETE IND APPLICATION DOCUMENT:
"""


def get_system_prompt() -> str:
    """Return the appropriate system prompt based on LLM provider."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "openai":
        return SYSTEM_PROMPT_OPENAI
    elif provider == "gemini":
        return SYSTEM_PROMPT_GEMINI
    return SYSTEM_PROMPT_LLAMA


def get_user_prompt(data: dict) -> str:
    """
    Return the appropriate user prompt with data filled in.
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    # Set defaults for optional fields
    defaults = {
        # Applicant defaults
        "applicant_name": "[Applicant Name - To Be Filled]",
        "applicant_address": "[Applicant Address - To Be Filled]",
        "applicant_phone": "[Phone Number - To Be Filled]",
        "application_date": "[Date]",
        
        # Investigator defaults
        "pi_name": "[Principal Investigator - To Be Filled]",
        "pi_credentials": "[Credentials]",
        "institution_name": "[Research Institution - To Be Filled]",
        "institution_address": "[Institution Address]",
        "irb_name": "[IRB Name - To Be Filled]",
        
        # Drug defaults
        "drug_class": "Small Molecule",
        "mechanism": "[Mechanism of action to be determined]",
        "dosing_regimen": "QD",
        "target_population": "Adult patients",
        
        # Clinical trial defaults
        "clinical_phase": "1",
        "expected_patients": "30",
        "study_duration": "12 weeks",
        
        # PK defaults
        "vss": "N/A",
        "to_trough": data.get("TO_trough", data.get("to_trough", "N/A")),
        
        # Safety defaults
        "herg_interpretation": "Acceptable" if data.get("herg_margin", 0) >= 30 else "Requires monitoring",
        "herg_monitoring": "Yes" if data.get("herg_margin", 0) < 30 else "Standard",
        "hepato_monitoring": "Yes" if data.get("hepato_risk", "Low") != "Low" else "Standard",
        "ames_result": "Negative (predicted)",
        "risk_category": "Low" if data.get("overall_score", 0) >= 70 else "Moderate",
        
        # Human PK Extra Defaults
        "cl": "N/A",
        "fup": "N/A",

        # Animal PK Defaults
        "rat_cl": "N/A", "rat_vss": "N/A", "rat_fup": "N/A",
        "dog_cl": "N/A", "dog_vss": "N/A", "dog_fup": "N/A",
        "monkey_cl": "N/A", "monkey_vss": "N/A", "monkey_fup": "N/A"
    }
    
    # Merge defaults with provided data
    merged_data = {**defaults, **data}
    
    # Ensure to_trough uses correct key
    if "TO_trough" in data and "to_trough" not in data:
        merged_data["to_trough"] = data["TO_trough"]
    
    if provider == "openai":
        return USER_PROMPT_OPENAI.format(**merged_data)
    elif provider == "gemini":
        return USER_PROMPT_GEMINI.format(**merged_data)
    else:
        return USER_PROMPT_LLAMA.format(**merged_data)


def format_qsar_results(qsar_predictions: dict, threshold: float = 0.20) -> str:
    """
    Format QSAR prediction results for inclusion in the prompt.
    """
    if not qsar_predictions:
        return "No QSAR predictions available."
    
    lines = ["| Endpoint | Probability | Prediction |", "|----------|-------------|------------|"]
    
    positive_count = 0
    for endpoint, pred in qsar_predictions.items():
        if hasattr(pred, 'probability'):
            prob = pred.probability
        elif isinstance(pred, dict):
            prob = pred.get('probability', pred.get('prob', 0.0))
        else:
            prob = float(pred)
        
        is_positive = prob >= threshold
        if is_positive:
            positive_count += 1
        
        prediction = "POSITIVE" if is_positive else "Negative"
        lines.append(f"| {endpoint} | {prob:.3f} | {prediction} |")
    
    lines.append("")
    lines.append(f"**Summary**: {positive_count}/{len(qsar_predictions)} endpoints positive (threshold: {threshold})")
    
    if positive_count <= 1:
        risk = "LOW"
    elif positive_count <= 3:
        risk = "MEDIUM"
    elif positive_count <= 5:
        risk = "HIGH"
    else:
        risk = "CRITICAL"
    
    lines.append(f"**Toxicity Risk Level**: {risk}")
    
    return "\n".join(lines)
