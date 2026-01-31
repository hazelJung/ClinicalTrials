# DATA_CANDIDATES.md

## 1. Data Source Candidates
This document lists potential data sources for the AI-Driven Clinical Trial Platform.

### A. Drug Properties & Bioactivity (For mPBPK & QSAR)
| Source Name | Key Data Types | Potential Usage |
| :--- | :--- | :--- |
| **DrugBank** | Drug Targets, PK parameters, Dosing Info | Mining $CL$, $V_{ss}$, Molecular Weight, Half-life. |
| **ChEMBL** | Bioactivity ($K_D$, $IC_{50}$, $EC_{50}$) | Extracting Binding Affinity ($K_D$) for Antibody-Target. |
| **PubChem** | Chemical Structures (SMILES), Physical Props | Input features for QSAR (SMILES generation). |
| **DrugCentral** | Structure, OMOP CDM format | Validation set for drug structures. |

### B. Pharmacogenomics & Safety (For QSAR & Toxicity)
| Source Name | Key Data Types | Potential Usage |
| :--- | :--- | :--- |
| **PharmGKB** | Gene-Drug Interactions, Variant Annotations | Identifying high-risk metabolic variants (e.g., CYP2D6). |
| **CPIC** | Clinical Guidelines based on Genotype | Rules for dose adjustment in virtual patients. |
| **FAERS** | Adverse Event Reports (Real-world) | Mining toxicity signals (Liver tox, Cardiotox). |
| **의약품안전나라** | Korean Approved Drugs & Side Effects | Safety labels specific to Korean market regulations. |

### C. Population & Genetics (For Virtual Cohort)
| Source Name | Key Data Types | Potential Usage |
| :--- | :--- | :--- |
| **1K Genomes** | Global Genomic Variants | Generating diverse virtual patient populations. |
| **KoGES** | Korean Genome and Epidemiology Study | **[Core]** Creating Korean-specific virtual cohort physiology. |
| **gnomAD** | Genome Aggregation Database | Reference for allele frequencies (rare variants). |

### D. Clinical Trials (For Validation)
| Source Name | Key Data Types | Potential Usage |
| :--- | :--- | :--- |
| **ClinicalTrials.gov** | Study Design, Outcomes, Adverse Events | Extracting observed clinical data for mPBPK validation. |

---

## 2. Selection Criteria (Agent Instructions)
1. **Accessibility:** Prioritize open-access or easily scrapable data.
2. **Relevance:**
   - **mPBPK:** Needs specific numbers ($K_D$, $CL$ in units). -> **ChEMBL, DrugBank** preferred.
   - **QSAR:** Needs Structure-Label pairs. -> **PubChem** (Structure) + **ClinTox** (Label) preferred.
   - **Cohort:** Needs Height/Weight/Organ-Size correlations. -> **KoGES** preferred.