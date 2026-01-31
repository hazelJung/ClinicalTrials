"""
DrugBank XML Parser - PK Parameter Extraction
==============================================
Extracts pharmacokinetic parameters from DrugBank Complete Database XML.

Filters: Biotech drugs (antibodies)
Outputs: Parameter ranges for virtual data generation

Author: AI-Driven Clinical Trial Platform
"""

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import iterparse
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

DRUGBANK_NS = "{http://www.drugbank.ca}"
INPUT_FILE = Path("data/raw/full database.xml")
OUTPUT_FILE = Path("data/processed/drugbank_pk_parameters.csv")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DrugInfo:
    """약물 정보"""
    drugbank_id: str = ""
    name: str = ""
    drug_type: str = ""
    groups: List[str] = None
    average_mass: float = None
    half_life: str = ""
    half_life_value: float = None
    half_life_unit: str = ""
    protein_binding: str = ""
    protein_binding_value: float = None
    volume_of_distribution: str = ""
    vd_value: float = None
    vd_unit: str = ""
    clearance: str = ""
    clearance_value: float = None
    clearance_unit: str = ""
    absorption: str = ""
    toxicity: str = ""
    # Dosage info
    dosage_form: str = ""
    route: str = ""
    strength: str = ""
    average_dose_mg: float = None


# =============================================================================
# Parsing Utilities
# =============================================================================

def parse_numeric_value(text: str) -> Tuple[Optional[float], str]:
    """
    텍스트에서 숫자와 단위 추출
    
    예: "14 days" -> (14, "days")
        "0.12 L/hr" -> (0.12, "L/hr")
        "50-70%" -> (60, "%")
    """
    if not text:
        return None, ""
    
    # Range: "50-70" -> average
    range_match = re.search(r'(\d+\.?\d*)\s*[-–~]\s*(\d+\.?\d*)', text)
    if range_match:
        low, high = float(range_match.group(1)), float(range_match.group(2))
        value = (low + high) / 2
        # Find unit after the range
        unit_match = re.search(r'[\d\.]+\s*([\w/%]+)', text[range_match.end():])
        unit = unit_match.group(1) if unit_match else ""
        return value, unit
    
    # Single value: "14 days"
    single_match = re.search(r'(\d+\.?\d*)\s*([\w/\*%]+)?', text)
    if single_match:
        value = float(single_match.group(1))
        unit = single_match.group(2) if single_match.group(2) else ""
        return value, unit
    
    return None, ""


def get_text(element, tag: str, ns: str = DRUGBANK_NS) -> str:
    """XML 요소에서 텍스트 추출"""
    child = element.find(f"{ns}{tag}")
    if child is not None and child.text:
        return child.text.strip()
    return ""


def get_groups(element, ns: str = DRUGBANK_NS) -> List[str]:
    """약물 승인 그룹 추출"""
    groups_elem = element.find(f"{ns}groups")
    if groups_elem is None:
        return []
    
    groups = []
    for group in groups_elem.findall(f"{ns}group"):
        if group.text:
            groups.append(group.text.strip())
    return groups


def parse_average_dose(strength: str) -> Optional[float]:
    """
    용량 문자열에서 평균 용량(mg) 추출
    예: "100 mg" -> 100.0
        "50 mg/ml" -> None (volume unknown) unless we assume typical volume
        "10-20 mg" -> 15.0
    """
    if not strength:
        return None
    
    # mg 단위만 우선 처리
    if 'mg' not in strength.lower():
        return None
        
    # Range: "10-20 mg"
    range_match = re.search(r'(\d+\.?\d*)\s*[-–~]\s*(\d+\.?\d*)', strength)
    if range_match:
        return (float(range_match.group(1)) + float(range_match.group(2))) / 2
        
    # Single value: "100 mg"
    single_match = re.search(r'(\d+\.?\d*)', strength)
    if single_match:
        return float(single_match.group(1))
        
    return None


def get_dosage_info(element, ns: str = DRUGBANK_NS) -> Tuple[str, str, str, Optional[float]]:
    """약물 용량 정보 추출 (Dosages -> Dosage)"""
    dosages_elem = element.find(f"{ns}dosages")
    if dosages_elem is None:
        return "", "", "", None
        
    # Prioritize injection/intravenous for biotech drugs
    best_dosage = None
    
    for dosage in dosages_elem.findall(f"{ns}dosage"):
        form = get_text(dosage, "form", ns)
        route = get_text(dosage, "route", ns)
        strength = get_text(dosage, "strength", ns)
        
        # Check for numeric dose
        avg_dose = parse_average_dose(strength)
        
        current_dosage = (form, route, strength, avg_dose)
        
        if not best_dosage:
            best_dosage = current_dosage
        
        # Prioritize Injection/IV with valid dose
        if avg_dose and ("inject" in route.lower() or "intraven" in route.lower()):
            return current_dosage
            
        # Prioritize having a valid dose
        if avg_dose and not best_dosage[3]:
            best_dosage = current_dosage
            
    return best_dosage if best_dosage else ("", "", "", None)


# =============================================================================
# Main Parser
# =============================================================================

def parse_drugbank_xml(filepath: Path, filter_biotech: bool = True) -> List[DrugInfo]:
    """
    DrugBank XML 파싱 (메모리 효율적인 iterparse 사용)
    
    Args:
        filepath: XML 파일 경로
        filter_biotech: True면 biotech(항체) 약물만 추출
        
    Returns:
        List[DrugInfo]: 추출된 약물 정보 리스트
    """
    drugs = []
    
    print(f"📂 Parsing: {filepath}")
    print("   This may take a few minutes for large files...")
    
    # Iterative parsing for memory efficiency
    context = iterparse(str(filepath), events=("end",))
    
    count = 0
    biotech_count = 0
    
    for event, elem in context:
        if elem.tag == f"{DRUGBANK_NS}drug":
            count += 1
            
            # Filter by drug type
            drug_type = elem.get("type", "")
            if filter_biotech and drug_type != "biotech":
                elem.clear()
                continue
            
            biotech_count += 1
            
            # Extract drug info
            drug = DrugInfo(
                drug_type=drug_type,
                groups=get_groups(elem)
            )
            
            # Basic info
            id_elem = elem.find(f"{DRUGBANK_NS}drugbank-id[@primary='true']")
            if id_elem is None:
                id_elem = elem.find(f"{DRUGBANK_NS}drugbank-id")
            drug.drugbank_id = id_elem.text if id_elem is not None else ""
            
            drug.name = get_text(elem, "name")
            
            # Mass
            mass_text = get_text(elem, "average-mass")
            if mass_text:
                try:
                    drug.average_mass = float(mass_text)
                except:
                    pass
            
            # PK parameters (text fields)
            drug.half_life = get_text(elem, "half-life")
            drug.protein_binding = get_text(elem, "protein-binding")
            drug.volume_of_distribution = get_text(elem, "volume-of-distribution")
            drug.clearance = get_text(elem, "clearance")
            drug.absorption = get_text(elem, "absorption")
            drug.toxicity = get_text(elem, "toxicity")
            
            # Dosage
            drug.dosage_form, drug.route, drug.strength, drug.average_dose_mg = get_dosage_info(elem)
            
            # Parse numeric values
            drug.half_life_value, drug.half_life_unit = parse_numeric_value(drug.half_life)
            drug.protein_binding_value, _ = parse_numeric_value(drug.protein_binding)
            drug.vd_value, drug.vd_unit = parse_numeric_value(drug.volume_of_distribution)
            drug.clearance_value, drug.clearance_unit = parse_numeric_value(drug.clearance)
            
            drugs.append(drug)
            
            if biotech_count % 100 == 0:
                print(f"   Processed {biotech_count} biotech drugs...")
            
            # Clear element to free memory
            elem.clear()
    
    print(f"✅ Parsing complete!")
    print(f"   Total drugs: {count}")
    print(f"   Biotech drugs: {biotech_count}")
    
    return drugs


def drugs_to_dataframe(drugs: List[DrugInfo]) -> pd.DataFrame:
    """DrugInfo 리스트를 DataFrame으로 변환"""
    data = []
    for d in drugs:
        data.append({
            'drugbank_id': d.drugbank_id,
            'name': d.name,
            'type': d.drug_type,
            'groups': ','.join(d.groups) if d.groups else '',
            'average_mass_Da': d.average_mass,
            'half_life_raw': d.half_life,
            'half_life_value': d.half_life_value,
            'half_life_unit': d.half_life_unit,
            'protein_binding_raw': d.protein_binding,
            'protein_binding_pct': d.protein_binding_value,
            'vd_raw': d.volume_of_distribution,
            'vd_value': d.vd_value,
            'vd_unit': d.vd_unit,
            'clearance_raw': d.clearance,
            'clearance_value': d.clearance_value,
            'clearance_unit': d.clearance_unit,
            'absorption': d.absorption[:500] if d.absorption else '',  # Truncate long text
            'toxicity': d.toxicity[:500] if d.toxicity else '',
            'dosage_form': d.dosage_form,
            'route': d.route,
            'strength': d.strength,
            'average_dose_mg': d.average_dose_mg
        })
    return pd.DataFrame(data)


def calculate_statistics(df: pd.DataFrame) -> Dict:
    """파라미터 통계 계산"""
    stats = {}
    
    # Molecular Weight (Da -> kDa)
    mw = df['average_mass_Da'].dropna()
    if len(mw) > 0:
        stats['MW_kDa'] = {
            'min': mw.min() / 1000,
            'max': mw.max() / 1000,
            'median': mw.median() / 1000,
            'count': len(mw)
        }
    
    # Half-life (days)
    hl = df[df['half_life_unit'].str.contains('day', case=False, na=False)]['half_life_value'].dropna()
    if len(hl) > 0:
        stats['half_life_day'] = {
            'min': hl.min(),
            'max': hl.max(),
            'median': hl.median(),
            'count': len(hl)
        }
    
    # Protein binding (fraction unbound = 1 - binding%)
    pb = df['protein_binding_pct'].dropna()
    if len(pb) > 0:
        stats['protein_binding_pct'] = {
            'min': pb.min(),
            'max': pb.max(),
            'median': pb.median(),
            'count': len(pb)
        }

    # Dose (mg)
    dose = df['average_dose_mg'].dropna()
    if len(dose) > 0:
        stats['dose_mg'] = {
            'min': dose.min(),
            'max': dose.max(),
            'median': dose.median(),
            'count': len(dose)
        }
    
    return stats


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("  DrugBank XML Parser - PK Parameter Extraction")
    print("="*60)
    
    if not INPUT_FILE.exists():
        print(f"❌ Error: File not found: {INPUT_FILE}")
        return
    
    # Parse XML
    drugs = parse_drugbank_xml(INPUT_FILE, filter_biotech=True)
    
    if not drugs:
        print("⚠️ No biotech drugs found!")
        return
    
    # Convert to DataFrame
    df = drugs_to_dataframe(drugs)
    
    # Save to CSV
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"\n📁 Saved to: {OUTPUT_FILE}")
    
    # Calculate and print statistics
    print("\n" + "="*60)
    print("  Parameter Statistics (Biotech Drugs)")
    print("="*60)
    
    stats = calculate_statistics(df)
    for param, values in stats.items():
        print(f"\n📊 {param}:")
        for k, v in values.items():
            if isinstance(v, float):
                print(f"   {k}: {v:.2f}")
            else:
                print(f"   {k}: {v}")
    
    # Preview
    print("\n" + "="*60)
    print("  Sample Data (First 5 drugs)")
    print("="*60)
    print(df[['drugbank_id', 'name', 'average_mass_Da', 'half_life_value']].head())
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
