"""
ChEMBL API Client - High-Quality Binding Affinity Extraction
=============================================================
Fetches binding data via ChEMBL REST API with quality filters.

Quality Filters Applied:
- assay_type = "B" (Binding assays only)
- standard_relation = "=" (exact values only)
- pchembl_value exists (quantified data only)

Outputs:
- chembl_kd_data.csv    : KD 전용 데이터
- chembl_ic50_data.csv  : IC50 전용 데이터
- chembl_herg_data.csv  : hERG 심장 독성 데이터

Author: AI-Driven Clinical Trial Platform
"""

import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# Output files (separated by data type)
OUTPUT_DIR = Path("data/processed")
KD_OUTPUT_FILE = OUTPUT_DIR / "chembl_kd_data.csv"                  # kd, ic50, herg 데이터셋 분리해서 저장
IC50_OUTPUT_FILE = OUTPUT_DIR / "chembl_ic50_data.csv"              # 이렇게 안하면 홉합돼서 학습 시 악영향
HERG_OUTPUT_FILE = OUTPUT_DIR / "chembl_herg_data.csv"
SUMMARY_OUTPUT_FILE = OUTPUT_DIR / "chembl_binding_summary.csv"

# API Parameters
MAX_RESULTS = 1000
RATE_LIMIT_DELAY = 0.5  # Seconds between requests

# hERG Target
HERG_TARGET_ID = "CHEMBL240"  # hERG (KCNH2) 채널


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BindingData:
    """결합력 데이터"""
    chembl_id: str = ""
    molecule_name: str = ""
    target_name: str = ""
    target_chembl_id: str = ""
    assay_type: str = ""
    assay_description: str = ""
    standard_type: str = ""  # Kd, Ki, IC50
    standard_relation: str = ""  # =, >, <
    standard_value: float = None
    standard_units: str = ""
    pchembl_value: float = None
    data_validity_comment: str = ""


# =============================================================================
# API Functions
# =============================================================================

def fetch_json(url: str, params: Dict = None) -> Optional[Dict]:
    """ChEMBL API 호출"""
    try:
        headers = {"Accept": "application/json"}
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️ API Error: {e}")
        return None


def fetch_high_quality_binding_data(
    standard_type: str = "Kd",
    limit: int = 200,
    offset: int = 0
) -> List[Dict]:
    """
    고품질 결합력 데이터 조회 (신뢰도 필터 적용)    / 데이터셋 한계 때문에 필요한 데이터만 뽑아오기 위한 필터링 작업
    
    Quality Filters:
    - assay_type = "B" (Binding assays only)
    - standard_relation = "=" (exact values, not > or <)
    - pchembl_value exists (quantified data)
    - standard_units = "nM" (consistent units)
    """
    url = f"{CHEMBL_API_BASE}/activity.json"
    params = {                          
        "standard_type": standard_type,
        "standard_units": "nM",
        "standard_relation": "=",             # 정확한 값만 (>, < 제외)
        "assay_type": "B",                    # Binding assay만
        "pchembl_value__isnull": "false",     # pChEMBL 값 있는 것만
        "data_validity_comment__isnull": "true",  # 유효성 문제 없는 것
        "limit": limit,
        "offset": offset
    }
    
    data = fetch_json(url, params)
    if data and "activities" in data:
        return data["activities"]
    return []


def fetch_kd_data(max_records: int = 300) -> List[BindingData]:
    """
    KD (Dissociation Constant) 데이터 수집
    - Binding affinity의 직접 측정값
    - 낮을수록 강한 결합력
    """
    print(f"\n📡 Fetching KD data (max {max_records} records)...")
    
    all_data = []
    offset = 0
    
    while len(all_data) < max_records:
        activities = fetch_high_quality_binding_data(
            standard_type="Kd",
            limit=100,
            offset=offset
        )
        
        if not activities:
            break
        
        for act in activities:
            binding = BindingData(
                chembl_id=act.get("molecule_chembl_id", ""),
                molecule_name=act.get("molecule_pref_name", ""),
                target_name=act.get("target_pref_name", ""),
                target_chembl_id=act.get("target_chembl_id", ""),
                assay_type=act.get("assay_type", ""),
                assay_description=act.get("assay_description", "")[:200] if act.get("assay_description") else "",
                standard_type="Kd",
                standard_relation=act.get("standard_relation", ""),
                standard_value=act.get("standard_value"),
                standard_units=act.get("standard_units", ""),
                pchembl_value=act.get("pchembl_value"),
                data_validity_comment=act.get("data_validity_comment", "") or ""
            )
            all_data.append(binding)
        
        print(f"   Fetched {len(all_data)} KD records...")
        offset += 100
        time.sleep(RATE_LIMIT_DELAY)
        
        if len(all_data) >= max_records:
            break
    
    print(f"✅ Total KD records: {len(all_data)}")
    return all_data


def fetch_ic50_data(max_records: int = 300) -> List[BindingData]:
    """
    IC50 (Half-maximal Inhibitory Concentration) 데이터 수집
    - 억제 활성 측정
    - Functional assay 결과
    """
    print(f"\n📡 Fetching IC50 data (max {max_records} records)...")
    
    all_data = []
    offset = 0
    
    while len(all_data) < max_records:
        activities = fetch_high_quality_binding_data(
            standard_type="IC50",
            limit=100,
            offset=offset
        )
        
        if not activities:
            break
        
        for act in activities:
            binding = BindingData(
                chembl_id=act.get("molecule_chembl_id", ""),
                molecule_name=act.get("molecule_pref_name", ""),
                target_name=act.get("target_pref_name", ""),
                target_chembl_id=act.get("target_chembl_id", ""),
                assay_type=act.get("assay_type", ""),
                assay_description=act.get("assay_description", "")[:200] if act.get("assay_description") else "",
                standard_type="IC50",
                standard_relation=act.get("standard_relation", ""),
                standard_value=act.get("standard_value"),
                standard_units=act.get("standard_units", ""),
                pchembl_value=act.get("pchembl_value"),
                data_validity_comment=act.get("data_validity_comment", "") or ""
            )
            all_data.append(binding)
        
        print(f"   Fetched {len(all_data)} IC50 records...")
        offset += 100
        time.sleep(RATE_LIMIT_DELAY)
        
        if len(all_data) >= max_records:
            break
    
    print(f"✅ Total IC50 records: {len(all_data)}")
    return all_data


def fetch_herg_data(max_records: int = 500) -> List[BindingData]:
    """
    hERG 심장 독성 데이터 수집 (고품질 필터 적용)
    
    hERG (KCNH2) 채널 억제 → QT 연장 → 심장 독성
    
    판정 기준:
    - IC50 < 1 µM (1000 nM): 고위험
    - IC50 1-10 µM: 중위험
    - IC50 > 10 µM: 저위험
    """
    print(f"\n📡 Fetching hERG (Cardiac Toxicity) data...")
    print(f"   Target: {HERG_TARGET_ID}")
    
    all_data = []
    
    for standard_type in ["IC50", "Ki"]:
        url = f"{CHEMBL_API_BASE}/activity.json"
        params = {
            "target_chembl_id": HERG_TARGET_ID,
            "standard_type": standard_type,
            "standard_units": "nM",
            "standard_relation": "=",
            "pchembl_value__isnull": "false",
            "limit": 200
        }
        
        data = fetch_json(url, params)
        if data and "activities" in data:
            for act in data["activities"]:
                binding = BindingData(
                    chembl_id=act.get("molecule_chembl_id", ""),
                    molecule_name=act.get("molecule_pref_name", ""),
                    target_name="hERG (KCNH2)",
                    target_chembl_id=HERG_TARGET_ID,
                    assay_type=act.get("assay_type", ""),
                    assay_description=act.get("assay_description", "")[:200] if act.get("assay_description") else "",
                    standard_type=standard_type,
                    standard_relation=act.get("standard_relation", ""),
                    standard_value=act.get("standard_value"),
                    standard_units=act.get("standard_units", ""),
                    pchembl_value=act.get("pchembl_value"),
                    data_validity_comment=act.get("data_validity_comment", "") or ""
                )
                all_data.append(binding)
            
            print(f"   Found {len(data['activities'])} {standard_type} records")
        
        time.sleep(RATE_LIMIT_DELAY)
        
        if len(all_data) >= max_records:
            break
    
    print(f"✅ Total hERG records: {len(all_data)}")
    return all_data


# =============================================================================
# Data Processing
# =============================================================================

def binding_to_dataframe(data: List[BindingData]) -> pd.DataFrame:
    """BindingData 리스트를 DataFrame으로 변환"""
    records = []
    for d in data:
        records.append({
            'chembl_id': d.chembl_id,
            'molecule_name': d.molecule_name,
            'target_name': d.target_name,
            'target_chembl_id': d.target_chembl_id,
            'assay_type': d.assay_type,
            'standard_type': d.standard_type,
            'standard_relation': d.standard_relation,
            'value_nM': d.standard_value,
            'units': d.standard_units,
            'pchembl_value': d.pchembl_value,
            'data_validity': 'VALID' if not d.data_validity_comment else 'FLAGGED'
        })
    return pd.DataFrame(records)


def classify_herg_toxicity(ic50_nM: float) -> str:      # 실제 제약사 분류 기준(internal tox triage 기준)
    """hERG IC50 → 심장 독성 위험도 분류"""
    if ic50_nM is None:
        return 'UNKNOWN'
    if ic50_nM < 1000:      # < 1 µM
        return 'HIGH'
    elif ic50_nM < 10000:   # 1-10 µM
        return 'MEDIUM'
    else:                    # > 10 µM
        return 'LOW'


def herg_to_dataframe(data: List[BindingData]) -> pd.DataFrame:
    """hERG 데이터 → DataFrame (독성 분류 포함)"""
    df = binding_to_dataframe(data)
    df['cardiac_toxicity_risk'] = df['value_nM'].apply(classify_herg_toxicity)
    return df


def calculate_statistics(df: pd.DataFrame, value_col: str = 'value_nM') -> Dict:
    """분포 통계 계산"""
    import numpy as np
    
    values = df[value_col].dropna()
    
    if len(values) == 0:
        return {'count': 0}
    
    return {
        'count': len(values),
        'min_nM': values.min(),
        'max_nM': values.max(),
        'median_nM': values.median(),
        'mean_nM': values.mean(),
        'std_nM': values.std(),
        'p25_nM': values.quantile(0.25),
        'p75_nM': values.quantile(0.75),
        'log10_range': f"{np.log10(max(values.min(), 0.001)):.2f} ~ {np.log10(values.max()):.2f}"
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("  ChEMBL API Client - High-Quality Binding Data")
    print("="*60)
    print("  Quality Filters: assay_type=B, relation='=', pchembl exists")
    print("="*60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    summary_data = []
    
    # ===========================================
    # 1. KD Data (Binding Affinity)
    # ===========================================
    print("\n" + "="*60)
    print("  [1/3] KD Data Collection")
    print("="*60)
    
    kd_data = fetch_kd_data(max_records=300)
    if kd_data:
        df_kd = binding_to_dataframe(kd_data)
        df_kd = df_kd.drop_duplicates(subset=['chembl_id', 'target_chembl_id', 'value_nM'])
        df_kd.to_csv(KD_OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"📁 Saved: {KD_OUTPUT_FILE}")
        
        stats = calculate_statistics(df_kd)
        stats['data_type'] = 'KD'
        summary_data.append(stats)
        
        print(f"   Count: {stats['count']}, Range: {stats.get('log10_range', 'N/A')}")
    
    # ===========================================
    # 2. IC50 Data (Inhibition)
    # ===========================================
    print("\n" + "="*60)
    print("  [2/3] IC50 Data Collection")
    print("="*60)
    
    ic50_data = fetch_ic50_data(max_records=300)
    if ic50_data:
        df_ic50 = binding_to_dataframe(ic50_data)
        df_ic50 = df_ic50.drop_duplicates(subset=['chembl_id', 'target_chembl_id', 'value_nM'])
        df_ic50.to_csv(IC50_OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"📁 Saved: {IC50_OUTPUT_FILE}")
        
        stats = calculate_statistics(df_ic50)
        stats['data_type'] = 'IC50'
        summary_data.append(stats)
        
        print(f"   Count: {stats['count']}, Range: {stats.get('log10_range', 'N/A')}")
    
    # ===========================================
    # 3. hERG Data (Cardiac Toxicity)
    # ===========================================
    print("\n" + "="*60)
    print("  [3/3] hERG (Cardiac Toxicity) Data Collection")
    print("="*60)
    
    herg_data = fetch_herg_data(max_records=500)
    if herg_data:
        df_herg = herg_to_dataframe(herg_data)
        df_herg = df_herg.drop_duplicates(subset=['chembl_id', 'value_nM'])
        df_herg.to_csv(HERG_OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"📁 Saved: {HERG_OUTPUT_FILE}")
        
        # Risk distribution
        print("\n📊 Cardiac Toxicity Risk Distribution:")
        risk_counts = df_herg['cardiac_toxicity_risk'].value_counts()
        for risk, count in risk_counts.items():
            print(f"   {risk}: {count} compounds")
        
        stats = calculate_statistics(df_herg)
        stats['data_type'] = 'hERG'
        summary_data.append(stats)
    
    # ===========================================
    # Summary
    # ===========================================
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(SUMMARY_OUTPUT_FILE, index=False)
        print(f"📁 Saved summary: {SUMMARY_OUTPUT_FILE}")
        
        print("\n📊 Data Collection Summary:")
        for s in summary_data:
            print(f"   {s['data_type']}: {s['count']} records")
    
    print("\n✅ All data collection complete!")
    print(f"\n📂 Output files:")
    print(f"   - {KD_OUTPUT_FILE}")
    print(f"   - {IC50_OUTPUT_FILE}")
    print(f"   - {HERG_OUTPUT_FILE}")
    print(f"   - {SUMMARY_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
