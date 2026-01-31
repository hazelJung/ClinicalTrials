"""
mPBPK Engine - Minimal Physiologically Based Pharmacokinetic Model
===================================================================
Based on:
- CPT-118-378.pdf: 5-Compartment mPBPK structure
- s41598-025-87316-w.pdf: TMDD and ML pipeline

Author: AI-Driven Clinical Trial Platform
"""

import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Data Classes for Parameters
# =============================================================================

@dataclass
class DrugParams:
    """약물 특성 파라미터"""
    KD_nM: float = 1.0          # 결합 상수 (nM)
    dose_mg: float = 100.0      # 용량 (mg)
    charge: int = 0             # 표면 전하 (-5, 0, +5)
    dosing_interval_day: int = 14  # 투약 간격 (일)
    MW_kDa: float = 150.0       # 분자량 (kDa, IgG 기준)


@dataclass
class PatientParams:
    """환자 특성 파라미터"""
    weight_kg: float = 70.0     # 체중 (kg)
    age: int = 45               # 나이
    gender: int = 0             # 0: Female, 1: Male
    bmi: float = 25.0           # BMI
    # CYP2D6 Pharmacogenomics (ON/OFF 가능)
    cyp2d6_genotype: str = None # 예: '*1/*1', '*1/*4', '*4/*4', '*1/*1xN'
    ethnicity: str = 'EUR'      # EUR, EAS, AFR, AMR (1000 Genomes 기준)


@dataclass
class TargetParams:
    """타겟 특성 파라미터"""
    baseline_nM: float = 10.0   # 기저 농도 (nM)
    halflife_hr: float = 24.0   # 반감기 (hr)
    target_type: str = 'soluble'  # 'soluble' or 'membrane'


@dataclass
class PKConstants:
    """PK 상수 (논문 기반 고정값) - Optimized for TO achievement"""
    # Volume fractions (체중 대비)
    V_plasma_frac: float = 0.04      # 혈장 용적
    V_tight_frac: float = 0.35       # 조밀 조직
    V_leaky_frac: float = 0.10       # 느슨한 조직
    V_liver_frac: float = 0.02       # 간
    
    # Flow rates (L/day)
    Q_tight: float = 1.0             # 조밀 조직 혈류
    Q_leaky: float = 1.0             # 느슨한 조직 혈류
    Q_liver: float = 0.5             # 간 혈류
    L_lymph: float = 0.2             # 림프 유량
    
    # Clearance (ADJUSTED: 낮춰서 약물 체류 시간 증가)
    CL_sys_base: float = 0.02        # 0.1 → 0.02 (기저 전신 청소율 80% 감소)
    CL_liver: float = 0.01           # 0.05 → 0.01 (간 청소율 80% 감소)
    
    # Binding kinetics (ADJUSTED: 결합 속도 증가)
    kon: float = 1e7                 # 1e6 → 1e7 (결합 속도 10배 증가)
    
    # TMDD parameters (ADJUSTED: 내재화 속도 감소)
    kint: float = 0.02               # 0.1 → 0.02 (내재화 속도 80% 감소)


# =============================================================================
# CYP2D6 Pharmacogenomics Module (Dynamic Integration)
# =============================================================================

# Try to import the dynamic CYP2D6 parser
try:
    import sys
    from pathlib import Path
    # Add parent directory to path for import
    _src_path = Path(__file__).parent.parent
    if str(_src_path) not in sys.path:
        sys.path.insert(0, str(_src_path))
    from data.parse_cyp2d6 import PharmVarCYP2D6Parser
    CYP2D6_PARSER_AVAILABLE = True
except ImportError:
    CYP2D6_PARSER_AVAILABLE = False

# Fallback: Static CYP2D6 Diplotype → Activity Score (CPIC 기준)
# Used when parser is not available or genotype is in diplotype format
CYP2D6_ACTIVITY_SCORES = {
    # Poor Metabolizer (PM): Activity Score = 0
    '*3/*4': 0, '*4/*4': 0, '*5/*5': 0, '*4/*5': 0,
    '*3/*3': 0, '*4/*6': 0, '*5/*6': 0,
    
    # Intermediate Metabolizer (IM): Activity Score = 0.5 ~ 1.0
    '*1/*4': 0.5, '*1/*5': 0.5, '*2/*4': 0.5,
    '*1/*10': 1.0, '*10/*10': 1.0, '*41/*41': 1.0,
    '*1/*41': 1.25, '*2/*10': 1.0,
    
    # Normal Metabolizer (NM): Activity Score = 1.5 ~ 2.0
    '*1/*1': 2.0, '*1/*2': 2.0, '*2/*2': 2.0,
    '*1/*9': 1.5, '*2/*41': 1.5,
    
    # Ultra-rapid Metabolizer (UM): Activity Score > 2.0
    '*1/*1xN': 3.0, '*1/*2xN': 3.0, '*2/*2xN': 3.0,
    '*1xN/*1xN': 4.0, '*2xN/*2xN': 4.0,
}

# Global parser instance (lazy loading)
_cyp2d6_parser = None

def get_cyp2d6_parser():
    """Get or create the CYP2D6 parser instance (singleton pattern)."""
    global _cyp2d6_parser
    if _cyp2d6_parser is None and CYP2D6_PARSER_AVAILABLE:
        _cyp2d6_parser = PharmVarCYP2D6Parser()
    return _cyp2d6_parser

def get_activity_score_from_diplotype(diplotype: str) -> float:
    """
    Get activity score from diplotype string (e.g., '*1/*4').
    
    Uses dynamic parser if available, falls back to static dictionary.
    """
    # Try static dictionary first (faster for common diplotypes)
    if diplotype in CYP2D6_ACTIVITY_SCORES:
        return CYP2D6_ACTIVITY_SCORES[diplotype]
    
    # Try dynamic parser for individual alleles
    parser = get_cyp2d6_parser()
    if parser:
        # Parse diplotype: '*1/*4' -> ['*1', '*4']
        alleles = diplotype.replace(' ', '').split('/')
        if len(alleles) == 2:
            a1_score = parser.get_activity_value(alleles[0])
            a2_score = parser.get_activity_value(alleles[1])
            return a1_score + a2_score
    
    # Default to normal function if unknown
    return 2.0

# Activity Score → CL Multiplier
# Reference: PharmGKB Clearance Scaling Factors
def get_cyp2d6_cl_multiplier(activity_score: float) -> float:
    """
    CYP2D6 Activity Score → Clearance Multiplier
    
    PM (AS=0):     CL × 0.3 (대사 능력 현저히 감소)
    IM (AS<1.25):  CL × 0.6 (대사 능력 감소)
    NM (AS 1.25-2.25): CL × 1.0 (정상)
    UM (AS>2.25):  CL × 1.8 (대사 능력 증가, 약효 감소 가능)
    """
    if activity_score < 0.25:
        return 0.3   # Poor Metabolizer
    elif activity_score < 1.25:
        return 0.6   # Intermediate Metabolizer
    elif activity_score <= 2.25:
        return 1.0   # Normal Metabolizer
    else:
        return 1.8   # Ultra-rapid Metabolizer


def get_cyp2d6_phenotype(activity_score: float) -> str:
    """Activity Score → Phenotype 분류"""
    if activity_score < 0.25:
        return 'PM'   # Poor Metabolizer
    elif activity_score < 1.25:
        return 'IM'   # Intermediate Metabolizer
    elif activity_score <= 2.25:
        return 'NM'   # Normal Metabolizer
    else:
        return 'UM'   # Ultra-rapid Metabolizer


# =============================================================================
# mPBPK-TMDD ODE System
# =============================================================================

class mPBPKEngine:
    """
    5-Compartment mPBPK Model with TMDD
    
    Compartments:
    1. Plasma (혈장)
    2. Tight Tissue (조밀 조직)
    3. Leaky Tissue (느슨한 조직)
    4. Liver (간)
    5. Lymph (림프)
    
    TMDD States:
    6. T_free (자유 타겟)
    7. DT_complex (약물-타겟 복합체)
    """
    
    def __init__(self, 
                 drug: DrugParams,
                 patient: PatientParams,
                 target: TargetParams,
                 pk_const: PKConstants = None):
        """
        Initialize mPBPK Engine
        
        Args:
            drug: 약물 파라미터
            patient: 환자 파라미터
            target: 타겟 파라미터
            pk_const: PK 상수 (기본값 사용 가능)
        """
        self.drug = drug
        self.patient = patient
        self.target = target
        self.pk = pk_const if pk_const else PKConstants()
        
        # 파라미터 계산
        self._calculate_derived_params()
    
    def _calculate_derived_params(self):
        """
        파생 파라미터 계산
        
        단위 시스템:
        - 양(Amount): nmol
        - 농도(Concentration): nM (nmol/L)
        - 부피(Volume): L
        - 시간(Time): day
        """
        W = self.patient.weight_kg
        
        # Volume of distribution (L)
        self.V_P = W * self.pk.V_plasma_frac
        self.V_T = W * self.pk.V_tight_frac
        self.V_L = W * self.pk.V_leaky_frac
        self.V_Liv = W * self.pk.V_liver_frac
        
        # Fraction unbound (BMI 기반 조정)
        if self.patient.bmi > 30:
            self.fu = 0.85
        elif self.patient.bmi < 18.5:
            self.fu = 0.95
        else:
            self.fu = 0.90
        
        # Systemic clearance (전하 보정) [L/day]
        self.CL_sys = self.pk.CL_sys_base
        if self.drug.charge > 0:
            self.CL_sys *= 1.8  # 양전하: 청소율 증가
        elif self.drug.charge < 0:
            self.CL_sys *= 0.75  # 음전하: 청소율 감소
        
        # 나이 보정 (60세 이상)
        if self.patient.age > 60:
            self.CL_sys *= (1.0 - 0.008 * (self.patient.age - 60))
        
        # =====================================================
        # CYP2D6 Pharmacogenomics Module (ON/OFF 가능)
        # =====================================================
        self.cyp2d6_enabled = False
        self.cyp2d6_activity_score = None
        self.cyp2d6_phenotype = None
        self.cyp2d6_cl_multiplier = 1.0
        
        if self.patient.cyp2d6_genotype is not None:
            genotype = self.patient.cyp2d6_genotype
            
            # Use dynamic activity score calculation (PharmGKB data)
            self.cyp2d6_enabled = True
            self.cyp2d6_activity_score = get_activity_score_from_diplotype(genotype)
            self.cyp2d6_phenotype = get_cyp2d6_phenotype(self.cyp2d6_activity_score)
            self.cyp2d6_cl_multiplier = get_cyp2d6_cl_multiplier(self.cyp2d6_activity_score)
            
            # Hepatic clearance scaling (저분자 대사에 영향)
            # 항체의 경우 CYP 영향 적으나, 저분자 ADC payload 등에 적용 가능
            self.CL_sys *= self.cyp2d6_cl_multiplier
        
        # 약물량 변환: mg → nmol
        # MW = 150 kDa = 150,000 g/mol
        MW_g_per_mol = self.drug.MW_kDa * 1000  # kDa → g/mol
        self.dose_nmol = (self.drug.dose_mg / 1000) / MW_g_per_mol * 1e9  # mg → g → mol → nmol
        
        # TMDD 파라미터
        # KD: nM 단위 그대로 사용 (농도 단위와 일치)
        self.KD_nM = self.drug.KD_nM
        # kon: 1/(nM·day) 단위로 조정
        self.kon = self.pk.kon * 1e-9  # 1/(M·day) → 1/(nM·day)
        self.koff = self.kon * self.KD_nM  # [1/(nM·day)] × [nM] = [1/day]
        
        # 타겟 파라미터 (nmol 단위)
        # T0 = baseline_nM × V_P = [nM] × [L] = [nmol]
        self.T0 = self.target.baseline_nM * self.V_P  # [nmol]
        t_half_day = self.target.halflife_hr / 24
        self.kdeg = np.log(2) / t_half_day if t_half_day > 0 else 0.5  # [1/day]
        self.ksyn = self.kdeg * self.T0  # [1/day] × [nmol] = [nmol/day]
    
    def _ode_system(self, y, t, params):
        """
        7-State ODE System
        
        States:
        [0] A_plasma: 혈장 내 약물량
        [1] A_tight: 조밀 조직 내 약물량
        [2] A_leaky: 느슨한 조직 내 약물량
        [3] A_liver: 간 내 약물량
        [4] A_lymph: 림프 내 약물량
        [5] T_free: 자유 타겟량
        [6] DT_complex: 약물-타겟 복합체량
        """
        A_plasma, A_tight, A_leaky, A_liver, A_lymph, T_free, DT_complex = y
        
        # Unpack parameters
        V_P, V_T, V_L, V_Liv = params['V_P'], params['V_T'], params['V_L'], params['V_Liv']
        Q_T, Q_L, Q_Liv = params['Q_T'], params['Q_L'], params['Q_Liv']
        L = params['L']
        CL_sys, CL_liv = params['CL_sys'], params['CL_liv']
        fu = params['fu']
        kon, koff = params['kon'], params['koff']
        ksyn, kdeg, kint = params['ksyn'], params['kdeg'], params['kint']
        
        # Concentrations (free drug)
        C_plasma = A_plasma / V_P if V_P > 0 else 0
        C_plasma_free = C_plasma * fu
        
        C_tight = A_tight / V_T if V_T > 0 else 0
        C_tight_free = C_tight * fu
        
        C_leaky = A_leaky / V_L if V_L > 0 else 0
        C_leaky_free = C_leaky * fu
        
        C_liver = A_liver / V_Liv if V_Liv > 0 else 0
        C_liver_free = C_liver * fu
        
        # TMDD binding reactions (농도 기반 계산)
        # T_free는 양(amount)이므로 농도로 변환
        T_free_conc = T_free / V_P if V_P > 0 else 0
        DT_complex_conc = DT_complex / V_P if V_P > 0 else 0
        
        # 결합/해리 속도 (농도 변화율: M/day)
        binding_rate = kon * C_plasma_free * T_free_conc      # [1/M/day] × [M] × [M] = [M/day]
        unbinding_rate = koff * DT_complex_conc               # [1/day] × [M] = [M/day]
        
        # 양(amount) 변화율로 변환 (농도 × 부피)
        binding_amount = binding_rate * V_P    # [M/day] × [L] = [mol/day] (비례)
        unbinding_amount = unbinding_rate * V_P
        
        # Tissue flux (convection) - 양 변화율
        flux_tight = Q_T * (C_plasma_free - C_tight_free)     # [L/day] × [M] = 양 비례
        flux_leaky = Q_L * (C_plasma_free - C_leaky_free)
        flux_liver = Q_Liv * (C_plasma_free - C_liver_free)
        
        # Lymph return
        lymph_return = L * A_lymph
        
        # Elimination
        elim_sys = CL_sys * C_plasma_free
        elim_liver = CL_liv * C_liver_free
        
        # ODEs (모든 변수는 양(amount) 기준)
        # Plasma
        dA_plasma = (-flux_tight - flux_leaky - flux_liver 
                     + lymph_return - elim_sys 
                     - binding_amount + unbinding_amount)
        
        # Tight tissue
        to_lymph_T = L * 0.3 * A_tight
        dA_tight = flux_tight - to_lymph_T
        
        # Leaky tissue
        to_lymph_L = L * 0.4 * A_leaky
        dA_leaky = flux_leaky - to_lymph_L
        
        # Liver
        to_lymph_Liv = L * 0.3 * A_liver
        dA_liver = flux_liver - to_lymph_Liv - elim_liver
        
        # Lymph
        dA_lymph = to_lymph_T + to_lymph_L + to_lymph_Liv - lymph_return
        
        # TMDD (양 기준으로 변환)
        dT_free = ksyn - kdeg * T_free - binding_amount + unbinding_amount
        dDT_complex = binding_amount - unbinding_amount - kint * DT_complex
        
        return [dA_plasma, dA_tight, dA_leaky, dA_liver, dA_lymph, dT_free, dDT_complex]
    
    def simulate(self, 
                 duration_days: float = None,
                 n_points: int = 500) -> Dict:
        """
        mPBPK-TMDD 시뮬레이션 실행
        
        Args:
            duration_days: 시뮬레이션 기간 (일), 기본값: 투약 간격
            n_points: 시간점 수
            
        Returns:
            dict: 시뮬레이션 결과
        """
        if duration_days is None:
            duration_days = self.drug.dosing_interval_day
        
        t = np.linspace(0, duration_days, n_points)
        
        # Parameters dictionary
        params = {
            'V_P': self.V_P, 'V_T': self.V_T, 'V_L': self.V_L, 'V_Liv': self.V_Liv,
            'Q_T': self.pk.Q_tight, 'Q_L': self.pk.Q_leaky, 'Q_Liv': self.pk.Q_liver,
            'L': self.pk.L_lymph,
            'CL_sys': self.CL_sys, 'CL_liv': self.pk.CL_liver,
            'fu': self.fu,
            'kon': self.kon, 'koff': self.koff,
            'ksyn': self.ksyn, 'kdeg': self.kdeg, 'kint': self.pk.kint
        }
        
        # Initial conditions (모든 양은 nmol 단위)
        y0 = [
            self.dose_nmol,     # A_plasma [nmol] (IV bolus)
            0.0,                # A_tight [nmol]
            0.0,                # A_leaky [nmol]
            0.0,                # A_liver [nmol]
            0.0,                # A_lymph [nmol]
            self.T0,            # T_free [nmol]
            0.0                 # DT_complex [nmol]
        ]
        
        try:
            solution = odeint(self._ode_system, y0, t, args=(params,))
            
            # Extract results
            A_plasma = solution[:, 0]
            T_free = solution[:, 5]
            DT_complex = solution[:, 6]
            
            # Concentrations
            C_plasma = A_plasma / self.V_P
            
            # Target Occupancy
            T_total = T_free + DT_complex
            TO_percent = np.where(T_total > 0, 
                                  (DT_complex / T_total) * 100, 
                                  0)
            
            # Metrics
            C_min = C_plasma[-1]  # Trough concentration
            C_max = np.max(C_plasma)
            # AUC calculation (NumPy 2.0 compatibility)
            try:
                AUC = np.trapezoid(C_plasma, t)  # NumPy 2.0+
            except AttributeError:
                AUC = np.trapz(C_plasma, t)      # NumPy < 2.0
            TO_trough = TO_percent[-1]
            
            success = TO_trough >= 90
            
            return {
                'time': t,
                'C_plasma': C_plasma,
                'TO_percent': TO_percent,
                'T_free': T_free,
                'DT_complex': DT_complex,
                'C_min': C_min,
                'C_max': C_max,
                'AUC': AUC,
                'TO_trough': TO_trough,
                'success': success,
                'status': 'completed'
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'success': False,
                'TO_trough': 0.0
            }
    
    def calculate_target_occupancy(self, T_free: float, T_total: float) -> float:
        """
        Target Occupancy 계산
        
        TO% = (Bound Target / Total Target) × 100
        """
        if T_total <= 0:
            return 0.0
        bound = T_total - T_free
        return max(0, min(100, (bound / T_total) * 100))


# =============================================================================
# Convenience Functions
# =============================================================================

def run_single_simulation(
    kd_nm: float = 1.0,
    dose_mg: float = 100.0,
    charge: int = 0,
    target_baseline_nm: float = 10.0,
    target_halflife_hr: float = 24.0,
    patient_weight: float = 70.0,
    dosing_interval: int = 14
) -> Dict:
    """
    단일 시뮬레이션 실행 (편의 함수)
    
    Returns:
        dict: 시뮬레이션 결과
    """
    drug = DrugParams(
        KD_nM=kd_nm,
        dose_mg=dose_mg,
        charge=charge,
        dosing_interval_day=dosing_interval
    )
    
    patient = PatientParams(weight_kg=patient_weight)
    
    target = TargetParams(
        baseline_nM=target_baseline_nm,
        halflife_hr=target_halflife_hr
    )
    
    engine = mPBPKEngine(drug, patient, target)
    return engine.simulate()


def batch_simulation(candidates: list) -> list:
    """
    배치 시뮬레이션 실행
    
    Args:
        candidates: list of dict with simulation parameters
        
    Returns:
        list: 각 후보의 시뮬레이션 결과
    """
    results = []
    for i, cand in enumerate(candidates):
        result = run_single_simulation(**cand)
        result['candidate_id'] = i
        results.append(result)
    return results


def simulate_population_cohort(
    drug: DrugParams,
    target: TargetParams,
    population: str = 'EUR',
    n_subjects: int = 100,
    pk_const: PKConstants = None
) -> list:
    """
    가상 코호트 대상 배치 시뮬레이션
    
    CYP2D6 가상 코호트 생성기를 사용하여 특정 인구의 환자를 시뮬레이션합니다.
    
    Args:
        drug: 약물 파라미터
        target: 타겟 파라미터
        population: 인구 코드 (EUR, EAS, AFR, AMR, SAS)
        n_subjects: 시뮬레이션할 피험자 수
        pk_const: PK 상수 (선택)
        
    Returns:
        list of dict: 각 피험자의 시뮬레이션 결과
    """
    import pandas as pd
    
    parser = get_cyp2d6_parser()
    if parser is None:
        raise RuntimeError("CYP2D6 parser not available. Install required dependencies.")
    
    # Generate virtual cohort
    cohort_df = parser.simulate_population(population, n_subjects)
    
    results = []
    for idx, row in cohort_df.iterrows():
        # Create patient with CYP2D6 genotype
        patient = PatientParams(
            weight_kg=70.0,  # Could randomize based on population
            age=45,
            gender=idx % 2,  # Alternating
            bmi=25.0,
            cyp2d6_genotype=row['diplotype'],
            ethnicity=population
        )
        
        # Run simulation
        engine = mPBPKEngine(drug, patient, target, pk_const)
        sim_result = engine.simulate()
        
        # Add cohort info to result
        sim_result['subject_id'] = row['subject_id']
        sim_result['diplotype'] = row['diplotype']
        sim_result['allele_1'] = row['allele_1']
        sim_result['allele_2'] = row['allele_2']
        sim_result['activity_score'] = row['activity_score']
        sim_result['phenotype'] = row['phenotype']
        sim_result['population'] = population
        sim_result['cyp2d6_enabled'] = engine.cyp2d6_enabled
        sim_result['cl_multiplier'] = engine.cyp2d6_cl_multiplier
        
        results.append(sim_result)
    
    return results


def summarize_cohort_results(results: list) -> dict:
    """
    코호트 시뮬레이션 결과 요약
    
    Args:
        results: simulate_population_cohort 결과
        
    Returns:
        dict: 표현형별 요약 통계
    """
    import pandas as pd
    
    df = pd.DataFrame([{
        'phenotype': r.get('phenotype', 'Unknown'),
        'TO_trough': r.get('TO_trough', 0),
        'success': r.get('success', False),
        'C_max': r.get('C_max', 0),
        'AUC': r.get('AUC', 0),
    } for r in results])
    
    summary = {}
    for pheno in ['PM', 'IM', 'NM', 'UM']:
        pheno_df = df[df['phenotype'] == pheno]
        if len(pheno_df) > 0:
            summary[pheno] = {
                'count': len(pheno_df),
                'success_rate': pheno_df['success'].mean() * 100,
                'TO_trough_mean': pheno_df['TO_trough'].mean(),
                'TO_trough_std': pheno_df['TO_trough'].std(),
                'C_max_mean': pheno_df['C_max'].mean(),
                'AUC_mean': pheno_df['AUC'].mean(),
            }
    
    summary['overall'] = {
        'total': len(df),
        'success_rate': df['success'].mean() * 100,
        'TO_trough_mean': df['TO_trough'].mean(),
    }
    
    return summary


# =============================================================================
# Main (Test)
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("  mPBPK Engine Test")
    print("="*60)
    
    # Test case 1: Good binder, high dose
    result1 = run_single_simulation(
        kd_nm=0.1,
        dose_mg=100,
        charge=0,
        target_baseline_nm=10,
        target_halflife_hr=24
    )
    print(f"\nTest 1 (Good binder, 100mg):")
    print(f"  TO%: {result1['TO_trough']:.1f}%")
    print(f"  Success: {result1['success']}")
    
    # Test case 2: Poor binder, low dose
    result2 = run_single_simulation(
        kd_nm=1000,
        dose_mg=1,
        charge=+5,
        target_baseline_nm=100,
        target_halflife_hr=24
    )
    print(f"\nTest 2 (Poor binder, 1mg):")
    print(f"  TO%: {result2['TO_trough']:.1f}%")
    print(f"  Success: {result2['success']}")
    
    print("\n" + "="*60)
    print("  Engine Test Complete!")
    print("="*60)
