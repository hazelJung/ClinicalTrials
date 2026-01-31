"""
mPBPK Engine - Minimal Physiologically Based Pharmacokinetic Model
===================================================================
5-Compartment mPBPK structure with TMDD and CYP2D6 Integration.
"""

import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import warnings
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Import CYP2D6 Parser from pksmart package
try:
    from pksmart.parse_cyp2d6 import PharmVarCYP2D6Parser
    CYP2D6_PARSER_AVAILABLE = True
except ImportError:
    # Fallback if pksmart isn't installed as package yet (e.g. running from script folder)
    try:
        from parse_cyp2d6 import PharmVarCYP2D6Parser
        CYP2D6_PARSER_AVAILABLE = True
    except ImportError:
        CYP2D6_PARSER_AVAILABLE = False


# =============================================================================
# Data Classes for Parameters
# =============================================================================

@dataclass
class DrugParams:
    """Drug Parameters"""
    KD_nM: float = 1.0          # Binding constant (nM)
    dose_mg: float = 100.0      # Dose (mg)
    charge: int = 0             # Surface charge (-5, 0, +5)
    dosing_interval_day: int = 14
    MW_kDa: float = 150.0       # Molecular Weight (kDa)


@dataclass
class PatientParams:
    """Patient Parameters"""
    weight_kg: float = 70.0
    age: int = 45
    gender: int = 0             # 0: Female, 1: Male
    bmi: float = 25.0
    # CYP2D6 Pharmacogenomics
    cyp2d6_genotype: str = None # e.g., '*1/*1', '*1/*4'
    ethnicity: str = 'EUR'


@dataclass
class TargetParams:
    """Target Parameters"""
    baseline_nM: float = 10.0
    halflife_hr: float = 24.0
    target_type: str = 'soluble'


@dataclass
class PKConstants:
    """PK Constants (Fixed/Optimized)"""
    # Volume fractions
    V_plasma_frac: float = 0.04
    V_tight_frac: float = 0.35
    V_leaky_frac: float = 0.10
    V_liver_frac: float = 0.02
    
    # Flow rates (L/day)
    Q_tight: float = 1.0
    Q_leaky: float = 1.0
    Q_liver: float = 0.5
    L_lymph: float = 0.2
    
    # Clearance & Kinetics
    CL_sys_base: float = 0.02
    CL_liver: float = 0.01
    kon: float = 1e7
    kint: float = 0.02


# =============================================================================
# CYP2D6 Pharmacogenomics Logic
# =============================================================================

CYP2D6_ACTIVITY_SCORES = {
    '*3/*4': 0, '*4/*4': 0, '*5/*5': 0, '*4/*5': 0,
    '*3/*3': 0, '*4/*6': 0, '*5/*6': 0,
    '*1/*4': 0.5, '*1/*5': 0.5, '*2/*4': 0.5,
    '*1/*10': 1.0, '*10/*10': 1.0, '*41/*41': 1.0,
    '*1/*41': 1.25, '*2/*10': 1.0,
    '*1/*1': 2.0, '*1/*2': 2.0, '*2/*2': 2.0,
    '*1/*9': 1.5, '*2/*41': 1.5,
    '*1/*1xN': 3.0, '*1/*2xN': 3.0, '*2/*2xN': 3.0,
    '*1xN/*1xN': 4.0, '*2xN/*2xN': 4.0,
}

_cyp2d6_parser = None

def get_cyp2d6_parser():
    global _cyp2d6_parser
    if _cyp2d6_parser is None and CYP2D6_PARSER_AVAILABLE:
        _cyp2d6_parser = PharmVarCYP2D6Parser()
    return _cyp2d6_parser

def get_activity_score_from_diplotype(diplotype: str) -> float:
    if diplotype in CYP2D6_ACTIVITY_SCORES:
        return CYP2D6_ACTIVITY_SCORES[diplotype]
    
    parser = get_cyp2d6_parser()
    if parser:
        alleles = diplotype.replace(' ', '').split('/')
        if len(alleles) == 2:
            a1 = parser.get_activity_value(alleles[0])
            a2 = parser.get_activity_value(alleles[1])
            return a1 + a2
    return 2.0

def get_cyp2d6_cl_multiplier(activity_score: float) -> float:
    if activity_score < 0.25: return 0.3
    elif activity_score < 1.25: return 0.6
    elif activity_score <= 2.25: return 1.0
    else: return 1.8

def get_cyp2d6_phenotype(activity_score: float) -> str:
    if activity_score < 0.25: return 'PM'
    elif activity_score < 1.25: return 'IM'
    elif activity_score <= 2.25: return 'NM'
    else: return 'UM'


# =============================================================================
# mPBPK Engine Class
# =============================================================================

class mPBPKEngine:
    def __init__(self, drug: DrugParams, patient: PatientParams, target: TargetParams, pk_const: PKConstants = None,
                 predicted_CL_L_day: float = None, predicted_Vd_L: float = None, predicted_Fu: float = None):
        self.drug = drug
        self.patient = patient
        self.target = target
        self.pk = pk_const if pk_const else PKConstants()
        
        # Overrides from ML Predictor
        self.predicted_CL_L_day = predicted_CL_L_day
        self.predicted_Vd_L = predicted_Vd_L
        self.predicted_Fu = predicted_Fu
        
        self._calculate_derived_params()
    
    def _calculate_derived_params(self):
        W = self.patient.weight_kg
        
        # 1. Volume of Distribution Logic
        if self.predicted_Vd_L:
            # If Vd is provided, scale internal compartments
            base_total_ratio = self.pk.V_plasma_frac + self.pk.V_tight_frac + self.pk.V_leaky_frac + self.pk.V_liver_frac
            scale = self.predicted_Vd_L / (base_total_ratio * W)
            
            self.V_P = W * self.pk.V_plasma_frac * scale
            self.V_T = W * self.pk.V_tight_frac * scale
            self.V_L = W * self.pk.V_leaky_frac * scale
            self.V_Liv = W * self.pk.V_liver_frac * scale
        else:
            # Default internal calculation
            self.V_P = W * self.pk.V_plasma_frac
            self.V_T = W * self.pk.V_tight_frac
            self.V_L = W * self.pk.V_leaky_frac
            self.V_Liv = W * self.pk.V_liver_frac
        
        # Fraction unbound
        if self.predicted_Fu is not None:
             self.fu = self.predicted_Fu
        else:
             # Based on BMI
             if self.patient.bmi > 30: self.fu = 0.85
             elif self.patient.bmi < 18.5: self.fu = 0.95
             else: self.fu = 0.90
        
        # 2. Clearance Logic
        if self.predicted_CL_L_day:
            # If CL is provided (L/day)
            self.CL_sys = self.predicted_CL_L_day
            
            # Apply Age adjustment if needed (or assume predicted accounts for it? PKSmart has age feature?)
            # PKSmart features are predominantly chemical. Physiological features are implicit or standard.
            # Let's apply physiological adjustments on TOP of the base predicted CL (which we assume is for a standard adult)
            # BUT: if predicted_CL is specific to this patient (i.e. we passed patient features to PKSmart), then we shouldn't adjust.
            # PKSmart predicts for "Human" (General). It doesn't take Age as input. 
            # So we SHOULD likely apply Age/Genotype adjustments on top of the "Standard Human CL".
            
            # HOWEVER: The predicted CL usually represents "Average Human".
            # So we treat it as the BASE CL.
            pass
        else:
            self.CL_sys = self.pk.CL_sys_base
            if self.drug.charge > 0: self.CL_sys *= 1.8
            elif self.drug.charge < 0: self.CL_sys *= 0.75
        
        # Age Adjustment (Apply to both predicted and default base)
        if self.patient.age > 60:
            self.CL_sys *= (1.0 - 0.008 * (self.patient.age - 60))
        
        # CYP2D6 Logic
        self.cyp2d6_enabled = False
        self.cyp2d6_cl_multiplier = 1.0
        
        if self.patient.cyp2d6_genotype is not None:
            self.cyp2d6_enabled = True
            as_score = get_activity_score_from_diplotype(self.patient.cyp2d6_genotype)
            self.cyp2d6_cl_multiplier = get_cyp2d6_cl_multiplier(as_score)
            self.CL_sys *= self.cyp2d6_cl_multiplier
        
        # Dose conversion to nmol
        MW_g_per_mol = self.drug.MW_kDa * 1000
        self.dose_nmol = (self.drug.dose_mg / 1000) / MW_g_per_mol * 1e9
        
        # TMDD parameters
        self.KD_nM = self.drug.KD_nM
        self.kon = self.pk.kon * 1e-9
        self.koff = self.kon * self.KD_nM
        
        self.T0 = self.target.baseline_nM * self.V_P
        t_half_day = self.target.halflife_hr / 24
        self.kdeg = np.log(2) / t_half_day if t_half_day > 0 else 0.5
        self.ksyn = self.kdeg * self.T0
    
    def _ode_system(self, y, t, params):
        A_plasma, A_tight, A_leaky, A_liver, A_lymph, T_free, DT_complex = y
        
        # Retrieve params
        p = params # Alias
        
        C_plasma = A_plasma / p['V_P'] if p['V_P'] > 0 else 0
        C_plasma_free = C_plasma * p['fu']
        
        C_tight_free = (A_tight / p['V_T']) * p['fu'] if p['V_T'] > 0 else 0
        C_leaky_free = (A_leaky / p['V_L']) * p['fu'] if p['V_L'] > 0 else 0
        C_liver_free = (A_liver / p['V_Liv']) * p['fu'] if p['V_Liv'] > 0 else 0
        
        T_free_conc = T_free / p['V_P'] if p['V_P'] > 0 else 0
        DT_complex_conc = DT_complex / p['V_P'] if p['V_P'] > 0 else 0
        
        binding_rate = p['kon'] * C_plasma_free * T_free_conc
        unbinding_rate = p['koff'] * DT_complex_conc
        
        binding_amount = binding_rate * p['V_P']
        unbinding_amount = unbinding_rate * p['V_P']
        
        flux_tight = p['Q_T'] * (C_plasma_free - C_tight_free)
        flux_leaky = p['Q_L'] * (C_plasma_free - C_leaky_free)
        flux_liver = p['Q_Liv'] * (C_plasma_free - C_liver_free)
        
        lymph_return = p['L'] * A_lymph
        elim_sys = p['CL_sys'] * C_plasma_free
        elim_liver = p['CL_liv'] * C_liver_free
        
        dA_plasma = -flux_tight - flux_leaky - flux_liver + lymph_return - elim_sys - binding_amount + unbinding_amount
        
        to_lymph_T = p['L'] * 0.3 * A_tight
        dA_tight = flux_tight - to_lymph_T
        
        to_lymph_L = p['L'] * 0.4 * A_leaky
        dA_leaky = flux_leaky - to_lymph_L
        
        to_lymph_Liv = p['L'] * 0.3 * A_liver
        dA_liver = flux_liver - to_lymph_Liv - elim_liver
        
        dA_lymph = to_lymph_T + to_lymph_L + to_lymph_Liv - lymph_return
        
        dT_free = p['ksyn'] - p['kdeg'] * T_free - binding_amount + unbinding_amount
        dDT_complex = binding_amount - unbinding_amount - p['kint'] * DT_complex
        
        return [dA_plasma, dA_tight, dA_leaky, dA_liver, dA_lymph, dT_free, dDT_complex]

    def simulate(self, duration_days: float = None, n_points: int = 500) -> Dict:
        if duration_days is None:
            duration_days = self.drug.dosing_interval_day
            
        t = np.linspace(0, duration_days, n_points)
        
        params = {
            'V_P': self.V_P, 'V_T': self.V_T, 'V_L': self.V_L, 'V_Liv': self.V_Liv,
            'Q_T': self.pk.Q_tight, 'Q_L': self.pk.Q_leaky, 'Q_Liv': self.pk.Q_liver,
            'L': self.pk.L_lymph,
            'CL_sys': self.CL_sys, 'CL_liv': self.pk.CL_liver,
            'fu': self.fu,
            'kon': self.kon, 'koff': self.koff,
            'ksyn': self.ksyn, 'kdeg': self.kdeg, 'kint': self.pk.kint
        }
        
        y0 = [self.dose_nmol, 0.0, 0.0, 0.0, 0.0, self.T0, 0.0]
        
        try:
            solution = odeint(self._ode_system, y0, t, args=(params,))
            
            A_plasma = solution[:, 0]
            A_tight = solution[:, 1]
            A_leaky = solution[:, 2]
            A_liver = solution[:, 3]
            
            T_free = solution[:, 5]
            DT_complex = solution[:, 6]
            
            C_plasma = A_plasma / self.V_P
            C_tight = A_tight / self.V_T
            C_leaky = A_leaky / self.V_L
            C_liver = A_liver / self.V_Liv
            
            T_total = T_free + DT_complex
            TO_percent = np.where(T_total > 0, (DT_complex / T_total) * 100, 0)
            
            C_min = C_plasma[-1]
            C_max = np.max(C_plasma)
            try:
                AUC = np.trapezoid(C_plasma, t)
            except AttributeError:
                AUC = np.trapz(C_plasma, t)
                
            TO_trough = TO_percent[-1]
            success = TO_trough >= 90
            
            # Step 4: Advanced Metrics Calculation
            # 1. T > TO90 (%)
            t_above_90 = np.sum(TO_percent >= 90) / len(TO_percent) * 100
            
            # 2. Mean TO (%)
            mean_to = np.mean(TO_percent)
            
            # 3. Safety Margin (Assuming Toxicity Threshold = 50 uM = 50000 nM as default, or calculated externally)
            # This is a placeholder default. In production, this should come from project data.
            tox_threshold_nM = 50000.0 
            safety_margin = tox_threshold_nM / C_max if C_max > 0 else 999.0

            return {
                'time': t, 
                'C_plasma': C_plasma, 
                'C_tight': C_tight,
                'C_leaky': C_leaky,
                'C_liver': C_liver,
                'TO_percent': TO_percent,
                'C_min': C_min, 'C_max': C_max, 'AUC': AUC,
                'TO_trough': TO_trough, 'success': success, 
                't_above_90': t_above_90,
                'mean_to': mean_to,
                'safety_margin': safety_margin,
                'status': 'completed'
            }
        except Exception as e:
            return {'status': 'failed', 'error': str(e), 'success': False}

# Convenience wrappers...
def run_single_simulation(**kwargs):
    # Mapping kwargs to simple sim...
    pass 

def simulate_population_cohort(drug: DrugParams, target: TargetParams, population: str = 'EUR', n_subjects: int = 100,
                               predicted_CL_L_day: float = None, predicted_Vd_L: float = None, predicted_Fu: float = None,
                               gender_mode: str = 'BOTH', age_min: int = 18, age_max: int = 80, female_ratio: float = 0.5) -> list:
    """
    Simulate a population cohort with demographic variability.
    
    Args:
        gender_mode: 'M' (Male), 'F' (Female), or 'BOTH' (Mix)
        age_min: Minimum age
        age_max: Maximum age
        female_ratio: Proportion of females (0.0 to 1.0) when mode is 'BOTH'
    """
    # Uses parser and loops through patients
    parser = get_cyp2d6_parser()
    if not parser:
         # Simplified fallback
         return []
         
    cohort_df = parser.simulate_population(population, n_subjects)
    
    results = []
    
    # Pre-generate demographics
    ages = np.random.randint(age_min, age_max + 1, size=n_subjects)
    
    if gender_mode == 'M':
        genders = [1] * n_subjects # 1: Male
    elif gender_mode == 'F':
        genders = [0] * n_subjects # 0: Female
    else:
        # Weighted random choice based on female_ratio
        # 0: Female, 1: Male
        # p(0) = female_ratio, p(1) = 1 - female_ratio
        p_female = max(0.0, min(1.0, female_ratio))
        p_male = 1.0 - p_female
        genders = np.random.choice([0, 1], size=n_subjects, p=[p_female, p_male])
        
    for idx, row in cohort_df.iterrows():
        # Assign demographics
        age = int(ages[idx])
        gender = int(genders[idx])
        
        # Weight Generation (Simple Logic)
        # Male: Mean 78kg, Std 12kg
        # Female: Mean 65kg, Std 10kg
        if gender == 1:
            weight = np.random.normal(78.0, 12.0)
        else:
            weight = np.random.normal(65.0, 10.0)
            
        weight = max(40.0, min(150.0, weight)) # Clip weight to realistic bounds
        
        patient = PatientParams(
            weight_kg=weight,
            age=age,
            gender=gender,
            cyp2d6_genotype=row['diplotype'],
            ethnicity=population
        )
        engine = mPBPKEngine(drug, patient, target, 
                             predicted_CL_L_day=predicted_CL_L_day, 
                             predicted_Vd_L=predicted_Vd_L,
                             predicted_Fu=predicted_Fu)
        res = engine.simulate()
        res['phenotype'] = row['phenotype']
        res['population'] = population
        res['age'] = age
        res['gender'] = 'Male' if gender == 1 else 'Female'
        res['weight'] = weight
        results.append(res)
    return results
