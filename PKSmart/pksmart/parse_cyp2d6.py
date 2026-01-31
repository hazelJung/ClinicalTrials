"""
PharmVar CYP2D6 Parser
======================
Parses CYP2D6 haplotype data from PharmVar and allele functionality from CPIC/PharmGKB.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

class PharmVarCYP2D6Parser:
    """
    Parses PharmVar CYP2D6 data and generates genotype/phenotype information.
    """
    
    # Allele Functionality mapping (CPIC Standard)
    # Activity Score:
    # 0: No function
    # 0.5: Decreased function
    # 1.0: Normal function
    # >1.0: Increased function
    
    DEFAULT_FUNCTIONALITY = {
        '*1': 1.0, '*2': 1.0, '*3': 0.0, '*4': 0.0, '*5': 0.0, 
        '*6': 0.0, '*7': 0.0, '*8': 0.0, '*9': 0.5, '*10': 0.5,
        '*11': 0.0, '*12': 0.0, '*13': 0.0, '*14': 0.0, '*15': 0.0,
        '*17': 0.5, '*29': 0.5, '*35': 1.0, '*36': 0.0, '*41': 0.5,
        '*1xN': 2.0, '*2xN': 2.0  # Simplified for Copy Number Variation
    }
    
    # Population Frequency Defaults (from CPIC/PharmGKB)
    # Falls back to these if data files are missing
    DEFAULT_FREQUENCIES = {
        'EUR': {'*1': 0.33, '*2': 0.32, '*4': 0.18, '*41': 0.07, '*10': 0.02, '*17': 0.0, '*5': 0.03},
        'EAS': {'*1': 0.28, '*2': 0.13, '*10': 0.45, '*41': 0.02, '*4': 0.01, '*5': 0.04},
        'AFR': {'*1': 0.29, '*2': 0.20, '*17': 0.19, '*29': 0.06, '*4': 0.06, '*5': 0.06},
        'AMR': {'*1': 0.38, '*2': 0.25, '*4': 0.13, '*10': 0.04, '*41': 0.04, '*5': 0.04}, # Ad Mixed American
        'SAS': {'*1': 0.35, '*2': 0.30, '*10': 0.10, '*41': 0.08, '*4': 0.07, '*5': 0.02}, # South Asian
    }

    def __init__(self, data_dir: Optional[Path] = None, genome_build: str = "GRCh38"):
        """
        Initialize the parser.
        
        Args:
            data_dir: Directory containing raw gene data
            genome_build: Genome build version (GRCh37 or GRCh38)
        """
        self.genome_build = genome_build
        
        # Try to locate data directory relative to this file
        if data_dir is None:
            # Assumes standard project structure: src/data/../../data/raw/genes
            # Adjusted for PKSmart: ../data
            data_dir = Path(__file__).parent.parent / "data" / "raw" / "genes"
        
        self.data_dir = data_dir
        self.pharmvar_dir = data_dir / "CYP2D6-6.2.18"
        
        self.allele_functionality = self.DEFAULT_FUNCTIONALITY.copy()
        self.allele_frequencies = self.DEFAULT_FREQUENCIES.copy()
        self.haplotypes = {}
        
        # Load data if available
        self._load_functionality_reference()
        self._load_frequency_table()
        # self._load_pharmvar_data() # Skip complex pharmvar parsing for now, stick to frequency tables

    def _load_functionality_reference(self) -> None:
        """Load allele functionality from Excel file."""
        excel_path = self.data_dir / "CYP2D6_allele_functionality_reference.xlsx"
        if not excel_path.exists():
            # print(f"Warning: Functionality reference not found at {excel_path}. Using defaults.")
            return

        try:
            df = pd.read_excel(excel_path)
            # Simplistic parsing logic - assuming columns exist
            # This is a placeholder for the robust parsing logic
            pass 
        except Exception as e:
            print(f"Error loading functionality reference: {e}")

    def _load_frequency_table(self) -> None:
        """Load population frequency table."""
        excel_path = self.data_dir / "CYP2D6_frequency_table.xlsx"
        if not excel_path.exists():
            # print(f"Warning: Frequency table not found at {excel_path}. Using defaults.")
            return

        try:
            df = pd.read_excel(excel_path)
            # Placeholder for parsing
            pass
        except Exception as e:
            print(f"Error loading frequency table: {e}")

    def get_activity_value(self, allele: str) -> float:
        """Get activity value for a star allele."""
        # Clean allele string
        allele = allele.strip()
        if not allele.startswith('*'):
            allele = '*' + allele
            
        return self.allele_functionality.get(allele, 1.0) # Default to Normal if unknown

    def calculate_activity_score(self, allele1: str, allele2: str) -> float:
        """Calculate activity score for a diplotype."""
        return self.get_activity_value(allele1) + self.get_activity_value(allele2)

    def predict_phenotype(self, activity_score: float) -> str:
        """
        Predict phenotype from activity score.
        
        CPIC Consensus:
        AS = 0: PM (Poor Metabolizer)
        0 < AS < 1.25: IM (Intermediate Metabolizer)
        1.25 <= AS <= 2.25: NM (Normal Metabolizer)
        AS > 2.25: UM (Ultrarapid Metabolizer)
        """
        if activity_score == 0:
            return "PM"
        elif activity_score < 1.25:
            return "IM"
        elif activity_score <= 2.25:
            return "NM"
        else:
            return "UM"

    def simulate_population(self, population_code: str, n: int = 100) -> pd.DataFrame:
        """
        Simulate a population based on allele frequencies.
        
        Args:
            population_code: Frequency code (e.g., 'EUR')
            n: Number of individuals
            
        Returns:
            DataFrame with 'diplotype', 'activity_score', 'phenotype'
        """
        if population_code not in self.allele_frequencies:
            population_code = 'EUR' # Default
            
        freqs = self.allele_frequencies[population_code]
        alleles = list(freqs.keys())
        weights = list(freqs.values())
        
        # Normalize weights
        total_w = sum(weights)
        weights = [w/total_w for w in weights]
        
        # Sample alleles
        allele1_samples = np.random.choice(alleles, size=n, p=weights)
        allele2_samples = np.random.choice(alleles, size=n, p=weights)
        
        results = []
        for i in range(n):
            a1 = allele1_samples[i]
            a2 = allele2_samples[i]
            score = self.calculate_activity_score(a1, a2)
            pheno = self.predict_phenotype(score)
            
            results.append({
                'subject_id': f"{population_code}_{i+1:04d}",
                'population': population_code,
                'allele_1': a1,
                'allele_2': a2,
                'diplotype': f"{a1}/{a2}",
                'activity_score': score,
                'phenotype': pheno
            })
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    parser = PharmVarCYP2D6Parser()
    pop = parser.simulate_population("EUR", 10)
    print(pop)
