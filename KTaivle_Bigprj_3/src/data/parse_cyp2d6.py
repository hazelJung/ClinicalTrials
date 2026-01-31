"""
CYP2D6 PharmVar Data Parser

PharmVar (Pharmacogene Variation Consortium) 공식 데이터를 파싱하여
CYP2D6 Star Allele 정의, 변이 매핑, 인종별 빈도 데이터를 제공합니다.

Data Source:
- PharmVar 6.2.18: https://www.pharmvar.org/gene/CYP2D6
- PharmGKB: CYP2D6 Frequency Table, Allele Functionality Reference
- Population Frequency: PharmGKB, CPIC Guidelines

References:
- Gaedigk et al. (2017) - The Pharmacogene Variation Consortium
- CPIC Guidelines for CYP2D6
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import random


@dataclass
class StarAllele:
    """CYP2D6 Star Allele 정보"""

    name: str  # e.g., "*1", "*2", "*4"
    function: str  # "normal function", "decreased function", "no function"
    activity_value: float  # 0.0, 0.5, 1.0, 2.0
    defining_variants: List[str]  # rsID 리스트


@dataclass
class Variant:
    """유전자 변이 정보"""

    rsid: str
    chromosome: str
    position: int
    ref_allele: str
    alt_allele: str
    variant_type: str  # substitution, deletion, insertion


class PharmVarCYP2D6Parser:
    """PharmVar CYP2D6 데이터 파서 - PharmGKB Excel 데이터 지원"""

    # Fallback: Star Allele 기능 분류 (CPIC 기준) - Excel 로딩 실패 시 사용
    DEFAULT_ALLELE_FUNCTION = {
        "*1": ("normal function", 1.0),
        "*2": ("normal function", 1.0),
        "*3": ("no function", 0.0),
        "*4": ("no function", 0.0),
        "*5": ("no function", 0.0),
        "*6": ("no function", 0.0),
        "*9": ("decreased function", 0.5),
        "*10": ("decreased function", 0.25),
        "*17": ("decreased function", 0.5),
        "*29": ("decreased function", 0.5),
        "*41": ("decreased function", 0.5),
        # CNV (Gene Duplication) - Activity > 2.0 for UM phenotype
        "*1xN": ("increased function", 2.0),  # Gene duplication of normal allele
        "*2xN": ("increased function", 2.0),  # Gene duplication of normal allele
    }

    # Fallback: 인종별 주요 대립유전자 빈도 - Excel 로딩 실패 시 사용
    DEFAULT_POPULATION_FREQUENCIES = {
        "EUR": {
            "*1": 0.38,
            "*2": 0.25,
            "*3": 0.02,
            "*4": 0.19,
            "*5": 0.03,
            "*6": 0.01,
            "*9": 0.02,
            "*10": 0.02,
            "*17": 0.00,
            "*41": 0.10,
        },
        "EAS": {
            "*1": 0.25,
            "*2": 0.15,
            "*3": 0.00,
            "*4": 0.01,
            "*5": 0.06,
            "*6": 0.00,
            "*9": 0.00,
            "*10": 0.45,
            "*17": 0.00,
            "*41": 0.03,
        },
        "AFR": {
            "*1": 0.35,
            "*2": 0.28,
            "*3": 0.00,
            "*4": 0.07,
            "*5": 0.04,
            "*6": 0.01,
            "*9": 0.01,
            "*10": 0.05,
            "*17": 0.20,
            "*41": 0.04,
        },
        "AMR": {
            "*1": 0.42,
            "*2": 0.20,
            "*3": 0.01,
            "*4": 0.12,
            "*5": 0.03,
            "*6": 0.01,
            "*9": 0.02,
            "*10": 0.08,
            "*17": 0.03,
            "*41": 0.06,
        },
        "SAS": {
            "*1": 0.40,
            "*2": 0.18,
            "*3": 0.01,
            "*4": 0.08,
            "*5": 0.04,
            "*6": 0.00,
            "*9": 0.01,
            "*10": 0.15,
            "*17": 0.02,
            "*41": 0.08,
        },
    }

    # Population column mapping from Excel to standard codes
    POPULATION_COLUMN_MAP = {
        "African American": "AFR",
        "American": "AMR",  # Note: 'American' in PharmGKB often refers to Admixed American
        "August": "AMR",  # Common Excel parsing error for 'Amr'
        "European": "EUR",
        "East Asian": "EAS",
        "Latino": "AMR",
        "Near Eastern": "SAS",
        "Sub-Saharan African": "AFR",
        "Oceanian": "OCE",
        "Central/South Asian": "SAS",
        "Caucasian": "EUR",  # Keeping for backward compatibility
    }

    POPULATIONS = ["EUR", "EAS", "AFR", "AMR", "SAS"]
    POPULATION_NAMES = {
        "EUR": "European",
        "EAS": "East Asian",
        "AFR": "African",
        "AMR": "Admixed American",
        "SAS": "South Asian",
    }

    def __init__(self, data_dir: Optional[Path] = None, genome_build: str = "GRCh38"):
        """
        Initialize PharmVar CYP2D6 parser.

        Args:
            data_dir: Path to raw genes data directory
            genome_build: Reference genome build (GRCh37 or GRCh38)
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "genes"

        self.data_dir = Path(data_dir)
        self.genome_build = genome_build
        self.pharmvar_dir = self.data_dir / "CYP2D6-6.2.18"

        # Data containers
        self.haplotypes_df: Optional[pd.DataFrame] = None
        self.star_alleles: Dict[str, StarAllele] = {}
        self.variants: Dict[str, Variant] = {}

        # Dynamic data from PharmGKB Excel files
        self.allele_function: Dict[str, Tuple[str, float]] = {}
        self.population_frequencies: Dict[str, Dict[str, float]] = {}

        # Load data
        self._load_functionality_reference()
        self._load_frequency_table()
        self._load_pharmvar_data()

    def _load_functionality_reference(self) -> None:
        """Load allele functionality from PharmGKB Excel file."""
        excel_path = self.data_dir / "CYP2D6_allele_functionality_reference.xlsx"

        if not excel_path.exists():
            print(f"[WARN] Functionality reference not found: {excel_path}")
            print("       Using default ALLELE_FUNCTION")
            self.allele_function = self.DEFAULT_ALLELE_FUNCTION.copy()
            return

        try:
            df = pd.read_excel(excel_path)

            # Identify columns (first column = allele, second = activity value)
            allele_col = df.columns[0]
            activity_col = df.columns[1]

            loaded_count = 0
            for _, row in df.iterrows():
                allele = str(row[allele_col]).strip()
                activity_raw = row[activity_col]

                # Skip header rows or invalid entries
                if not allele.startswith("*") or pd.isna(activity_raw):
                    continue

                # Parse activity value
                try:
                    if isinstance(activity_raw, str):
                        # Handle "≥3.0" format
                        if "≥" in activity_raw or ">=" in activity_raw:
                            activity_value = 3.0
                        else:
                            activity_value = float(activity_raw)
                    else:
                        activity_value = float(activity_raw)
                except (ValueError, TypeError):
                    activity_value = 1.0  # Default to normal function

                # Determine function label based on activity value
                if activity_value == 0.0:
                    function_label = "no function"
                elif activity_value < 1.0:
                    function_label = "decreased function"
                elif activity_value == 1.0:
                    function_label = "normal function"
                else:
                    function_label = "increased function"

                self.allele_function[allele] = (function_label, activity_value)
                loaded_count += 1

            print(f"[OK] Loaded {loaded_count} alleles from functionality reference")

            # Inject CNV allele activity values (not in PharmGKB Excel)
            CNV_ACTIVITY = {
                "*1xN": ("increased function", 2.0),
                "*2xN": ("increased function", 2.0),
            }
            for allele, func_info in CNV_ACTIVITY.items():
                if allele not in self.allele_function:
                    self.allele_function[allele] = func_info

        except Exception as e:
            print(f"[WARN] Error loading functionality reference: {e}")
            print("       Using default ALLELE_FUNCTION")
            self.allele_function = self.DEFAULT_ALLELE_FUNCTION.copy()

    def _load_frequency_table(self) -> None:
        """Load population frequencies from PharmGKB Excel file."""
        excel_path = self.data_dir / "CYP2D6_frequency_table.xlsx"

        if not excel_path.exists():
            print(f"[WARN] Frequency table not found: {excel_path}")
            print("       Using default POPULATION_FREQUENCIES")
            self.population_frequencies = self.DEFAULT_POPULATION_FREQUENCIES.copy()
            return

        try:
            # Skip first row (descriptive text), actual headers are in row 1
            df = pd.read_excel(excel_path, header=1)

            # Initialize frequency dictionaries
            for pop in self.POPULATIONS:
                self.population_frequencies[pop] = {}

            # Identify allele column (first column)
            allele_col = df.columns[0]

            # Map Excel columns to population codes
            col_to_pop = {}
            for col in df.columns[1:]:
                col_str = str(col)
                for excel_name, pop_code in self.POPULATION_COLUMN_MAP.items():
                    if excel_name.lower() in col_str.lower():
                        col_to_pop[col] = pop_code
                        break

            # Parse frequency data
            loaded_count = 0
            for _, row in df.iterrows():
                allele = str(row[allele_col]).strip()

                # Skip non-allele rows
                if not allele.startswith("*"):
                    continue

                # Clean allele name (remove xN for CNV)
                base_allele = allele.split("x")[0] if "x" in allele else allele

                for col, pop_code in col_to_pop.items():
                    if pop_code not in self.population_frequencies:
                        continue

                    freq_raw = row[col]

                    try:
                        if pd.notna(freq_raw):
                            freq = float(freq_raw)
                            # Accumulate frequencies for base allele
                            if base_allele in self.population_frequencies[pop_code]:
                                self.population_frequencies[pop_code][base_allele] += (
                                    freq
                                )
                            else:
                                self.population_frequencies[pop_code][base_allele] = (
                                    freq
                                )
                                loaded_count += 1
                    except (ValueError, TypeError):
                        continue

            # Normalize frequencies for each population
            for pop in self.POPULATIONS:
                if self.population_frequencies[pop]:
                    total = sum(self.population_frequencies[pop].values())
                    if total > 0:
                        self.population_frequencies[pop] = {
                            k: v / total
                            for k, v in self.population_frequencies[pop].items()
                        }

            print(
                f"[OK] Loaded frequencies for {len(self.population_frequencies)} populations ({loaded_count} allele-pop pairs)"
            )

            # =====================================================
            # CNV (Gene Duplication) Frequencies - Literature Based
            # =====================================================
            # PharmGKB Excel does not include CNV data (*1xN, *2xN).
            # Adding literature-based frequencies for UM phenotype support.
            # References:
            # - Gaedigk et al. (2017) Pharmacogenomics J
            # - CPIC CYP2D6 Guideline Supplement
            # - Ingelman-Sundberg et al. (2007) TIPS
            CNV_FREQUENCIES = {
                # EUR: ~3-5% UM carriers (mainly *1xN, *2xN)
                "EUR": {"*1xN": 0.015, "*2xN": 0.020},
                # EAS: ~1% UM (rare gene duplications)
                "EAS": {"*1xN": 0.005, "*2xN": 0.005},
                # AFR: Higher UM frequency, especially Ethiopian populations
                "AFR": {"*1xN": 0.030, "*2xN": 0.025},
                # AMR: ~2% UM (admixed)
                "AMR": {"*1xN": 0.010, "*2xN": 0.010},
                # SAS: ~2% UM
                "SAS": {"*1xN": 0.010, "*2xN": 0.010},
            }

            # Inject CNV frequencies and re-normalize
            for pop in self.POPULATIONS:
                if pop in CNV_FREQUENCIES and pop in self.population_frequencies:
                    for cnv_allele, cnv_freq in CNV_FREQUENCIES[pop].items():
                        self.population_frequencies[pop][cnv_allele] = cnv_freq

                    # Re-normalize after adding CNV
                    total = sum(self.population_frequencies[pop].values())
                    if total > 0:
                        self.population_frequencies[pop] = {
                            k: v / total
                            for k, v in self.population_frequencies[pop].items()
                        }

            print(f"[OK] Injected CNV frequencies for UM phenotype support")

        except Exception as e:
            print(f"[WARN] Error loading frequency table: {e}")
            print("       Using default POPULATION_FREQUENCIES")
            self.population_frequencies = self.DEFAULT_POPULATION_FREQUENCIES.copy()

    def _load_pharmvar_data(self) -> None:
        """Load PharmVar haplotype definition file."""
        tsv_path = (
            self.pharmvar_dir
            / self.genome_build
            / f"CYP2D6.NC_000022.11.haplotypes.tsv"
        )

        if not tsv_path.exists():
            print(f"[WARN] PharmVar TSV not found: {tsv_path}")
            return

        # Load TSV file
        self.haplotypes_df = pd.read_csv(tsv_path, sep="\t", comment="#", dtype=str)

        print(f"[OK] Loaded PharmVar haplotypes: {len(self.haplotypes_df)} rows")

        # Parse star alleles and variants
        self._parse_star_alleles()
        self._parse_variants()

    def _parse_star_alleles(self) -> None:
        """Parse unique star alleles from haplotype data."""
        if self.haplotypes_df is None:
            return

        unique_alleles = self.haplotypes_df["Haplotype Name"].unique()

        for allele_name in unique_alleles:
            # Extract base star allele (e.g., "*2" from "*2.001")
            base_allele = allele_name.split(".")[0]

            if base_allele not in self.star_alleles:
                # Get function and activity from loaded data or fallback
                func_info = self.allele_function.get(
                    base_allele,
                    self.DEFAULT_ALLELE_FUNCTION.get(
                        base_allele, ("uncertain function", 1.0)
                    ),
                )

                # Get defining variants for this allele
                allele_rows = self.haplotypes_df[
                    self.haplotypes_df["Haplotype Name"].str.startswith(base_allele)
                ]
                rsids = allele_rows["rsID"].dropna().unique().tolist()

                self.star_alleles[base_allele] = StarAllele(
                    name=base_allele,
                    function=func_info[0],
                    activity_value=func_info[1],
                    defining_variants=rsids,
                )

        print(f"[OK] Parsed {len(self.star_alleles)} unique star alleles")

    def _parse_variants(self) -> None:
        """Parse unique variants from haplotype data."""
        if self.haplotypes_df is None:
            return

        for _, row in self.haplotypes_df.iterrows():
            rsid = row.get("rsID", "")
            if pd.isna(rsid) or rsid == "" or rsid == "-":
                continue

            if rsid not in self.variants:
                self.variants[rsid] = Variant(
                    rsid=rsid,
                    chromosome="22",
                    position=int(row.get("Variant Start", 0))
                    if pd.notna(row.get("Variant Start"))
                    else 0,
                    ref_allele=str(row.get("Reference Allele", "")),
                    alt_allele=str(row.get("Variant Allele", "")),
                    variant_type=str(row.get("Type", "substitution")),
                )

        print(f"[OK] Parsed {len(self.variants)} unique variants")

    def get_star_allele_info(self, allele: str) -> Optional[StarAllele]:
        """Get information for a specific star allele."""
        base_allele = allele.split(".")[0]
        return self.star_alleles.get(base_allele)

    def get_activity_value(self, allele: str) -> float:
        """Get activity value for a star allele."""
        # First check loaded allele_function
        base_allele = allele.split(".")[0]
        if base_allele in self.allele_function:
            return self.allele_function[base_allele][1]

        # Then check parsed star_alleles
        info = self.get_star_allele_info(allele)
        if info:
            return info.activity_value

        # Default to normal function for unknown alleles
        return 1.0

    def get_allele_frequencies(self, population: str) -> Dict[str, float]:
        """Get allele frequencies for a specific population."""
        return self.population_frequencies.get(population, {})

    def sample_diplotype(self, population: str) -> Tuple[str, str, float]:
        """
        Sample a random diplotype based on population allele frequencies.

        Args:
            population: Population code (EUR, EAS, AFR, AMR, SAS)

        Returns:
            Tuple of (allele1, allele2, combined_activity_score)
        """
        freq_dict = self.get_allele_frequencies(population)
        if not freq_dict:
            return ("*1", "*1", 2.0)

        alleles = list(freq_dict.keys())
        frequencies = [freq_dict[a] for a in alleles]

        # Normalize
        total = sum(frequencies)
        if total == 0:
            return ("*1", "*1", 2.0)
        frequencies = [f / total for f in frequencies]

        # Sample two alleles
        allele1, allele2 = random.choices(alleles, weights=frequencies, k=2)

        # Calculate combined activity
        activity1 = self.get_activity_value(allele1)
        activity2 = self.get_activity_value(allele2)
        combined = activity1 + activity2

        return (allele1, allele2, combined)

    def activity_to_phenotype(self, activity_score: float) -> str:
        """
        Convert activity score to metabolizer phenotype.

        Based on CPIC 2024 Guidelines:
        - PM (Poor Metabolizer): AS = 0
        - IM (Intermediate Metabolizer): AS = 0.25 to 0.5
        - NM (Normal Metabolizer): AS = 1.0 to 2.0
        - UM (Ultra-rapid Metabolizer): AS > 2.0

        Note: AS = 0.75 is sometimes classified as IM or NM depending on drug.
        We classify 0.75 as IM to be conservative.
        """
        if activity_score == 0.0:
            return "PM"  # Poor Metabolizer
        elif activity_score <= 0.75:
            return "IM"  # Intermediate Metabolizer (0.25, 0.5, 0.75)
        elif activity_score <= 2.0:
            return "NM"  # Normal Metabolizer (1.0, 1.25, 1.5, 1.75, 2.0)
        else:
            return "UM"  # Ultra-rapid Metabolizer (> 2.0)

    def activity_to_clearance_multiplier(self, activity_score: float) -> float:
        """
        Convert activity score to CYP2D6-mediated clearance multiplier.

        Based on CPIC phenotype-to-clearance relationships:
        - PM: ~20% of normal CYP2D6 clearance
        - IM: ~50% of normal CYP2D6 clearance
        - NM: 100% (baseline)
        - UM: Linear increase above 2.0
        """
        if activity_score == 0.0:
            return 0.2  # PM: minimal CYP2D6 activity
        elif activity_score <= 0.75:
            return 0.3 + (activity_score / 0.75) * 0.3  # IM: 0.3-0.6
        elif activity_score <= 2.0:
            return 0.6 + ((activity_score - 0.75) / 1.25) * 0.4  # NM: 0.6-1.0
        else:
            return 1.0 + (activity_score - 2.0) * 0.3  # UM: linear increase

    def simulate_population(
        self, population: str, n_subjects: int = 1000
    ) -> pd.DataFrame:
        """
        Simulate a population with CYP2D6 diplotypes.

        Args:
            population: Population code
            n_subjects: Number of subjects

        Returns:
            DataFrame with simulated subjects
        """
        subjects = []

        for i in range(n_subjects):
            allele1, allele2, activity = self.sample_diplotype(population)
            phenotype = self.activity_to_phenotype(activity)
            cl_mult = self.activity_to_clearance_multiplier(activity)

            subjects.append(
                {
                    "subject_id": i + 1,
                    "population": population,
                    "allele_1": allele1,
                    "allele_2": allele2,
                    "diplotype": f"{allele1}/{allele2}",
                    "activity_score": activity,
                    "phenotype": phenotype,
                    "clearance_multiplier": cl_mult,
                }
            )

        return pd.DataFrame(subjects)

    def get_phenotype_distribution(
        self, population: str, n_samples: int = 10000
    ) -> Dict[str, float]:
        """
        Calculate phenotype distribution for a population via simulation.

        Args:
            population: Population code
            n_samples: Number of samples for estimation

        Returns:
            Dictionary of phenotype → frequency
        """
        phenotype_counts = {"PM": 0, "IM": 0, "NM": 0, "UM": 0}

        for _ in range(n_samples):
            _, _, activity = self.sample_diplotype(population)
            phenotype = self.activity_to_phenotype(activity)
            phenotype_counts[phenotype] += 1

        return {k: v / n_samples for k, v in phenotype_counts.items()}

    def get_defining_variants(self, allele: str) -> List[str]:
        """Get list of defining variants (rsIDs) for a star allele."""
        info = self.get_star_allele_info(allele)
        return info.defining_variants if info else []

    def print_summary(self) -> None:
        """Print summary of loaded data."""
        print("\n" + "=" * 60)
        print("PharmVar CYP2D6 Data Summary")
        print("=" * 60)
        print(f"Genome Build: {self.genome_build}")
        print(
            f"Allele Function Data: {len(self.allele_function)} alleles (from PharmGKB)"
        )
        print(f"Population Frequencies: {len(self.population_frequencies)} populations")
        print(f"Star Alleles (PharmVar): {len(self.star_alleles)}")
        print(f"Unique Variants: {len(self.variants)}")

        # Function distribution from loaded data
        func_dist = {}
        for allele, (func, _) in self.allele_function.items():
            func_dist[func] = func_dist.get(func, 0) + 1

        print("\nAllele Function Distribution (PharmGKB):")
        for func, count in sorted(func_dist.items()):
            print(f"  {func}: {count}")

        # Top alleles per population
        print("\nTop Alleles by Population:")
        for pop in self.POPULATIONS:
            freqs = self.get_allele_frequencies(pop)
            if freqs:
                top = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:5]
                top_str = ", ".join([f"{a}:{f:.2f}" for a, f in top])
                print(f"  {pop}: {top_str}")

        # Population phenotype estimates
        print("\nEstimated Phenotype Distribution by Population:")
        for pop in self.POPULATIONS:
            dist = self.get_phenotype_distribution(pop, n_samples=5000)
            print(f"\n  {self.POPULATION_NAMES[pop]} ({pop}):")
            for pheno in ["PM", "IM", "NM", "UM"]:
                print(f"    {pheno}: {dist[pheno] * 100:.1f}%")


def main():
    """Main function to demonstrate the parser."""
    # Initialize parser
    parser = PharmVarCYP2D6Parser()

    # Print summary
    parser.print_summary()

    # Example: Get star allele info
    print("\n" + "=" * 60)
    print("Star Allele Examples")
    print("=" * 60)
    for allele in ["*1", "*2", "*4", "*10", "*17"]:
        info = parser.get_star_allele_info(allele)
        if info:
            print(f"{allele}: {info.function} (activity={info.activity_value})")
            variants = info.defining_variants[:3] if info.defining_variants else []
            print(f"  Defining variants: {variants}...")

    # Simulate population
    print("\n" + "=" * 60)
    print("Simulated EUR Population (n=10)")
    print("=" * 60)
    sim_df = parser.simulate_population("EUR", n_subjects=10)
    print(sim_df.to_string(index=False))


if __name__ == "__main__":
    main()
