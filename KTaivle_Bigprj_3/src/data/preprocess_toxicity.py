"""
Toxicity Data Preprocessor
==========================
Phase 0: 3개 독성 데이터셋 전처리 파이프라인

데이터셋:
- Tox21: 12개 독성 엔드포인트 (학습용)
- ClinTox: FDA 승인/거부 기록 (외부 검증용)
- DILIrank: 간독성 등급 (특화 검증용)

기능:
- SMILES 유효성 검증 (RDKit)
- 41개 분자 기술자 계산
- 데이터셋 통합 및 저장

Author: AI-Driven Clinical Trial Platform
Date: 2026-01-26
"""

import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from src.data.fetch_pubchem import PubChemClient

    PUBCHEM_AVAILABLE = True
except ImportError:
    PUBCHEM_AVAILABLE = False
    print(
        "Warning: could not import PubChemClient. DILIrank processing may fail if SMILES missing."
    )

import pandas as pd

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Fragments
    from rdkit import RDLogger

    RDLogger.logger().setLevel(RDLogger.ERROR)  # Suppress RDKit warnings
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Install with: pip install rdkit-pypi")


# =============================================================================
# 41개 분자 기술자 정의 (QSAR_INTEGRATION_PLAN.md 기준)
# =============================================================================

if RDKIT_AVAILABLE:
    DESCRIPTOR_FUNCTIONS = {
        # 물리화학적 특성 (12개)
        "MolWt": Descriptors.MolWt,
        "MolLogP": Descriptors.MolLogP,
        "TPSA": Descriptors.TPSA,
        "MolMR": Descriptors.MolMR,
        "LabuteASA": Descriptors.LabuteASA,
        "NumValenceElectrons": Descriptors.NumValenceElectrons,
        "MaxPartialCharge": Descriptors.MaxPartialCharge,
        "MinPartialCharge": Descriptors.MinPartialCharge,
        "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge,
        "MinAbsPartialCharge": Descriptors.MinAbsPartialCharge,
        "FpDensityMorgan1": Descriptors.FpDensityMorgan1,
        "FpDensityMorgan2": Descriptors.FpDensityMorgan2,
        # 구조적 특성 (15개)
        "HeavyAtomCount": Descriptors.HeavyAtomCount,
        "NumHDonors": Lipinski.NumHDonors,
        "NumHAcceptors": Lipinski.NumHAcceptors,
        "NumRotatableBonds": Descriptors.NumRotatableBonds,
        "NumHeteroatoms": Descriptors.NumHeteroatoms,
        "RingCount": Descriptors.RingCount,
        "NumAromaticRings": Descriptors.NumAromaticRings,
        "NumAliphaticRings": Descriptors.NumAliphaticRings,
        "NumSaturatedRings": Descriptors.NumSaturatedRings,
        "FractionCSP3": Descriptors.FractionCSP3,
        "NumAromaticHeterocycles": Descriptors.NumAromaticHeterocycles,
        "NumAromaticCarbocycles": Descriptors.NumAromaticCarbocycles,
        "NumAliphaticHeterocycles": Descriptors.NumAliphaticHeterocycles,
        "NumAliphaticCarbocycles": Descriptors.NumAliphaticCarbocycles,
        "NumSaturatedHeterocycles": Descriptors.NumSaturatedHeterocycles,
        # 독성 관련 (14개)
        "NOCount": Lipinski.NOCount,
        "NHOHCount": Lipinski.NHOHCount,
        "NumRadicalElectrons": Descriptors.NumRadicalElectrons,
        "fr_Al_OH": Fragments.fr_Al_OH,  # 알코올
        "fr_Ar_OH": Fragments.fr_Ar_OH,  # 페놀
        "fr_aldehyde": Fragments.fr_aldehyde,  # 알데히드
        "fr_ketone": Fragments.fr_ketone,  # 케톤
        "fr_ether": Fragments.fr_ether,  # 에테르
        "fr_ester": Fragments.fr_ester,  # 에스테르
        "fr_nitro": Fragments.fr_nitro,  # 니트로
        "fr_nitrile": Fragments.fr_nitrile,  # 니트릴
        "fr_halogen": Fragments.fr_halogen,  # 할로겐
        "fr_sulfide": Fragments.fr_sulfide,  # 황화물
        "fr_amide": Fragments.fr_amide,  # 아미드
    }
else:
    DESCRIPTOR_FUNCTIONS = {}


@dataclass
class PreprocessingResult:
    """전처리 결과"""

    total_compounds: int
    valid_compounds: int
    invalid_smiles: int
    descriptors_computed: int
    dataset_name: str


class ToxicityDataPreprocessor:
    """독성 데이터 전처리기"""

    # Tox21 엔드포인트 목록
    TOX21_ENDPOINTS = [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ]

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize preprocessor.

        Args:
            data_dir: Path to raw data directory
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / "raw"

        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir.parent / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        if not RDKIT_AVAILABLE:
            raise ImportError(
                "RDKit is required for SMILES validation and descriptor calculation."
            )

    def validate_smiles(self, smiles: str) -> Optional[Chem.Mol]:
        """
        Validate SMILES string and return RDKit Mol object.

        Args:
            smiles: SMILES string

        Returns:
            RDKit Mol object if valid, None otherwise
        """
        if pd.isna(smiles) or not isinstance(smiles, str):
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            # Additional validation: check atom count
            if mol.GetNumAtoms() < 2:
                return None
            return mol
        except Exception:
            return None

    def calculate_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate 41 molecular descriptors for a molecule.

        Args:
            mol: RDKit Mol object

        Returns:
            Dictionary of descriptor name -> value
        """
        descriptors = {}

        for name, func in DESCRIPTOR_FUNCTIONS.items():
            try:
                value = func(mol)
                # Handle NaN and Inf values
                if value is None or np.isnan(value) or np.isinf(value):
                    value = 0.0
                descriptors[name] = float(value)
            except Exception:
                descriptors[name] = 0.0

        return descriptors

    def preprocess_tox21(self) -> Tuple[pd.DataFrame, PreprocessingResult]:
        """
        Preprocess Tox21 dataset.

        Returns:
            Processed DataFrame, PreprocessingResult
        """
        print("\n>>> Processing Tox21 dataset...")

        tox21_path = self.data_dir / "tox21.csv"
        if not tox21_path.exists():
            raise FileNotFoundError(f"Tox21 file not found: {tox21_path}")

        df = pd.read_csv(tox21_path)
        total = len(df)
        print(f"    Loaded {total} compounds from Tox21")

        # Identify SMILES column (lowercase)
        smiles_col = "smiles" if "smiles" in df.columns else "SMILES"

        # Validate SMILES and calculate descriptors
        valid_rows = []
        invalid_count = 0

        for idx, row in df.iterrows():
            smiles = row[smiles_col]
            mol = self.validate_smiles(smiles)

            if mol is None:
                invalid_count += 1
                continue

            # Calculate descriptors
            descriptors = self.calculate_descriptors(mol)

            # Build row data
            row_data = {
                "smiles": smiles,
                "mol_id": row.get("mol_id", f"TOX_{idx}"),
                **descriptors,
            }

            # Add toxicity labels
            for endpoint in self.TOX21_ENDPOINTS:
                if endpoint in df.columns:
                    value = row[endpoint]
                    # Convert to binary (handle missing values)
                    if pd.isna(value):
                        row_data[endpoint] = np.nan
                    else:
                        row_data[endpoint] = int(value)

            valid_rows.append(row_data)

            if (idx + 1) % 1000 == 0:
                print(f"    Processed {idx + 1}/{total} compounds...")

        result_df = pd.DataFrame(valid_rows)

        result = PreprocessingResult(
            total_compounds=total,
            valid_compounds=len(valid_rows),
            invalid_smiles=invalid_count,
            descriptors_computed=len(DESCRIPTOR_FUNCTIONS),
            dataset_name="tox21",
        )

        print(
            f"    Valid: {result.valid_compounds}, Invalid SMILES: {result.invalid_smiles}"
        )

        return result_df, result

    def preprocess_clintox(self) -> Tuple[pd.DataFrame, PreprocessingResult]:
        """
        Preprocess ClinTox dataset.

        Returns:
            Processed DataFrame, PreprocessingResult
        """
        print("\n>>> Processing ClinTox dataset...")

        clintox_path = self.data_dir / "clintox.csv"
        if not clintox_path.exists():
            raise FileNotFoundError(f"ClinTox file not found: {clintox_path}")

        df = pd.read_csv(clintox_path)
        total = len(df)
        print(f"    Loaded {total} compounds from ClinTox")

        # Identify columns
        smiles_col = "smiles" if "smiles" in df.columns else "SMILES"

        valid_rows = []
        invalid_count = 0

        for idx, row in df.iterrows():
            smiles = row[smiles_col]
            mol = self.validate_smiles(smiles)

            if mol is None:
                invalid_count += 1
                continue

            descriptors = self.calculate_descriptors(mol)

            row_data = {
                "smiles": smiles,
                "compound_id": f"CLINTOX_{idx}",
                **descriptors,
                "FDA_APPROVED": int(row.get("FDA_APPROVED", 0)),
                "CT_TOX": int(row.get("CT_TOX", 0)),
            }

            valid_rows.append(row_data)

            if (idx + 1) % 500 == 0:
                print(f"    Processed {idx + 1}/{total} compounds...")

        result_df = pd.DataFrame(valid_rows)

        result = PreprocessingResult(
            total_compounds=total,
            valid_compounds=len(valid_rows),
            invalid_smiles=invalid_count,
            descriptors_computed=len(DESCRIPTOR_FUNCTIONS),
            dataset_name="clintox",
        )

        print(
            f"    Valid: {result.valid_compounds}, Invalid SMILES: {result.invalid_smiles}"
        )

        return result_df, result

    def preprocess_dilirank(self) -> Tuple[pd.DataFrame, PreprocessingResult]:
        """
        Preprocess DILIrank dataset.

        Returns:
            Processed DataFrame, PreprocessingResult
        """
        print("\n>>> Processing DILIrank dataset...")

        # Try to find the DILIrank file (has long name)
        dili_files = list(self.data_dir.glob("*DILI*"))
        if not dili_files:
            dili_files = list(self.data_dir.glob("*Liver Injury*"))

        if not dili_files:
            print("    Warning: DILIrank file not found. Skipping...")
            return pd.DataFrame(), PreprocessingResult(0, 0, 0, 0, "dilirank")

        dili_path = dili_files[0]
        print(f"    Found: {dili_path.name}")

        try:
            # Header is on the second row (index 1)
            df = pd.read_excel(dili_path, header=1)
        except Exception as e:
            print(f"    Error reading Excel file: {e}")
            return pd.DataFrame(), PreprocessingResult(0, 0, 0, 0, "dilirank")

        total = len(df)
        print(f"    Loaded {total} compounds from DILIrank")

        # Initialize PubChem Client
        pubchem_client = PubChemClient() if PUBCHEM_AVAILABLE else None

        # DILI severity mapping
        DILI_MAPPING = {
            "Most-DILI-Concern": 3,
            "Less-DILI-Concern": 2,
            "Ambiguous DILI-concern": 1,
            "No-DILI-Concern": 0,
        }

        valid_rows = []
        invalid_count = 0

        # Create a cache for SMILES to avoid redundant API calls
        smiles_cache = {}

        print("    Fetching SMILES from PubChem (this may take a while)...")
        print("    NOTE: Limiting to first 50 compounds for rapid prototyping.")

        for idx, row in df.iterrows():
            if idx >= 50:
                print("    --- Limit reached (50 compounds) ---")
                break

            drug_name = row.get("CompoundName", "")

            if pd.isna(drug_name) or str(drug_name).strip() == "":
                invalid_count += 1
                continue

            drug_name = str(drug_name).strip()
            print(f"DEBUG: Processing {drug_name}")

            # 1. Try to get SMILES

            smiles = None
            if drug_name in smiles_cache:
                smiles = smiles_cache[drug_name]
            elif pubchem_client:
                # Fetch from PubChem
                try:
                    smiles = pubchem_client.get_smiles(drug_name)
                    if smiles:
                        smiles_cache[drug_name] = smiles
                except Exception:
                    pass

            if not smiles:
                # print(f"      Could not find SMILES for: {drug_name}")
                invalid_count += 1
                continue

            mol = self.validate_smiles(smiles)

            if mol is None:
                invalid_count += 1
                continue

            descriptors = self.calculate_descriptors(mol)

            # Get DILI rank
            dili_rank = row.get("vDILI-Concern", "Unknown")
            dili_severity = DILI_MAPPING.get(str(dili_rank).strip(), -1)

            row_data = {
                "smiles": smiles,
                "drug_name": drug_name,
                **descriptors,
                "dili_rank": str(dili_rank),
                "dili_severity": dili_severity,
                "is_dili_concern": 1 if dili_severity >= 2 else 0,
            }

            valid_rows.append(row_data)

            if (idx + 1) % 50 == 0:
                print(f"    Processed {idx + 1}/{total} compounds...")

        result_df = pd.DataFrame(valid_rows)

        result = PreprocessingResult(
            total_compounds=total,
            valid_compounds=len(valid_rows),
            invalid_smiles=invalid_count,
            descriptors_computed=len(DESCRIPTOR_FUNCTIONS),
            dataset_name="dilirank",
        )

        print(
            f"    Valid: {result.valid_compounds}, Invalid/No-SMILES: {result.invalid_smiles}"
        )

        return result_df, result

    def run_full_preprocessing(self) -> Dict[str, pd.DataFrame]:
        """
        Run full preprocessing pipeline on all datasets.

        Returns:
            Dictionary of dataset name -> processed DataFrame
        """
        print("=" * 60)
        print("  Toxicity Data Preprocessing Pipeline")
        print("=" * 60)

        results = {}

        # Process Tox21 (training data)
        tox21_out = self.processed_dir / "tox21_descriptors.csv"
        if tox21_out.exists():
            print(f"\n>>> Tox21 output exists ({tox21_out.name}). Skipping...")
            results["tox21"] = pd.read_csv(tox21_out)
        else:
            tox21_df, tox21_result = self.preprocess_tox21()
            if not tox21_df.empty:
                tox21_df.to_csv(tox21_out, index=False)
                results["tox21"] = tox21_df
                print(f"    Saved: tox21_descriptors.csv ({len(tox21_df)} rows)")

        # Process ClinTox (external validation)
        clintox_out = self.processed_dir / "clintox_descriptors.csv"
        if clintox_out.exists():
            print(f"\n>>> ClinTox output exists ({clintox_out.name}). Skipping...")
            results["clintox"] = pd.read_csv(clintox_out)
        else:
            clintox_df, clintox_result = self.preprocess_clintox()
            if not clintox_df.empty:
                clintox_df.to_csv(clintox_out, index=False)
                results["clintox"] = clintox_df
                print(f"    Saved: clintox_descriptors.csv ({len(clintox_df)} rows)")

        # Process DILIrank (liver toxicity validation)
        dili_out = self.processed_dir / "dilirank_descriptors.csv"
        if dili_out.exists():
            print(f"\n>>> DILIrank output exists ({dili_out.name}). Skipping...")
            results["dilirank"] = pd.read_csv(dili_out)
        else:
            dili_df, dili_result = self.preprocess_dilirank()
            if not dili_df.empty:
                dili_df.to_csv(dili_out, index=False)
                results["dilirank"] = dili_df
                print(f"    Saved: dilirank_descriptors.csv ({len(dili_df)} rows)")

        # Summary
        print("\n" + "=" * 60)
        print("  Preprocessing Summary")
        print("=" * 60)
        for name, df in results.items():
            print(f"  {name}: {len(df)} valid compounds")
        print(f"\n  Descriptors per compound: {len(DESCRIPTOR_FUNCTIONS)}")
        print(f"  Output directory: {self.processed_dir}")

        return results


def get_descriptor_names() -> List[str]:
    """Get list of all descriptor names."""
    return list(DESCRIPTOR_FUNCTIONS.keys())


def calculate_descriptors_for_smiles(smiles: str) -> Optional[Dict[str, float]]:
    """
    Calculate descriptors for a single SMILES string.

    Args:
        smiles: SMILES string

    Returns:
        Dictionary of descriptors or None if invalid SMILES
    """
    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    preprocessor = ToxicityDataPreprocessor.__new__(ToxicityDataPreprocessor)
    return preprocessor.calculate_descriptors(mol)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    preprocessor = ToxicityDataPreprocessor()
    datasets = preprocessor.run_full_preprocessing()

    print("\n>>> Preprocessing Complete!")
