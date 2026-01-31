"""
PubChem API Client
==================
PubChem REST API (PUG REST)를 사용하여 약물 정보를 조회합니다.

Features:
- 약물명으로 SMILES, CID, 분자량 조회
- CID로 상세 정보 조회
- 배치 조회 지원

References:
- https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest

Author: AI-Driven Clinical Trial Platform
Date: 2026-01-26
"""

import requests
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import warnings

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PubChemCompound:
    """PubChem 화합물 정보"""

    cid: int
    name: str
    smiles: str
    molecular_weight: float
    molecular_formula: str
    iupac_name: str
    xlogp: Optional[float] = None
    tpsa: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "cid": self.cid,
            "name": self.name,
            "smiles": self.smiles,
            "molecular_weight": self.molecular_weight,
            "molecular_formula": self.molecular_formula,
            "iupac_name": self.iupac_name,
            "xlogp": self.xlogp,
            "tpsa": self.tpsa,
        }


# =============================================================================
# PubChem Client
# =============================================================================


class PubChemClient:
    """PubChem PUG REST API 클라이언트"""

    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()

    def _request(self, url: str) -> Optional[Dict]:
        """API 요청 공통 처리 (Retry 로직 포함)"""
        for i in range(self.MAX_RETRIES):
            try:
                response = self.session.get(url, timeout=self.timeout)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    # Not found is not an error
                    return None
                elif response.status_code == 503:
                    # Server busy
                    time.sleep(self.RETRY_DELAY * (i + 1))
                    continue
                else:
                    response.raise_for_status()

            except Exception as e:
                if i == self.MAX_RETRIES - 1:
                    print(f"Error fetching from PubChem: {e}")
                    return None
                time.sleep(self.RETRY_DELAY)

        return None

    def get_smiles(self, drug_name: str) -> Optional[str]:
        """
        약물명으로 SMILES 조회

        Args:
            drug_name: 약물 이름 (예: "Aspirin")

        Returns:
            SMILES string or None
        """
        compound = self.get_compound_by_name(drug_name)
        return compound.smiles if compound else None

    def get_compound_by_name(self, drug_name: str) -> Optional[PubChemCompound]:
        """
        약물명으로 상세 정보 조회

        Properties:
        - CanonicalSMILES
        - IsomericSMILES
        - MolecularWeight
        - MolecularFormula
        - IUPACName
        - XLogP
        - TPSA
        """
        props = "CanonicalSMILES,IsomericSMILES,MolecularWeight,MolecularFormula,IUPACName,XLogP,TPSA"
        url = f"{self.BASE_URL}/compound/name/{drug_name}/property/{props}/JSON"

        data = self._request(url)
        if not data:
            return None

        try:
            # 첫 번째 결과 사용
            info = data["PropertyTable"]["Properties"][0]

            smiles = info.get("CanonicalSMILES", "")
            if not smiles:
                smiles = info.get("IsomericSMILES", "")
            if not smiles:
                smiles = info.get("SMILES", "")

            return PubChemCompound(
                cid=info.get("CID", 0),
                name=drug_name,
                smiles=smiles,
                molecular_weight=float(info.get("MolecularWeight", 0)),
                molecular_formula=info.get("MolecularFormula", ""),
                iupac_name=info.get("IUPACName", ""),
                xlogp=float(info.get("XLogP", 0)) if "XLogP" in info else None,
                tpsa=float(info.get("TPSA", 0)) if "TPSA" in info else None,
            )
        except Exception as e:
            print(f"Error parsing PubChem response: {e}")
            return None

    def get_compound_by_cid(self, cid: int) -> Optional[PubChemCompound]:
        """CID로 상세 정보 조회"""
        props = "CanonicalSMILES,MolecularWeight,MolecularFormula,IUPACName,XLogP,TPSA"
        url = f"{self.BASE_URL}/compound/cid/{cid}/property/{props}/JSON"

        data = self._request(url)
        if not data:
            return None

        try:
            info = data["PropertyTable"]["Properties"][0]

            return PubChemCompound(
                cid=cid,
                name=f"CID:{cid}",  # Name is not returned by property query
                smiles=info.get("CanonicalSMILES", ""),
                molecular_weight=float(info.get("MolecularWeight", 0)),
                molecular_formula=info.get("MolecularFormula", ""),
                iupac_name=info.get("IUPACName", ""),
                xlogp=float(info.get("XLogP", 0)) if "XLogP" in info else None,
                tpsa=float(info.get("TPSA", 0)) if "TPSA" in info else None,
            )
        except Exception:
            return None

    def batch_get_smiles(self, drug_names: List[str]) -> Dict[str, Optional[str]]:
        """
        여러 약물 SMILES 일괄 조회 (순차 처리)

        Note: PUG REST API 제한을 고려하여 딜레이 추가
        """
        results = {}
        for name in drug_names:
            results[name] = self.get_smiles(name)
            time.sleep(0.2)  # Rate limit prevention (5 requests/sec limit)
        return results


# =============================================================================
# Convenience Functions
# =============================================================================


def get_smiles(drug_name: str) -> Optional[str]:
    """SMILES 조회 편의 함수"""
    client = PubChemClient()
    return client.get_smiles(drug_name)


def get_compound_info(drug_name: str) -> Optional[Dict]:
    """화합물 정보 조회 편의 함수"""
    client = PubChemClient()
    compound = client.get_compound_by_name(drug_name)
    return compound.to_dict() if compound else None


# =============================================================================
# Main (Test)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PubChem Client Test")
    print("=" * 60)

    test_drugs = ["Aspirin", "Acetaminophen", "Ibuprofen", "NonExistentDrug123"]

    client = PubChemClient()

    for drug in test_drugs:
        print(f"\nFetching: {drug}...")
        compound = client.get_compound_by_name(drug)

        if compound:
            print(f"  CID: {compound.cid}")
            print(f"  SMILES: {compound.smiles}")
            print(f"  MW: {compound.molecular_weight}")
            print(f"  Formula: {compound.molecular_formula}")
        else:
            print("  Not found.")

    print("\n" + "=" * 60)
    print("  Test Complete!")
    print("=" * 60)
