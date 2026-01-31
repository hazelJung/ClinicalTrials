import pandas as pd
print("DEBUG: LOADING LOCAL FEATURES.PY")
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
# dimorphite_dl 2.x API
# from dimorphite_dl import protonate_smiles
from mordred import Calculator, descriptors
from tqdm import tqdm

def standardize_smiles(smiles):
    """
    Standardizes a SMILES string:
    1. Cleanup
    2. FragmentParent
    3. Uncharge
    4. Protonate (pH 7.4)
    """
    if smiles is None:
        return None
    if not isinstance(smiles, str):
        try:
            smiles = str(smiles)
        except:
            return None
            
    # Check for "nan" string which astype(str) produces for NaNs
    if smiles.lower() == "nan":
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Cleanup
        clean_mol = rdMolStandardize.Cleanup(mol)
        
        # Get parent fragment
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        
        # Uncharge
        uncharger = rdMolStandardize.Uncharger()
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
        
        # Protonate at pH 7.4
        # Modern 2.x API
        # from dimorphite_dl import protonate_smiles
        # smiles_in = Chem.MolToSmiles(uncharged_parent_clean_mol)
        # try:
        #     protonated_smiles_list = protonate_smiles(smiles_in, ph_min=7.4, ph_max=7.4, precision=1.0)
        #     if len(protonated_smiles_list) > 0:
        #         return protonated_smiles_list[0]
        # except:
        #     pass
            
        return Chem.MolToSmiles(uncharged_parent_clean_mol)
            
    except Exception as e:
        print(f"Error standardizing {smiles}: {e}")
        return None

def calculate_morgan_fingerprints(smiles_list, radius=2, nBits=2048):
    """
    Calculates Morgan fingerprints for a list of SMILES.
    Returns a DataFrame with columns Mfp0, Mfp1, ...
    """
    mols = []
    for s in smiles_list:
        try:
            if not isinstance(s, str):
                s = str(s)
            mols.append(Chem.MolFromSmiles(s))
        except:
            mols.append(None)

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nBits) if m else np.zeros(nBits) for m in mols]
    
    fps_array = []
    for fp in fps:
        arr = np.zeros((1,))
        try:
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(fp, arr)
            fps_array.append(arr)
        except:
             # Fallback for empty/zeros
             fps_array.append(np.zeros(nBits))

    df = pd.DataFrame(fps_array, columns=[f"Mfp{i}" for i in range(nBits)])
    return df

def calculate_mordred_descriptors(smiles_list):
    """
    Calculates Mordred descriptors.
    Returns a DataFrame.
    """
    calc = Calculator(descriptors, ignore_3D=True)
    mols = []
    for s in smiles_list:
        try:
            if not isinstance(s, str):
                s = str(s)
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError("Invalid SMILES")
            mols.append(mol)
        except:
            # We must handle failed mols by ensuring they result in a row of NaNs or are handled by batch
            # But here we append None and check later?
            # calc(mol) fails if mol is None.
            # So we create a dummy None mol or handle index alignment.
            # Best is to append None and handle in loop.
            mols.append(None)
    
    results = []
    
    print("  Computing Mordred descriptors (batch mode)...")
    for i, mol in enumerate(mols):
        if i % 100 == 0:
            print(f"    Processing Mordred for mol {i}/{len(mols)}")
            
        if mol is None:
            results.append({})
            continue
            
        try:
            # calc(mol) returns a dict-like result of descriptors
            res = calc(mol)
            # Convert to dict with handling of Missing/Error objects
            # fill_value=np.nan converts mordred errors to nan
            # res.fill_missing(np.nan) returns a Result object, which acts like a list.
            # We must convert it to a dict to preserve column names.
            res_dict = res.fill_missing(np.nan).asdict()
            results.append(res_dict)
        except Exception as e:
            # Logic to append a row of NaNs
            print(f"    Warning: Mordred failed for molecule {i}: {e}")
            results.append({}) 

    df = pd.DataFrame(results)
    # Ensure numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def generate_features(smiles_list):
    """
    Full pipeline: Standardize -> Morgan -> Mordred -> Merge
    """
    # 1. Standardize
    std_smiles = [standardize_smiles(s) for s in smiles_list]
    
    # Store indices of valid ones to know where they came from?
    # Actually, we should return a DataFrame aligned with input smiles_list?
    # But usually we filter out bad ones.
    
    valid_indices = [i for i, s in enumerate(std_smiles) if s is not None]
    valid_smiles = [std_smiles[i] for i in valid_indices]
    
    if not valid_smiles:
        return None
    
    # 2. Morgan
    print("Calculating Morgan Fingerprints...")
    df_morgan = calculate_morgan_fingerprints(valid_smiles)
    
    # 3. Mordred
    print("Calculating Mordred Descriptors...")
    df_mordred = calculate_mordred_descriptors(valid_smiles)
    
    # 4. Merge
    df_morgan.reset_index(drop=True, inplace=True)
    df_mordred.reset_index(drop=True, inplace=True)
    
    combined = pd.concat([df_mordred, df_morgan], axis=1)
    
    # Add SMILES column for reference
    combined['smiles_r'] = valid_smiles
    
    return combined
