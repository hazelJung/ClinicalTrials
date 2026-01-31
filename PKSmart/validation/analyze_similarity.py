
import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
    # 시각화 설정
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["figure.figsize"] = (10, 6)
    try:
        plt.rcParams['font.family'] = 'AppleGothic'
    except:
        pass
except ImportError:
    print("Warning: Matplotlib or Seaborn not found. Plotting will be skipped.")
    PLOT_AVAILABLE = False

# 경로 설정
PROJECT_ROOT = ".." # validation 폴더에서 실행한다고 가정
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Human_PK_data.csv")
EXTERNAL_DATA_PATH = os.path.join(PROJECT_ROOT, "External_Test_ALL_Lombardo_FDA_CL_fu_V2", "All_external_test_Sets.csv")

# 데이터 로드
if not os.path.exists(TRAIN_DATA_PATH):
    # 스크립트가 프로젝트 루트에서 실행될 경우를 대비
    PROJECT_ROOT = "."
    TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "PKSmart", "data", "Human_PK_data.csv")
    EXTERNAL_DATA_PATH = os.path.join(PROJECT_ROOT, "PKSmart", "External_Test_ALL_Lombardo_FDA_CL_fu_V2", "All_external_test_Sets.csv")

print(f"Loading train data from: {TRAIN_DATA_PATH}")
print(f"Loading external data from: {EXTERNAL_DATA_PATH}")

df_train = pd.read_csv(TRAIN_DATA_PATH)
df_ext = pd.read_csv(EXTERNAL_DATA_PATH)

print(f"학습 데이터 크기: {len(df_train)}")
print(f"외부 데이터 크기: {len(df_ext)}")

# SMILES 컬럼 찾기
train_smiles_col = 'Structure' if 'Structure' in df_train.columns else 'SMILES'
if 'smiles_r' in df_train.columns: train_smiles_col = 'smiles_r'

ext_smiles_col = 'smiles_r'

print(f"Train SMILES col: {train_smiles_col}")
print(f"External SMILES col: {ext_smiles_col}")

def get_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return mol
        return None
    except:
        return None

def get_fp(mol):
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

# Fingerprint 미리 계산
print("Computing Train Fingerprints...")
train_mols = [get_mol(s) for s in df_train[train_smiles_col]]
train_fps = [get_fp(m) for m in train_mols]
train_fps = [fp for fp in train_fps if fp is not None] # 유효한 것만 유지

print("Computing External Fingerprints...")
ext_mols = [get_mol(s) for s in df_ext[ext_smiles_col]]
ext_fps = [get_fp(m) for m in ext_mols]

# 유사도 계산
max_similarities = []

print("Calculating Similarities...")
for i, ext_fp in enumerate(tqdm(ext_fps)):
    if ext_fp is None:
        max_similarities.append(0.0)
        continue
    
    # Bulk similarity calculation against all train FPs
    sims = DataStructs.BulkTanimotoSimilarity(ext_fp, train_fps)
    max_similarities.append(max(sims))

df_ext['max_similarity'] = max_similarities

# 분포 시각화 및 저장
if PLOT_AVAILABLE:
    try:
        plt.figure(figsize=(10, 5))
        sns.histplot(df_ext['max_similarity'], bins=50, kde=True)
        plt.axvline(0.4, color='r', linestyle='--', label='Threshold 0.4')
        plt.axvline(0.5, color='orange', linestyle='--', label='Threshold 0.5')
        plt.axvline(0.6, color='g', linestyle='--', label='Threshold 0.6')
        plt.title('Distribution of Max Tanimoto Similarity (External vs Train)')
        plt.xlabel('Max Tanimoto Similarity')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig('similarity_distribution.png')
        print("Saved plot to similarity_distribution.png")
    except Exception as e:
        print(f"Plotting failed: {e}")
else:
    print("Skipping plot generation.")

# 데이터 통계
print(df_ext['max_similarity'].describe())

# 필터링 및 저장
THRESHOLD = 0.5
df_filtered = df_ext[df_ext['max_similarity'] >= THRESHOLD].reset_index(drop=True)

print(f"Filtering with threshold {THRESHOLD}...")
print(f"Original: {len(df_ext)} -> Filtered: {len(df_filtered)}")
print(f"Retention Rate: {len(df_filtered)/len(df_ext)*100:.1f}%")

SAVE_PATH = "external_test_filtered.csv"
df_filtered.to_csv(SAVE_PATH, index=False)
print(f"Filtered dataset saved to: {SAVE_PATH}")
