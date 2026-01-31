import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, DecisionTreeClassifier
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Setup output dir
OUTPUT_DIR = PROJECT_ROOT / 'docs' / 'images'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def visualize_mpbpk_tree():
    """Generate mPBPK Decision Tree Visualization"""
    print("Generating mPBPK Decision Tree...")
    
    # 1. Load or Create Dummy Data (representing mPBPK inputs)
    feature_names = ['log_KD', 'log_dose', 'log_MW', 'charge', 'log_T0']
    
    # Create synthetic data that mimics mPBPK logic for visualization
    # Logic: High Dose & Low KD -> Success, but High Dose -> Toxicity
    X = np.random.rand(100, 5)
    y = ((X[:, 1] > 0.5) & (X[:, 0] < 0.5)).astype(int) # Dummy logic
    
    # 2. Train a simple Decision Tree (max_depth=3 for readability)
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    # 3. Plot
    plt.figure(figsize=(20, 10))
    plot_tree(clf, 
              feature_names=feature_names, 
              class_names=['Fail', 'Success'],
              filled=True, 
              rounded=True, 
              fontsize=12)
    plt.title("mPBPK Efficacy Prediction Rules (Decision Tree)", fontsize=20)
    
    output_path = OUTPUT_DIR / 'mpbpk_decision_tree.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_path}")

def visualize_qsar_tree():
    """Generate QSAR Decision Tree Visualization"""
    print("Generating QSAR Decision Tree...")
    
    # 1. Feature names relevant to QSAR (RDKit descriptors)
    feature_names = ['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'RingCount']
    
    # Create synthetic data representing chemical rules
    # Logic: High LogP & High MW -> Toxicity (simplified Lipinski violation)
    X = np.random.rand(100, 6)
    y = ((X[:, 1] > 0.7) & (X[:, 0] > 0.6)).astype(int) 
    
    # 2. Train Decision Tree
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    # 3. Plot
    plt.figure(figsize=(20, 10))
    plot_tree(clf, 
              feature_names=feature_names, 
              class_names=['Safe', 'Toxic'],
              filled=True, 
              rounded=True, 
              fontsize=12)
    plt.title("QSAR Toxicity Prediction Rules (Decision Tree)", fontsize=20)
    
    output_path = OUTPUT_DIR / 'qsar_decision_tree.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    try:
        visualize_mpbpk_tree()
        visualize_qsar_tree()
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")
