"""
Generate Validation Plots
=========================
Generates visualization plots from validation results (CSV).
Alternative to running the Jupyter notebook.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

RESULTS_DIR = 'validation/results'
if not os.path.exists(RESULTS_DIR):
    RESULTS_DIR = 'results' # Try current dir relative path if running from validation/

def generate_pk_plots():
    print("Generating PK plots...")
    results_path = os.path.join(RESULTS_DIR, 'pk_validation_results.csv')
    preds_path = os.path.join(RESULTS_DIR, 'pk_predictions.csv')
    
    if not os.path.exists(results_path):
        print(f"  Skipping: {results_path} not found")
        return
        
    pk_results = pd.read_csv(results_path)
    pk_predictions = pd.read_csv(preds_path)
    
    # 1. Metrics Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # R2
    ax = axes[0]
    colors = ['#2196F3' if r > 0.5 else '#f44336' for r in pk_results['R2']]
    ax.barh(pk_results['Model'], pk_results['R2'], color=colors)
    ax.axvline(x=0.5, color='gray', linestyle='--', label='R²=0.5')
    ax.set_xlabel('R² Score')
    ax.set_title('PK Models: R² Score')
    ax.set_xlim(0, 1)
    
    # RMSE
    ax = axes[1]
    ax.barh(pk_results['Model'], pk_results['RMSE'], color='#FF9800')
    ax.set_xlabel('RMSE')
    ax.set_title('PK Models: RMSE')
    
    # 2-Fold
    ax = axes[2]
    colors = ['#4CAF50' if acc > 50 else '#f44336' for acc in pk_results['2-Fold%']]
    ax.barh(pk_results['Model'], pk_results['2-Fold%'], color=colors)
    ax.axvline(x=50, color='gray', linestyle='--', label='50%')
    ax.set_xlabel('2-Fold Accuracy (%)')
    ax.set_title('PK Models: 2-Fold Accuracy')
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pk_metrics_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Variable Importance (skip - not available in results csv)
    
    # 3. Pred vs Actual
    models = pk_predictions['model'].unique()
    n_models = len(models)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        data = pk_predictions[pk_predictions['model'] == model]
        ax.scatter(data['y_true'], data['y_pred'], alpha=0.5, s=30)
        
        lims = [min(data['y_true'].min(), data['y_pred'].min()),
                max(data['y_true'].max(), data['y_pred'].max())]
        ax.plot(lims, lims, 'r--', linewidth=2)
        
        if data['log_transformed'].iloc[0]:
            fold_2 = np.log10(2)
            ax.plot(lims, [lims[0]+fold_2, lims[1]+fold_2], 'g--', alpha=0.5)
            ax.plot(lims, [lims[0]-fold_2, lims[1]-fold_2], 'g--', alpha=0.5)
            
        ax.set_title(f'{model}')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pk_pred_vs_actual.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  PK plots saved.")

def generate_ld50_plots():
    print("Generating LD50 plots...")
    results_path = os.path.join(RESULTS_DIR, 'ld50_validation_results.csv')
    preds_path = os.path.join(RESULTS_DIR, 'ld50_predictions.csv')
    
    if not os.path.exists(results_path):
        print(f"  Skipping: {results_path} not found")
        return
        
    ld50_results = pd.read_csv(results_path)
    ld50_predictions = pd.read_csv(preds_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter
    ax = axes[0]
    ax.scatter(ld50_predictions['y_true'], ld50_predictions['y_pred'], alpha=0.3, s=20)
    
    lims = [min(ld50_predictions['y_true'].min(), ld50_predictions['y_pred'].min()),
            max(ld50_predictions['y_true'].max(), ld50_predictions['y_pred'].max())]
    ax.plot(lims, lims, 'r--', linewidth=2)
    
    fold_2 = np.log10(2)
    ax.plot(lims, [lims[0]+fold_2, lims[1]+fold_2], 'g--', alpha=0.5)
    ax.plot(lims, [lims[0]-fold_2, lims[1]-fold_2], 'g--', alpha=0.5)
    
    ax.set_title(f"LD50: Predicted vs Actual (R²={ld50_results['R2'].iloc[0]:.3f})")
    ax.set_xlabel('Actual log10(LD50)')
    ax.set_ylabel('Predicted log10(LD50)')
    
    # Residuals
    ax = axes[1]
    residuals = ld50_predictions['y_pred'] - ld50_predictions['y_true']
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='#FF9800')
    ax.axvline(x=0, color='red', linestyle='--')
    ax.set_title('LD50 Residual Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'ld50_validation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  LD50 plots saved.")

def generate_tox21_plots():
    print("Generating Tox21 plots...")
    results_path = os.path.join(RESULTS_DIR, 'tox21_validation_results.csv')
    preds_path = os.path.join(RESULTS_DIR, 'tox21_predictions.csv')
    
    if not os.path.exists(results_path):
        print(f"  Skipping: {results_path} not found")
        return
        
    tox21_results = pd.read_csv(results_path)
    tox21_predictions = pd.read_csv(preds_path)
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 10))
    metrics_df = tox21_results.set_index('Model')[['AUC', 'Accuracy', 'F1', 'Precision', 'Recall']]
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1, ax=ax)
    ax.set_title('Tox21 Models: Performance Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'tox21_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ROC Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, 12))
    
    for idx, model_name in enumerate(tox21_results['Model']):
        data = tox21_predictions[tox21_predictions['model'] == model_name]
        if len(data) < 5: continue
            
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[idx], lw=2, label=f'{model_name} (AUC={roc_auc:.2f})')
        
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_title('Tox21: ROC Curves')
    ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'tox21_roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Tox21 plots saved.")

if __name__ == "__main__":
    generate_pk_plots()
    generate_ld50_plots()
    generate_tox21_plots()
    print("All plots generated.")
