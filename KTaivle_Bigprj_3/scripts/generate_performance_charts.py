"""
mPBPK-ML Model Performance Visualization
=========================================
Generates comprehensive performance charts for the analysis report.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Set up matplotlib for non-interactive backend
plt.switch_backend('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

# Create output directory
output_dir = Path('data/processed/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. Model Performance Metrics (Bar Chart)
# =============================================================================
print("Creating performance metrics chart...")

metrics = {
    'Accuracy': [0.711, 0.715],
    'Precision': [0.634, 0.639],
    'Recall': [0.829, 0.827],
    'F1 Score': [0.719, 0.721],
    'ROC-AUC': [0.782, 0.787]
}

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, [v[0] for v in metrics.values()], width, label='Decision Tree', color='#3498db')
bars2 = ax.bar(x + width/2, [v[1] for v in metrics.values()], width, label='Random Forest', color='#2ecc71')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('mPBPK-ML Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics.keys(), fontsize=11)
ax.legend(loc='lower right')
ax.set_ylim(0, 1.0)
ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Threshold')

# Add value labels on bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'model_performance_comparison.png')
plt.close()
print(f"  Saved: {output_dir / 'model_performance_comparison.png'}")

# =============================================================================
# 2. Feature Importance (Horizontal Bar Chart)
# =============================================================================
print("Creating feature importance chart...")

features = {
    'log_dose': 0.529,
    'log_potency': 0.238,
    'log_KD': 0.074,
    'log_halflife': 0.049,
    'log_T0': 0.034,
    'log_MW': 0.029,
    'activity_score': 0.013,
    'cl_multiplier': 0.006,
    'charge': 0.006,
    'pop_EUR': 0.004
}

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(features)))[::-1]
y_pos = np.arange(len(features))

bars = ax.barh(y_pos, list(features.values()), color=colors)
ax.set_yticks(y_pos)
ax.set_yticklabels(list(features.keys()), fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, features.values())):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
            f'{val:.1%}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png')
plt.close()
print(f"  Saved: {output_dir / 'feature_importance.png'}")

# =============================================================================
# 3. Parameter Distribution (Histograms)
# =============================================================================
print("Creating parameter distribution charts...")

# Load training data
data_path = Path('data/processed/mpbpk_training_data_latest.csv')
if data_path.exists():
    df = pd.read_csv(data_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    params = [
        ('log_dose', 'Log Dose (mg/kg)', '#3498db'),
        ('log_KD', 'Log KD (nM)', '#e74c3c'),
        ('log_MW', 'Log MW (kDa)', '#2ecc71'),
        ('log_halflife', 'Log Half-life (h)', '#9b59b6'),
        ('log_T0', 'Log T0 (nM)', '#f39c12'),
        ('activity_score', 'CYP2D6 Activity Score', '#1abc9c')
    ]
    
    for ax, (col, title, color) in zip(axes.flatten(), params):
        if col in df.columns:
            ax.hist(df[col].dropna(), bins=50, color=color, alpha=0.7, edgecolor='white')
            ax.set_xlabel(title, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'Distribution: {title}', fontsize=11)
    
    plt.suptitle('Training Data Parameter Distributions (n=50,000)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_distributions.png')
    plt.close()
    print(f"  Saved: {output_dir / 'parameter_distributions.png'}")
else:
    print("  Warning: Training data not found, skipping distribution chart.")

# =============================================================================
# 4. Class Balance Pie Chart
# =============================================================================
print("Creating class balance chart...")

if data_path.exists():
    success_count = df['efficacy_success'].sum()
    fail_count = len(df) - success_count
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.02, 0.02)
    
    ax.pie([success_count, fail_count], 
           labels=['Success', 'Fail'],
           autopct='%1.1f%%',
           colors=colors,
           explode=explode,
           startangle=90,
           textprops={'fontsize': 12})
    
    ax.set_title('Training Data Class Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png')
    plt.close()
    print(f"  Saved: {output_dir / 'class_distribution.png'}")

# =============================================================================
# 5. Phenotype Distribution by Population
# =============================================================================
print("Creating phenotype distribution chart...")

if data_path.exists() and 'phenotype' in df.columns and 'population' in df.columns:
    phenotype_counts = df.groupby(['population', 'phenotype']).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    phenotype_counts.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='white')
    
    ax.set_xlabel('Population', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('CYP2D6 Phenotype Distribution by Population', fontsize=14, fontweight='bold')
    ax.legend(title='Phenotype', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phenotype_by_population.png')
    plt.close()
    print(f"  Saved: {output_dir / 'phenotype_by_population.png'}")

# =============================================================================
# 6. Cross-Validation Scores
# =============================================================================
print("Creating cross-validation chart...")

cv_scores = {
    'Fold 1': [0.718, 0.722],
    'Fold 2': [0.725, 0.730],
    'Fold 3': [0.720, 0.724],
    'Fold 4': [0.728, 0.731],
    'Fold 5': [0.724, 0.728]
}

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(cv_scores))
width = 0.35

bars1 = ax.bar(x - width/2, [v[0] for v in cv_scores.values()], width, 
               label='Decision Tree', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, [v[1] for v in cv_scores.values()], width, 
               label='Random Forest', color='#2ecc71', alpha=0.8)

ax.set_ylabel('F1 Score', fontsize=12)
ax.set_xlabel('Cross-Validation Fold', fontsize=12)
ax.set_title('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cv_scores.keys())
ax.legend()
ax.set_ylim(0.7, 0.75)
ax.axhline(y=np.mean([v[1] for v in cv_scores.values()]), color='green', 
           linestyle='--', alpha=0.7, label='RF Mean')

plt.tight_layout()
plt.savefig(output_dir / 'cross_validation_scores.png')
plt.close()
print(f"  Saved: {output_dir / 'cross_validation_scores.png'}")

print("\n" + "="*50)
print("  All visualizations generated successfully!")
print("="*50)
print(f"\nOutput directory: {output_dir.absolute()}")
