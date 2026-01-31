"""
Visualization of Training Data Distributions
=============================================
Displays drug parameter distributions and patient cohort composition.
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
from pathlib import Path

# Load training data
data_path = Path('data/processed/mpbpk_training_data_latest.csv')
df = pd.read_csv(data_path)

print(f"Loaded {len(df):,} samples from {data_path}")

# Create output directory
output_dir = Path('data/processed/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig_dpi = 150

# =============================================================================
# Figure 1: Drug Parameter Distributions
# =============================================================================
fig1, axes = plt.subplots(2, 3, figsize=(14, 8))
fig1.suptitle('Virtual Drug Candidate Parameter Distributions', fontsize=14, fontweight='bold')

# KD distribution (log scale)
ax = axes[0, 0]
ax.hist(df['log_KD'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax.set_xlabel('log₁₀(KD) [nM]')
ax.set_ylabel('Frequency')
ax.set_title(f'KD Distribution (n={len(df):,})')
ax.axvline(df['log_KD'].median(), color='red', linestyle='--', label=f'Median: {10**df["log_KD"].median():.1f} nM')
ax.legend()

# Dose distribution (log scale)
ax = axes[0, 1]
ax.hist(df['log_dose'], bins=50, color='forestgreen', edgecolor='white', alpha=0.8)
ax.set_xlabel('log₁₀(Dose) [mg/kg]')
ax.set_ylabel('Frequency')
ax.set_title('Dose Distribution')
ax.axvline(df['log_dose'].median(), color='red', linestyle='--', label=f'Median: {10**df["log_dose"].median():.1f} mg/kg')
ax.legend()

# T0 distribution
ax = axes[0, 2]
ax.hist(df['log_T0'], bins=50, color='coral', edgecolor='white', alpha=0.8)
ax.set_xlabel('log₁₀(T0) [nM]')
ax.set_ylabel('Frequency')
ax.set_title('Target Baseline (T0) Distribution')
ax.axvline(df['log_T0'].median(), color='red', linestyle='--', label=f'Median: {10**df["log_T0"].median():.1f} nM')
ax.legend()

# Half-life distribution
ax = axes[1, 0]
ax.hist(df['log_halflife'], bins=50, color='mediumpurple', edgecolor='white', alpha=0.8)
ax.set_xlabel('log₁₀(Half-life) [hr]')
ax.set_ylabel('Frequency')
ax.set_title('Half-life Distribution')
ax.axvline(df['log_halflife'].median(), color='red', linestyle='--', label=f'Median: {10**df["log_halflife"].median():.1f} hr')
ax.legend()

# Charge distribution
ax = axes[1, 1]
charge_counts = df['charge'].value_counts().sort_index()
ax.bar(charge_counts.index.astype(str), charge_counts.values, color=['salmon', 'lightgray', 'skyblue'], edgecolor='black')
ax.set_xlabel('Surface Charge')
ax.set_ylabel('Frequency')
ax.set_title('Charge Distribution')
for i, (ch, cnt) in enumerate(charge_counts.items()):
    ax.text(i, cnt + 500, f'{cnt:,}', ha='center', fontsize=10)

# Efficacy success rate
ax = axes[1, 2]
success_counts = df['efficacy_success'].value_counts()
colors = ['#ff6b6b', '#51cf66']
wedges, texts, autotexts = ax.pie(
    success_counts.values, 
    labels=['Failure', 'Success'],
    colors=colors,
    autopct='%1.1f%%',
    explode=(0.02, 0.02),
    startangle=90
)
ax.set_title(f'Efficacy Success Rate (TO ≥ 90%)')

plt.tight_layout()
fig1.savefig(output_dir / 'drug_parameter_distributions.png', dpi=fig_dpi, bbox_inches='tight')
print(f"Saved: {output_dir / 'drug_parameter_distributions.png'}")

# =============================================================================
# Figure 2: Patient Cohort Distributions
# =============================================================================
fig2, axes = plt.subplots(1, 3, figsize=(14, 5))
fig2.suptitle('Virtual Patient Cohort Composition', fontsize=14, fontweight='bold')

# Population distribution
ax = axes[0]
pop_counts = df['population'].value_counts()
colors_pop = ['#4c6ef5', '#40c057', '#fab005', '#ff6b6b', '#845ef7']
bars = ax.bar(pop_counts.index, pop_counts.values, color=colors_pop, edgecolor='black')
ax.set_xlabel('Population')
ax.set_ylabel('Sample Count')
ax.set_title('Population Distribution')
for bar, cnt in zip(bars, pop_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, cnt + 200, f'{cnt:,}', ha='center', fontsize=9)

# Phenotype distribution
ax = axes[1]
pheno_order = ['PM', 'IM', 'NM', 'UM']
pheno_counts = df['phenotype'].value_counts().reindex(pheno_order)
colors_pheno = ['#ff6b6b', '#fab005', '#40c057', '#4c6ef5']
bars = ax.bar(pheno_counts.index, pheno_counts.values, color=colors_pheno, edgecolor='black')
ax.set_xlabel('CYP2D6 Phenotype')
ax.set_ylabel('Sample Count')
ax.set_title('CYP2D6 Phenotype Distribution')
for bar, cnt in zip(bars, pheno_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, cnt + 200, f'{cnt:,}', ha='center', fontsize=9)

# Activity score distribution
ax = axes[2]
ax.hist(df['activity_score'], bins=20, color='teal', edgecolor='white', alpha=0.8)
ax.set_xlabel('CYP2D6 Activity Score')
ax.set_ylabel('Frequency')
ax.set_title('Activity Score Distribution')
ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='PM (0)')
ax.axvline(1, color='orange', linestyle='--', alpha=0.5, label='IM (0.5-1)')
ax.axvline(2, color='green', linestyle='--', alpha=0.5, label='NM (1.5-2)')
ax.axvline(3, color='blue', linestyle='--', alpha=0.5, label='UM (>2)')
ax.legend(fontsize=8)

plt.tight_layout()
fig2.savefig(output_dir / 'cohort_distributions.png', dpi=fig_dpi, bbox_inches='tight')
print(f"Saved: {output_dir / 'cohort_distributions.png'}")

# =============================================================================
# Figure 3: Success Rate by Category
# =============================================================================
fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle('Efficacy Success Rate by Category', fontsize=14, fontweight='bold')

# Success rate by population
ax = axes[0]
success_by_pop = df.groupby('population')['efficacy_success'].mean() * 100
bars = ax.bar(success_by_pop.index, success_by_pop.values, color=colors_pop, edgecolor='black')
ax.set_xlabel('Population')
ax.set_ylabel('Success Rate (%)')
ax.set_title('Success Rate by Population')
ax.set_ylim(0, 100)
ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
for bar, rate in zip(bars, success_by_pop.values):
    ax.text(bar.get_x() + bar.get_width()/2, rate + 2, f'{rate:.1f}%', ha='center', fontsize=9)

# Success rate by phenotype
ax = axes[1]
success_by_pheno = df.groupby('phenotype')['efficacy_success'].mean().reindex(pheno_order) * 100
bars = ax.bar(success_by_pheno.index, success_by_pheno.values, color=colors_pheno, edgecolor='black')
ax.set_xlabel('CYP2D6 Phenotype')
ax.set_ylabel('Success Rate (%)')
ax.set_title('Success Rate by Phenotype')
ax.set_ylim(0, 100)
ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
for bar, rate in zip(bars, success_by_pheno.values):
    ax.text(bar.get_x() + bar.get_width()/2, rate + 2, f'{rate:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
fig3.savefig(output_dir / 'success_rate_by_category.png', dpi=fig_dpi, bbox_inches='tight')
print(f"Saved: {output_dir / 'success_rate_by_category.png'}")

print("\n✅ All visualizations saved!")
print(f"\n📁 Output directory: {output_dir.absolute()}")
