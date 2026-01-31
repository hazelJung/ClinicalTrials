"""Extract parameter ranges from DrugBank data for batch_simulator update"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/drugbank_pk_parameters.csv')

print("=" * 50)
print("  DRUGBANK PARAMETER STATISTICS (Biotech Drugs)")
print("=" * 50)
print(f"Total drugs: {len(df)}")

# Half-life
hl = df['half_life_value'].dropna()
# Filter reasonable values (minutes to months)
hl_hr = hl[(hl > 0) & (hl < 10000)]  # up to ~400 days

print("\n📊 Half-life Distribution (hours):")
print(f"   Valid samples: {len(hl_hr)}")
print(f"   5th percentile: {hl_hr.quantile(0.05):.1f}")
print(f"   95th percentile: {hl_hr.quantile(0.95):.1f}")
print(f"   Median: {hl_hr.median():.1f}")
print(f"   Recommended range: ({hl_hr.quantile(0.05):.0f}, {hl_hr.quantile(0.95):.0f})")

# Mass data is sparse in DrugBank for biotech
# Use known antibody MW range
print("\n📊 Molecular Weight:")
print("   Antibodies: 140-160 kDa (known)")
print("   Peptides: 1-10 kDa (known)")

# Recommended config update
print("\n" + "=" * 50)
print("  RECOMMENDED CONFIG FOR batch_simulator.py")
print("=" * 50)
config_text = """
# From DrugBank biotech drug analysis:
halflife_range: ({:.0f}, {:.0f})  # 5th-95th percentile

# Literature-based ranges for antibodies:
kd_range: (0.01, 100)        # nM (ChEMBL typical range)
dose_range: (0.5, 50)        # mg/kg (FDA antibody approvals)
mw_range: (140, 160)         # kDa (IgG antibodies)
t0_range: (1, 100)           # nM (target expression)
""".format(hl_hr.quantile(0.05), hl_hr.quantile(0.95))
print(config_text)

# Save summary
with open('data/processed/drugbank_parameter_summary.txt', 'w') as f:
    f.write("DRUGBANK BIOTECH PARAMETER SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Half-life (hours):\n")
    f.write(f"  5th percentile: {hl_hr.quantile(0.05):.1f}\n")
    f.write(f"  95th percentile: {hl_hr.quantile(0.95):.1f}\n")
    f.write(f"  Recommended range: ({hl_hr.quantile(0.05):.0f}, {hl_hr.quantile(0.95):.0f})\n")
    f.write(config_text)

print("✅ Summary saved to: data/processed/drugbank_parameter_summary.txt")
