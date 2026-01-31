import pandas as pd
import numpy as np

df = pd.read_csv('data/processed/drugbank_pk_parameters.csv')
df = df[df['half_life_value'].notna()].copy()

def get_label(groups):
    if pd.isna(groups): return None
    g = groups.lower()
    if 'withdrawn' in g: return 0
    elif 'approved' in g: return 1
    return None

df['label'] = df['groups'].apply(get_label)
df = df[df['label'].notna()].copy()
# Already biotech drugs from parser, no need for mass filter

print(f'Antibodies: {len(df)}')
print(f'Approved: {int((df.label == 1).sum())}, Withdrawn: {int((df.label == 0).sum())}')

df['pred'] = (df['half_life_value'] >= 100).astype(int)
acc = (df['label'] == df['pred']).mean()
print(f'Accuracy: {acc*100:.1f}%')

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(df['label'], df['pred'], labels=[0, 1])
print(f'\nConfusion Matrix:')
print(f'Actual FAIL:    {cm[0,0]:3d} correct, {cm[0,1]:3d} wrong')
print(f'Actual SUCCESS: {cm[1,0]:3d} wrong, {cm[1,1]:3d} correct')

# Show withdrawn drugs
print('\nWithdrawn drugs:')
wd = df[df['label'] == 0][['name', 'half_life_value', 'pred']]
for _, r in wd.iterrows():
    s = 'CORRECT' if r['pred'] == 0 else 'WRONG'
    print(f"  {r['name']}: t1/2={r['half_life_value']:.0f}h [{s}]")
