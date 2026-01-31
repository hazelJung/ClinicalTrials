import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os

# Define the specific drugs to test
drugs = [
    {
        'name': 'Proleukin (Aldesleukin)',
        'type': 'Cytokine (IL-2)',
        'MW': 15.3,          # kDa
        'halflife': 1.42,    # hours
        'dose': 0.037,       # mg/kg (single dose)
        'KD': 0.01,          # nM (High affinity)
        'T0': 10.0,          # nM (Est. receptor conc)
        'status': 'Approved'
    },
    {
        'name': 'Nemvaleukin alfa (ALKS 4230)',
        'type': 'Fusion Protein',
        'MW': 34.4,          # kDa
        'halflife': 10.0,    # hours (Est.)
        'dose': 0.006,       # mg/kg (6 ug/kg)
        'KD': 1.0,           # nM (Intermediate affinity)
        'T0': 10.0,          # nM
        'status': 'Phase 3 (Fast Track)'
    }
]

# Load training data
data_path = 'data/processed/mpbpk_training_data_latest.csv'
if not os.path.exists(data_path):
    print(f"Error: Training data not found at {data_path}")
    exit(1)

print("Loading and preparing training data...")
df_train = pd.read_csv(data_path)

# ---------------------------------------------------------
# Feature Engineering (Replicated from mpbpk_ml.py)
# ---------------------------------------------------------

FEATURE_COLUMNS_NUMERIC = [
    'log_KD', 'log_dose', 'log_potency', 'charge', 'log_MW',
    'log_T0', 'log_halflife',
    'activity_score', 'cl_multiplier',
]

def prepare_features(df):
    X_list = []
    feature_names = FEATURE_COLUMNS_NUMERIC.copy()
    
    # Numeric features
    for col in FEATURE_COLUMNS_NUMERIC:
        if col in df.columns:
            X_list.append(df[col].values.reshape(-1, 1))
        elif col == 'log_potency':
            if 'log_dose' in df.columns and 'log_KD' in df.columns:
                # Calculate potency on the fly if not in dataframe
                potency = df['log_dose'].values - df['log_KD'].values
                X_list.append(potency.reshape(-1, 1))
        else:
            print(f"Warning: {col} missing in dataframe")
    
    # One-hot encode population
    # We need to manually handle this to ensure columns match between Train and Test
    # In training, we use pd.get_dummies. In test, we must have same columns.
    
    pops = ['pop_AFR', 'pop_AMR', 'pop_EAS', 'pop_EUR', 'pop_SAS']
    if 'population' in df.columns:
        # Check if dummy columns already exist (unlikely if raw)
        # Create dummies
        pop_dummies = pd.get_dummies(df['population'], prefix='pop')
        # Ensure all expected columns exist
        for p in pops:
            if p not in pop_dummies.columns:
                pop_dummies[p] = 0
        
        # Add in specific order
        for p in pops:
            X_list.append(pop_dummies[p].values.reshape(-1, 1))
            feature_names.append(p)
    elif all(p in df.columns for p in pops):
         # If already encoded/provided
         for p in pops:
            X_list.append(df[p].values.reshape(-1, 1))
            feature_names.append(p)

    # Encode phenotype as ordinal
    if 'phenotype' in df.columns:
        phenotype_map = {'PM': 0, 'IM': 1, 'NM': 2, 'UM': 3}
        pheno_encoded = df['phenotype'].map(phenotype_map).fillna(2).values
        X_list.append(pheno_encoded.reshape(-1, 1))
        feature_names.append('phenotype_encoded')
    elif 'phenotype_encoded' in df.columns:
        X_list.append(df['phenotype_encoded'].values.reshape(-1, 1))
        feature_names.append('phenotype_encoded')

    X = np.hstack(X_list)
    return X, feature_names

# Prepare Training Data
X_train, feature_names_train = prepare_features(df_train)
y_train = df_train['efficacy_success'].values # Or 'success' check column name

# Prepare Test Data
test_rows = []
for drug in drugs:
    # We will simulate a "Standard Patient" (NM phenotype, European) for the prediction
    row = {}
    row['log_MW'] = np.log(drug['MW'])
    row['log_halflife'] = np.log(drug['halflife'])
    row['log_dose'] = np.log(drug['dose'])
    row['log_KD'] = np.log(drug['KD'])
    row['log_T0'] = np.log(drug['T0'])
    
    row['charge'] = 0.0
    row['activity_score'] = 1.63
    row['cl_multiplier'] = 1.34
    
    # Categorical (pre-dummy for simplicity or string)
    # We'll use the 'pre-encoded' path of the function
    row['pop_AFR'] = 0
    row['pop_AMR'] = 0
    row['pop_EAS'] = 0
    row['pop_EUR'] = 1
    row['pop_SAS'] = 0
    row['phenotype_encoded'] = 2 # NM
    
    test_rows.append(row)

df_test = pd.DataFrame(test_rows)
X_test, feature_names_test = prepare_features(df_test)

# Check alignment
if feature_names_train != feature_names_test:
    print("Error: Feature mismatch!")
    print("Train:", feature_names_train)
    print("Test:", feature_names_test)
    exit(1)

# Train
print("Training Random Forest...")
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, class_weight='balanced')
clf.fit(X_train_imputed, y_train)

# Predict
probs = clf.predict_proba(X_test_imputed)[:, 1]

output_file = 'prediction_results.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("  PREDICTION RESULTS FOR CYTOKINES\n")
    f.write("="*60 + "\n")

    for i, drug in enumerate(drugs):
        prob = probs[i]
        f.write(f"\nDrug: {drug['name']}\n")
        f.write(f"   Type: {drug['type']}\n")
        f.write(f"   PK: MW={drug['MW']}kDa, T1/2={drug['halflife']}h, KD={drug['KD']}nM, Dose={drug['dose']}mg/kg\n")
        f.write(f"   Real Status: {drug['status']}\n")
        f.write(f"   Model Probability: {prob:.1%}\n")
        Verdict = "SUCCESS/HIGH" if prob > 0.6 else "FAIL/LOW" if prob < 0.4 else "UNCERTAIN"
        f.write(f"   Limit Check: Dose (Log {np.log(drug['dose']):.2f}) | Train Range: -0.69 to 3.0\n")
        f.write(f"   Verdict: {Verdict}\n")

    f.write("\nAnalysis:\n")
    f.write("- The model was trained on antibodies (High MW, High Dose).\n")
    f.write("- Cytokines have extremely low doses and short half-lives.\n")
    f.write("- If probability is low, it likely reflects the 'Penalty' for low dose/half-life in an Antibody model.\n")

print(f"Results saved to {output_file}")
