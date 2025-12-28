"""
V10: ENSEMBLE V2 + V9 STRATEGIES

Strategy:
- Train V2 approach (no SMOTE, weights 1,2.5,7) → Get test probabilities
- Train V9 approach (SMOTE, weights 1,2.5,7) → Get test probabilities
- Average probabilities → Predict
- Expected: Combine V2's generalization + V9's High class detection

V2: 0.883 LB (conservative, generalizes well)
V9: 0.888 LB (better High class, some overfitting)
Expected V10: ~0.891-0.895
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')
from engineer_features_v6 import prepare_data_v6

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def optimize_thresholds(y_true, y_proba):
    """Find optimal thresholds for macro-F1"""
    def objective(thresholds):
        predictions = np.argmax(y_proba - thresholds, axis=1)
        return -f1_score(y_true, predictions, average='macro')
    result = minimize(objective, np.zeros(3), method='Nelder-Mead', options={'maxiter': 500})
    return result.x

def predict_with_thresholds(y_proba, thresholds):
    return np.argmax(y_proba - thresholds, axis=1)

print("="*80)
print("V10: ENSEMBLE V2 (NO SMOTE) + V9 (SMOTE)")
print("="*80)

# ============================================
# LOAD DATA
# ============================================
print("\n[1/6] Loading data...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

X, y, X_test, test_ids, categorical_cols = prepare_data_v6(train_df, test_df)

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_encoded = y.map(label_mapping)

print(f"Features: {X.shape[1]}")
print(f"Samples: {len(X):,}")

# ============================================
# PREPARE V2 DATA (NO SMOTE)
# ============================================
print("\n[2/6] Preparing V2 data (original, no SMOTE)...")
X_v2 = X.copy()
X_test_v2 = X_test.copy()
y_v2 = y_encoded.copy()

# ============================================
# PREPARE V9 DATA (WITH SMOTE)
# ============================================
print("\n[3/6] Preparing V9 data (SMOTE augmented)...")

# Encode categoricals for SMOTE
X_v9 = X.copy()
X_test_v9 = X_test.copy()

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]])
    le.fit(combined)
    X_v9[col] = le.transform(X[col])
    X_test_v9[col] = le.transform(X_test[col])

# Apply SMOTE
sampling_strategy = {
    0: (y_encoded == 0).sum(),
    1: (y_encoded == 1).sum(),
    2: 1000
}

smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE, k_neighbors=5)
X_v9_resampled, y_v9 = smote.fit_resample(X_v9, y_encoded)
X_v9_resampled = pd.DataFrame(X_v9_resampled, columns=X_v9.columns)

print(f"  V9 samples after SMOTE: {len(X_v9_resampled):,}")

# ============================================
# TRAIN V2 MODELS (NO SMOTE)
# ============================================
print("\n[4/6] Training V2 models (no SMOTE) on full data...")

class_weights = {0: 1.0, 1: 2.5, 2: 7.0}

print("  Training V2 CatBoost...", end=' ')
v2_cat = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    class_weights=class_weights,
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=RANDOM_STATE,
    verbose=False
)
v2_cat.fit(X_v2, y_v2, cat_features=categorical_cols, verbose=False)
v2_cat_proba = v2_cat.predict_proba(X_test_v2)
print("Done")

print("  Training V2 LightGBM...", end=' ')
# Encode categoricals for LightGBM
X_v2_encoded = X_v2.copy()
X_test_v2_encoded = X_test_v2.copy()
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_v2[col], X_test_v2[col]])
    le.fit(combined)
    X_v2_encoded[col] = le.transform(X_v2[col])
    X_test_v2_encoded[col] = le.transform(X_test_v2[col])

v2_lgb = LGBMClassifier(
    objective='multiclass',
    num_class=3,
    metric='multi_logloss',
    class_weight=class_weights,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=RANDOM_STATE,
    verbose=-1
)
v2_lgb.fit(X_v2_encoded, y_v2)
v2_lgb_proba = v2_lgb.predict_proba(X_test_v2_encoded)
print("Done")

# Blend V2 probabilities
v2_proba = (v2_cat_proba + v2_lgb_proba) / 2

# ============================================
# TRAIN V9 MODELS (WITH SMOTE)
# ============================================
print("\n[5/6] Training V9 models (SMOTE augmented) on full data...")

print("  Training V9 CatBoost...", end=' ')
v9_cat = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    class_weights=class_weights,
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=RANDOM_STATE,
    verbose=False
)
v9_cat.fit(X_v9_resampled, y_v9, verbose=False)
v9_cat_proba = v9_cat.predict_proba(X_test_v9)
print("Done")

print("  Training V9 LightGBM...", end=' ')
v9_lgb = LGBMClassifier(
    objective='multiclass',
    num_class=3,
    metric='multi_logloss',
    class_weight=class_weights,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=RANDOM_STATE,
    verbose=-1
)
v9_lgb.fit(X_v9_resampled, y_v9)
v9_lgb_proba = v9_lgb.predict_proba(X_test_v9)
print("Done")

# Blend V9 probabilities
v9_proba = (v9_cat_proba + v9_lgb_proba) / 2

# ============================================
# ENSEMBLE V2 + V9
# ============================================
print("\n[6/6] Ensembling V2 + V9 probabilities...")

# Average probabilities (equal weight)
ensemble_proba = (v2_proba + v9_proba) / 2

# Make predictions (no threshold tuning - use argmax)
ensemble_pred_encoded = np.argmax(ensemble_proba, axis=1)
ensemble_predictions = [reverse_mapping[p] for p in ensemble_pred_encoded]

# Distribution
pred_counts = pd.Series(ensemble_predictions).value_counts()
print(f"\nV10 Ensemble predictions:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = count / len(ensemble_predictions) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

# Compare to V2 and V9
v2_sub = pd.read_csv('submissions/submission_v2_catboost_lgb_blend.csv')
v9_sub = pd.read_csv('submissions/submission_v9_smote_high_class.csv')

print(f"\nComparison:")
print(f"  V2:  Low={len(v2_sub[v2_sub['Target']=='Low'])}, Med={len(v2_sub[v2_sub['Target']=='Medium'])}, High={len(v2_sub[v2_sub['Target']=='High'])}")
print(f"  V9:  Low={len(v9_sub[v9_sub['Target']=='Low'])}, Med={len(v9_sub[v9_sub['Target']=='Medium'])}, High={len(v9_sub[v9_sub['Target']=='High'])}")
print(f"  V10: Low={pred_counts.get('Low',0)}, Med={pred_counts.get('Medium',0)}, High={pred_counts.get('High',0)}")

# Save
submission = pd.DataFrame({'ID': test_ids, 'Target': ensemble_predictions})
filename = 'submissions/submission_v10_ensemble_v2_v9.csv'
submission.to_csv(filename, index=False)

print(f"\n{'='*80}")
print("V10 COMPLETE")
print(f"{'='*80}")
print(f"File: {filename}")
print(f"Strategy: Ensemble V2 (no SMOTE) + V9 (SMOTE)")
print(f"Expected LB: ~0.891-0.895")
print(f"\nLogic:")
print(f"  - V2 (0.883 LB): Conservative, excellent generalization")
print(f"  - V9 (0.888 LB): Better High class detection")
print(f"  - V10: Combines strengths, reduces weaknesses")
print(f"{'='*80}")
