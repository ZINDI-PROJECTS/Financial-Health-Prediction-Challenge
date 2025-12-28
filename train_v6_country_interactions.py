"""
V6: COUNTRY-SPECIFIC INTERACTIONS

Breakthrough hypothesis:
Financial health definitions vary by country (χ²=831, p<0.000001)
- Eswatini: 11.5% High (profit_margin~2.0)
- Lesotho: 0.3% High (profit_margin~4.75)

Strategy:
- V4 refined features (credit_access, profit_margin, business_maturity)
- + Country interactions (country × profit_margin, × insurance, × credit_access)
- CatBoost + LightGBM ensemble

Expected gain: +0.01-0.03 if country effects are strong (HIGH POTENTIAL)
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
print("V6: COUNTRY-SPECIFIC INTERACTIONS")
print("="*80)

# ============================================
# 1. LOAD AND PREPARE DATA
# ============================================
print("\n[1/6] Loading data with V6 country-interaction features...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

X, y, X_test, test_ids, categorical_cols = prepare_data_v6(train_df, test_df)

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_encoded = y.map(label_mapping)

print(f"Features: {X.shape[1]} (37 original + 3 V4 + 12 country interactions)")
print(f"Samples: {len(X):,}")

# ============================================
# 2. PREPARE FOR LIGHTGBM (ENCODE CATEGORICALS)
# ============================================
print("\n[2/6] Encoding categorical features for LightGBM...")

X_cat = X.copy()  # For CatBoost
X_lgb = X.copy()  # For LightGBM
X_test_cat = X_test.copy()
X_test_lgb = X_test.copy()

# Encode categoricals for LightGBM
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]])
    le.fit(combined)
    X_lgb[col] = le.transform(X[col])
    X_test_lgb[col] = le.transform(X_test[col])

print(f"  CatBoost: {X_cat.shape} (native categorical support)")
print(f"  LightGBM: {X_lgb.shape} (label encoded)")

# ============================================
# 3. CROSS-VALIDATION WITH BLENDING
# ============================================
print("\n[3/6] Training ensemble with 5-fold CV...")

class_weights = {0: 1.0, 1: 2.5, 2: 7.0}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

fold_f1_scores = []
all_y_true = []
all_y_proba_blend = []

print(f"\n{'='*80}")
print("CROSS-VALIDATION: CatBoost + LightGBM with Country Interactions")
print(f"{'='*80}")

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_cat, y_encoded), 1):
    print(f"\nFold {fold_idx}/5:")

    X_train_cat, X_val_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
    X_train_lgb, X_val_lgb = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]
    y_train, y_val = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]

    # Train CatBoost
    print("  Training CatBoost...", end=' ')
    cat_model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='TotalF1',
        class_weights=class_weights,
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_state=RANDOM_STATE,
        verbose=False,
        early_stopping_rounds=50
    )
    cat_model.fit(X_train_cat, y_train, cat_features=categorical_cols,
                  eval_set=(X_val_cat, y_val), verbose=False)
    cat_proba = cat_model.predict_proba(X_val_cat)
    print("Done")

    # Train LightGBM
    print("  Training LightGBM...", end=' ')
    lgb_model = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        metric='multi_logloss',
        class_weight=class_weights,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        verbose=-1,
        early_stopping_rounds=50
    )
    lgb_model.fit(X_train_lgb, y_train, eval_set=[(X_val_lgb, y_val)])
    lgb_proba = lgb_model.predict_proba(X_val_lgb)
    print("Done")

    # Blend probabilities (simple mean)
    blend_proba = (cat_proba + lgb_proba) / 2

    # Optimize thresholds
    thresholds = optimize_thresholds(y_val.values, blend_proba)
    y_pred = predict_with_thresholds(blend_proba, thresholds)

    # Metrics
    f1 = f1_score(y_val, y_pred, average='macro')
    f1_per_class = f1_score(y_val, y_pred, average=None)

    fold_f1_scores.append(f1)
    all_y_true.extend(y_val.values)
    all_y_proba_blend.append(blend_proba)

    print(f"  Blend F1: {f1:.6f} | Low={f1_per_class[0]:.4f}, Med={f1_per_class[1]:.4f}, High={f1_per_class[2]:.4f}")

# Aggregate results
all_y_proba_blend = np.vstack(all_y_proba_blend)
all_y_true = np.array(all_y_true)

global_thresholds = optimize_thresholds(all_y_true, all_y_proba_blend)
y_pred_global = predict_with_thresholds(all_y_proba_blend, global_thresholds)

cv_f1 = f1_score(all_y_true, y_pred_global, average='macro')
f1_per_class = f1_score(all_y_true, y_pred_global, average=None)

# Confusion matrix
cm = confusion_matrix(all_y_true, y_pred_global)

print(f"\n{'='*80}")
print("V6 CV RESULTS")
print(f"{'='*80}")
print(f"\nCV Macro-F1: {cv_f1:.6f}")
print(f"\nPer-class F1:")
print(f"  Low (0):    {f1_per_class[0]:.4f}")
print(f"  Medium (1): {f1_per_class[1]:.4f}")
print(f"  High (2):   {f1_per_class[2]:.4f}")
print(f"\nConfusion Matrix:")
print(f"         Pred_Low  Pred_Med  Pred_High")
print(f"True_Low    {cm[0,0]:5d}    {cm[0,1]:5d}     {cm[0,2]:5d}")
print(f"True_Med    {cm[1,0]:5d}    {cm[1,1]:5d}     {cm[1,2]:5d}")
print(f"True_High   {cm[2,0]:5d}    {cm[2,1]:5d}     {cm[2,2]:5d}")

# Compare to baselines
print(f"\n{'='*80}")
print("PERFORMANCE COMPARISON")
print(f"{'='*80}")
print(f"  V2 (Blend, no features):    0.803 CV → 0.883 Public")
print(f"  V5 (Blend + V4 features):   0.800 CV → ???")
print(f"  V6 (Blend + Country Int):   {cv_f1:.3f} CV → ???")

gain_v2 = cv_f1 - 0.802901
gain_v5 = cv_f1 - 0.799639

print(f"\n  Gain vs V2:  {gain_v2:+.6f}")
print(f"  Gain vs V5:  {gain_v5:+.6f}")

if cv_f1 > 0.802901:
    expected_public = cv_f1 + 0.08  # Approximate gap from V2
    print(f"\n  🎯 V6 BEATS V2 CV! Country interactions working!")
    print(f"  Expected public: ~{expected_public:.3f}")
    if expected_public >= 0.90:
        print(f"  🔥 PROJECTED TO HIT 0.90+ TARGET!")
    elif expected_public >= 0.89:
        print(f"  ⚡ Close to target! One more optimization might hit 0.90!")
elif cv_f1 > 0.799639:
    print(f"\n  ✓ V6 beats V5! Country interactions added signal.")
    print(f"  Expected public: ~{cv_f1 + 0.08:.3f}")
else:
    print(f"\n  ⚠ Country interactions didn't help in CV. May still help on test.")

# ============================================
# 4. TRAIN FINAL MODELS ON FULL DATA
# ============================================
print(f"\n[4/6] Training final ensemble on 100% of data...")

# CatBoost
print("  Training CatBoost...", end=' ')
final_cat = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    class_weights=class_weights,
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=RANDOM_STATE,
    verbose=False
)
final_cat.fit(X_cat, y_encoded, cat_features=categorical_cols, verbose=False)
print("Done")

# LightGBM
print("  Training LightGBM...", end=' ')
final_lgb = LGBMClassifier(
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
final_lgb.fit(X_lgb, y_encoded)
print("Done")

# ============================================
# 5. FEATURE IMPORTANCE (CATBOOST)
# ============================================
print(f"\n[5/6] Feature Importance (CatBoost Top 15):")
importance = final_cat.get_feature_importance()
importance_df = pd.DataFrame({
    'feature': X_cat.columns,
    'importance': importance
}).sort_values('importance', ascending=False)

v6_features = [col for col in X_cat.columns if '_x_' in col or col in ['credit_access_score', 'profit_margin', 'business_maturity']]

for idx, row in importance_df.head(15).iterrows():
    marker = "🆕" if row['feature'] in v6_features else "  "
    print(f"  {marker} {row['feature']:<50s}: {row['importance']:>6.2f}")

# Check country interaction rankings
print(f"\n  Country interaction rankings:")
country_interactions = importance_df[importance_df['feature'].str.contains('_x_')]
for idx, row in country_interactions.head(5).iterrows():
    rank = importance_df.index.get_loc(idx) + 1
    print(f"    #{rank:2d} - {row['feature']:<50s}: {row['importance']:>6.2f}")

# ============================================
# 6. GENERATE TEST PREDICTIONS
# ============================================
print(f"\n[6/6] Generating final test predictions...")

# Predict with both models
cat_test_proba = final_cat.predict_proba(X_test_cat)
lgb_test_proba = final_lgb.predict_proba(X_test_lgb)

# Blend
blend_test_proba = (cat_test_proba + lgb_test_proba) / 2

# Apply learned thresholds
test_pred_encoded = predict_with_thresholds(blend_test_proba, global_thresholds)
test_predictions = [reverse_mapping[p] for p in test_pred_encoded]

# Distribution
pred_counts = pd.Series(test_predictions).value_counts()
print(f"\nTest predictions:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = count / len(test_predictions) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

# Save submission
submission = pd.DataFrame({'ID': test_ids, 'Target': test_predictions})
filename = 'submissions/submission_v6_country_interactions.csv'
submission.to_csv(filename, index=False)

print(f"\n{'='*80}")
print("V6 SUBMISSION COMPLETE")
print(f"{'='*80}")
print(f"\nFile: {filename}")
print(f"Strategy: CatBoost + LightGBM WITH country-specific interactions")
print(f"Features: {X.shape[1]} (37 base + 3 V4 + 12 country interactions)")
print(f"CV Macro-F1: {cv_f1:.6f}")
print(f"Class weights: {class_weights}")
print(f"\nPer-class F1:")
print(f"  Low:    {f1_per_class[0]:.4f}")
print(f"  Medium: {f1_per_class[1]:.4f}")
print(f"  High:   {f1_per_class[2]:.4f}")
print(f"\nExpected Public LB: {cv_f1 + 0.08:.3f} (if CV→LB gap holds)")
print(f"{'='*80}")

if cv_f1 > 0.802901:
    print("\n🔥 BREAKTHROUGH! Country interactions unlocked new signal!")
    print("This could be our path to 0.90+!")
else:
    print("\n📊 Country interactions tested. Check if test distribution differs from CV.")
print("="*80)
