"""
STEP 3: Missingness as Signal
Add binary missing indicators for high-impact features
"""
import sys
sys.path.append('src')

import numpy as np
import pandas as pd
from features import prepare_features, get_label_mapping
from model import train_catboost_cv

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("STEP 3: MISSINGNESS INDICATORS")
print("="*80)

# Load data
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

# Key features with high missingness and high predictive power (from EDA)
missing_cols = [
    'funeral_insurance',      # 43.5% missing, Cramér's V: 0.451 (top feature!)
    'has_loan_account',       # 41.6% missing, Cramér's V: 0.289
    'personal_income',        # 1.1% missing but strong predictor
    'business_turnover',      # 2.2% missing but strongest numeric
    'medical_insurance',      # 43.5% missing, Cramér's V: 0.246
    'has_debit_card',         # 41.6% missing, Cramér's V: 0.185
    'uses_informal_lender',   # 46.7% missing
    'uses_friends_family_savings'  # 46.7% missing
]

# Best class weights from STEP 2
best_weights = {0: 1.0, 1: 2.5, 2: 7.0}

# Test configurations
configs = [
    {
        'name': 'BASELINE (no indicators)',
        'config': {
            'add_missing_indicators': False,
            'log_transform_numerics': False
        }
    },
    {
        'name': 'Top 4 missing indicators',
        'config': {
            'add_missing_indicators': True,
            'missing_indicator_cols': missing_cols[:4],
            'log_transform_numerics': False
        }
    },
    {
        'name': 'All 8 missing indicators',
        'config': {
            'add_missing_indicators': True,
            'missing_indicator_cols': missing_cols,
            'log_transform_numerics': False
        }
    }
]

results_summary = []

for cfg in configs:
    print(f"\n{'='*80}")
    print(f"TESTING: {cfg['name']}")
    print(f"{'='*80}")

    # Prepare features with current config
    X, y, X_test, test_ids, categorical_cols = prepare_features(
        train_df, test_df, config=cfg['config']
    )
    y_encoded = y.map(get_label_mapping())

    print(f"Features shape: {X.shape}")

    # Train with best class weights
    results = train_catboost_cv(
        X, y_encoded, categorical_cols,
        n_folds=5,
        class_weights=best_weights,
        random_state=RANDOM_STATE
    )

    results_summary.append({
        'config': cfg['name'],
        'n_features': X.shape[1],
        'macro_f1': results['f1_global'],
        'f1_low': results['f1_per_class_global'][0],
        'f1_medium': results['f1_per_class_global'][1],
        'f1_high': results['f1_per_class_global'][2]
    })

# Print comparison
print(f"\n{'='*80}")
print("STEP 3 SUMMARY: MISSINGNESS INDICATORS")
print(f"{'='*80}\n")
print(f"{'Configuration':<35} {'Features':>8} {'Macro-F1':>10} {'F1-Low':>8} {'F1-Med':>8} {'F1-High':>8}")
print("-" * 90)

baseline_f1 = results_summary[0]['macro_f1']
for r in results_summary:
    gain = r['macro_f1'] - baseline_f1
    gain_str = f"({gain:+.4f})" if r['config'] != 'BASELINE (no indicators)' else ""
    print(f"{r['config']:<35} {r['n_features']:>8} {r['macro_f1']:>10.6f} {gain_str:>9} "
          f"{r['f1_low']:>8.4f} {r['f1_medium']:>8.4f} {r['f1_high']:>8.4f}")

# Find best
best = max(results_summary, key=lambda x: x['macro_f1'])
print(f"\n{'='*80}")
print(f"BEST: {best['config']}")
print(f"  Macro-F1: {best['macro_f1']:.6f}")
print(f"  Improvement over baseline: {best['macro_f1'] - baseline_f1:+.6f}")
print(f"  Per-class: Low={best['f1_low']:.4f}, Med={best['f1_medium']:.4f}, High={best['f1_high']:.4f}")
print(f"{'='*80}")

# Decision
if best['macro_f1'] - baseline_f1 > 0.002:
    print(f"\n✓ ACCEPTED: Missingness indicators improve macro-F1 by {best['macro_f1'] - baseline_f1:.4f}")
else:
    print(f"\n✗ REJECTED: Gain too small ({best['macro_f1'] - baseline_f1:.4f}), not worth complexity")
