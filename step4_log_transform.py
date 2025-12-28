"""
STEP 4: Log Transform Heavy-Tailed Numerics
Apply log1p to skewed financial features
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
print("STEP 4: LOG TRANSFORM HEAVY-TAILED NUMERICS")
print("="*80)

# Load data
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

# Heavy-tailed numeric features from EDA (high skewness)
# personal_income: skew=33.6, business_expenses: skew=60.5, business_turnover: skew=23.5
log_transform_cols = [
    'personal_income',
    'business_expenses',
    'business_turnover'
]

# Best class weights from STEP 2
best_weights = {0: 1.0, 1: 2.5, 2: 7.0}

# Test configurations
configs = [
    {
        'name': 'BASELINE (no transforms)',
        'config': {
            'add_missing_indicators': False,
            'log_transform_numerics': False
        }
    },
    {
        'name': 'Log transform: income, expenses, turnover',
        'config': {
            'add_missing_indicators': False,
            'log_transform_numerics': True,
            'log_transform_cols': log_transform_cols
        }
    },
    {
        'name': 'Log transform: income + turnover only',
        'config': {
            'add_missing_indicators': False,
            'log_transform_numerics': True,
            'log_transform_cols': ['personal_income', 'business_turnover']
        }
    },
    {
        'name': 'Log transform: turnover only',
        'config': {
            'add_missing_indicators': False,
            'log_transform_numerics': True,
            'log_transform_cols': ['business_turnover']
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
print("STEP 4 SUMMARY: LOG TRANSFORMS")
print(f"{'='*80}\n")
print(f"{'Configuration':<45} {'Features':>8} {'Macro-F1':>10} {'F1-Low':>8} {'F1-Med':>8} {'F1-High':>8}")
print("-" * 100)

baseline_f1 = results_summary[0]['macro_f1']
for r in results_summary:
    gain = r['macro_f1'] - baseline_f1
    gain_str = f"({gain:+.4f})" if r['config'] != 'BASELINE (no transforms)' else ""
    print(f"{r['config']:<45} {r['n_features']:>8} {r['macro_f1']:>10.6f} {gain_str:>9} "
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
    print(f"\n✓ ACCEPTED: Log transforms improve macro-F1 by {best['macro_f1'] - baseline_f1:.4f}")
else:
    print(f"\n✗ REJECTED: Gain too small ({best['macro_f1'] - baseline_f1:.4f}), not worth complexity")
