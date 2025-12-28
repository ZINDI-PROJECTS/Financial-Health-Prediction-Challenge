"""
STEP 2: Manual Class Weights Tuning
Test explicit class weights: Low:1, Medium:2-3, High:6-8
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
print("STEP 2: MANUAL CLASS WEIGHTS OPTIMIZATION")
print("="*80)

# Load data
print("\nLoading data...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

# Prepare features
feature_config = {'add_missing_indicators': False, 'log_transform_numerics': False}
X, y, X_test, test_ids, categorical_cols = prepare_features(train_df, test_df, config=feature_config)
y_encoded = y.map(get_label_mapping())

print(f"Features shape: {X.shape}")

# Test different class weight configurations
weight_configs = [
    {'name': 'Auto (baseline)', 'weights': None},
    {'name': 'Low:1, Med:2, High:6', 'weights': {0: 1.0, 1: 2.0, 2: 6.0}},
    {'name': 'Low:1, Med:2.5, High:7', 'weights': {0: 1.0, 1: 2.5, 2: 7.0}},
    {'name': 'Low:1, Med:3, High:8', 'weights': {0: 1.0, 1: 3.0, 2: 8.0}},
    {'name': 'Low:1, Med:2, High:10', 'weights': {0: 1.0, 1: 2.0, 2: 10.0}},
]

results_summary = []

for config in weight_configs:
    print(f"\n{'='*80}")
    print(f"TESTING: {config['name']}")
    print(f"{'='*80}")

    results = train_catboost_cv(
        X, y_encoded, categorical_cols,
        n_folds=5,
        class_weights=config['weights'],
        random_state=RANDOM_STATE
    )

    results_summary.append({
        'config': config['name'],
        'weights': config['weights'],
        'macro_f1': results['f1_global'],
        'f1_low': results['f1_per_class_global'][0],
        'f1_medium': results['f1_per_class_global'][1],
        'f1_high': results['f1_per_class_global'][2]
    })

# Print comparison
print(f"\n{'='*80}")
print("STEP 2 SUMMARY: CLASS WEIGHTS COMPARISON")
print(f"{'='*80}\n")
print(f"{'Configuration':<30} {'Macro-F1':>10} {'F1-Low':>8} {'F1-Med':>8} {'F1-High':>8}")
print("-" * 80)

for r in results_summary:
    print(f"{r['config']:<30} {r['macro_f1']:>10.6f} {r['f1_low']:>8.4f} {r['f1_medium']:>8.4f} {r['f1_high']:>8.4f}")

# Find best config
best = max(results_summary, key=lambda x: x['macro_f1'])
print(f"\n{'='*80}")
print(f"BEST CONFIGURATION: {best['config']}")
print(f"  Macro-F1: {best['macro_f1']:.6f}")
print(f"  F1 per class: Low={best['f1_low']:.4f}, Med={best['f1_medium']:.4f}, High={best['f1_high']:.4f}")
print(f"  Weights: {best['weights']}")
print(f"{'='*80}")

# Check if Medium collapsed
print(f"\nVALIDATION:")
baseline_medium = results_summary[0]['f1_medium']
for r in results_summary[1:]:
    if r['f1_medium'] < baseline_medium - 0.05:
        print(f"  ⚠ WARNING: {r['config']} collapsed Medium class!")
        print(f"    Baseline Med F1: {baseline_medium:.4f}")
        print(f"    Current Med F1: {r['f1_medium']:.4f}")
    else:
        gain_high = r['f1_high'] - results_summary[0]['f1_high']
        loss_med = baseline_medium - r['f1_medium']
        net_macro = r['macro_f1'] - results_summary[0]['macro_f1']
        print(f"  ✓ {r['config']}: High +{gain_high:.4f}, Med {-loss_med:+.4f}, Net {net_macro:+.4f}")
