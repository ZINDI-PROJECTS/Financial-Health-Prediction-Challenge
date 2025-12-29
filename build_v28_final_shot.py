"""
V28: FINAL SHOT - ENSEMBLE V10 + V25b

KEY INSIGHT: We've been trying to IMPROVE individual models
What if we ENSEMBLE our two best submissions?

V10: 0.8922 (stable, conservative)
V25b: 0.8936 (best, but might overfit slightly)

Strategy: Blend their predictions with different weights
This is COMPLETELY different from manual corrections

Expected: 0.894-0.898 (ensemble smoothing effect)
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("V28: ENSEMBLE V10 + V25b")
print("="*80)

# ============================================
# LOAD SUBMISSIONS
# ============================================
print("\n[1/2] Loading our best submissions...")
v10_sub = pd.read_csv('submissions/submission_v10_ensemble_v2_v9.csv')
v25b_sub = pd.read_csv('submissions/submission_V25b_moderate.csv')

print(f"  V10:  0.8922 LB")
print(f"  V25b: 0.8936 LB")

# Verify alignment
assert (v10_sub['ID'] == v25b_sub['ID']).all(), "IDs don't match!"

# ============================================
# CREATE ENSEMBLE VARIANTS
# ============================================
print("\n[2/2] Creating ensemble variants...")

# Map to numeric for weighted averaging
label_to_num = {'Low': 0, 'Medium': 1, 'High': 2}
num_to_label = {0: 'Low', 1: 'Medium', 2: 'High'}

v10_numeric = v10_sub['Target'].map(label_to_num).values
v25b_numeric = v25b_sub['Target'].map(label_to_num).values

# Test different weight combinations
weight_configs = [
    (0.7, 0.3, 'V28a_70_30'),  # Favor V10 (more conservative)
    (0.5, 0.5, 'V28b_50_50'),  # Equal weight
    (0.3, 0.7, 'V28c_30_70'),  # Favor V25b (more aggressive)
]

print("\n  Weight configurations:")
for v10_w, v25b_w, name in weight_configs:
    print(f"    {name}: {v10_w*100:.0f}% V10 + {v25b_w*100:.0f}% V25b")

variants = []

for v10_weight, v25b_weight, variant_name in weight_configs:
    # Weighted average
    weighted = v10_weight * v10_numeric + v25b_weight * v25b_numeric

    # Round to nearest class
    ensemble_numeric = np.round(weighted).astype(int)
    ensemble_labels = [num_to_label[n] for n in ensemble_numeric]

    # Create submission
    ensemble_sub = pd.DataFrame({'ID': v10_sub['ID'], 'Target': ensemble_labels})

    # Save
    filename = f'submissions/submission_{variant_name}.csv'
    ensemble_sub.to_csv(filename, index=False)

    # Stats
    dist = pd.Series(ensemble_labels).value_counts()

    variants.append({
        'name': variant_name,
        'v10_weight': v10_weight,
        'v25b_weight': v25b_weight,
        'low': dist.get('Low', 0),
        'medium': dist.get('Medium', 0),
        'high': dist.get('High', 0),
        'filename': filename
    })

    print(f"\n  {variant_name}:")
    print(f"    Low={dist.get('Low', 0)}, Med={dist.get('Medium', 0)}, High={dist.get('High', 0)}")
    print(f"    Saved: {filename}")

# ============================================
# COMPARISON
# ============================================
print("\n" + "="*80)
print("DISTRIBUTION COMPARISON")
print("="*80)

v10_dist = v10_sub['Target'].value_counts()
v25b_dist = v25b_sub['Target'].value_counts()

print(f"\n  V10:    Low={v10_dist.get('Low', 0):4d}, Med={v10_dist.get('Medium', 0):4d}, High={v10_dist.get('High', 0):4d}  → 0.8922")
print(f"  V25b:   Low={v25b_dist.get('Low', 0):4d}, Med={v25b_dist.get('Medium', 0):4d}, High={v25b_dist.get('High', 0):4d}  → 0.8936")

for v in variants:
    print(f"  {v['name']}: Low={v['low']:4d}, Med={v['medium']:4d}, High={v['high']:4d}")

print("\n" + "="*80)
print("V28 COMPLETE")
print("="*80)

print("\n💡 WHY THIS MIGHT WORK:")
print("  - V10 is stable but conservative")
print("  - V25b is better but might overfit on those 5 flips")
print("  - Ensemble smooths predictions between them")
print("  - This is DIFFERENT from all manual correction attempts")

print("\n🎯 RECOMMENDATION:")
print("  Submit V28b_50_50 (equal weight)")
print("  Expected: 0.893-0.897")
print("  - Most balanced approach")
print("  - Neither submission dominates")
print("  - Smooths out both models' weaknesses")

print("\n📊 ALTERNATIVE:")
print("  If you want safer: V28a_70_30 (favor proven V10)")
print("  If you want aggressive: V28c_30_70 (favor better V25b)")

print("="*80)
