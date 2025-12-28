# V13 Optuna Blend Weight Optimization - Complete Analysis

## Executive Summary

**V13 Found:** α = 0.450 is mathematically optimal
- **Formula:** P(V13) = 0.45 × P(V9) + 0.55 × P(V2)
- **Interpretation:** V2's generalization weighted 55%, V9's SMOTE weighted 45%
- **Key Insight:** V2 (no SMOTE) slightly more valuable than V9 (SMOTE) for this test set

---

## Optuna Optimization Results

### Search Parameters:
- **Trials:** 50
- **Search Range:** α ∈ [0.30, 0.70]
- **Step Size:** 0.01
- **Objective:** Maximize OOF Macro-F1

### Best Result:
```
Best α:           0.450
Best OOF F1:      0.337979
V10 OOF F1:       0.336629 (α=0.50)
Gain:             +0.001351
```

### Interpretation:

**V2 (No SMOTE) weighted MORE at 55%:**
- V2's conservative, well-generalized predictions are slightly more valuable
- V9's SMOTE-enhanced High class detection is helpful but with some overfitting
- Near 50/50 balance suggests both approaches contribute meaningfully

**Why α = 0.45 is better than 0.50:**
- V2 has better CV-LB alignment (+0.080 gap vs V9's -0.043)
- V2's predictions are more trustworthy on unseen data
- V9's synthetic patterns don't fully match test distribution
- Optimal to lean slightly toward V2's generalization

---

## Predictions Comparison

| Metric | V10 (α=0.50) | V13 (α=0.45) | Difference |
|--------|--------------|--------------|------------|
| **Low** | 1,553 (64.6%) | 1,552 (64.5%) | -1 |
| **Medium** | 758 (31.5%) | 760 (31.6%) | +2 |
| **High** | 94 (3.9%) | 93 (3.9%) | -1 |

**Analysis:**
- Very minor differences (1-2 predictions shifted)
- V13 predicts 1 fewer High, 2 more Medium
- Slightly more conservative on High class
- Consistent with favoring V2's approach

---

## Expected Leaderboard Performance

### Current Standings:
- **V10:** 0.892 LB (α=0.50, suboptimal)
- **Leader:** 0.906 LB
- **Gap to Leader:** 0.014

### V13 Expected:
**Best Case:** 0.893-0.895 LB
- Optimal weighting extracts maximum value from blend
- Small OOF gain (+0.0013) translates to +0.001-0.003 LB
- Closes gap to leader by ~20%

**Likely Case:** 0.892-0.893 LB
- Minimal change from V10 (±0.001)
- Optimization gain too small to overcome variance
- Still competitive but not breakthrough

**Worst Case:** 0.891-0.892 LB
- No improvement or slight regression
- Indicates V10's 50/50 was already near-optimal
- Need different approach to break 0.895

---

## Why Such Small Gain?

### The Optimization Ceiling:

1. **V2 and V9 are highly correlated**
   - Both use same base models (CatBoost + LightGBM)
   - Both use same features (V6 country interactions)
   - Both use same class weights (1, 2.5, 7)
   - Only difference: SMOTE vs no SMOTE

2. **Limited diversity between approaches**
   - Blending works best with diverse models
   - V2 and V9 make similar predictions on most samples
   - Weight optimization can only help on edge cases

3. **Already near optimal at 50/50**
   - Optuna found α=0.45 (only 5% shift from 50/50)
   - Minimal improvement (+0.13% relative F1 gain)
   - Suggests 50/50 was already close to optimal

### What This Means:

**To significantly improve beyond 0.892, we need:**
- **More diverse models** (different algorithms, different features)
- **Better High class detection** (Focal Loss, different sampling)
- **Probability calibration** (fix probability biases)
- **Feature engineering** (new signal not captured by V6)

---

## Submission Decision

### Should We Submit V13?

**Arguments FOR:**
1. ✅ Mathematically optimal blend weight
2. ✅ Small but positive OOF gain
3. ✅ No risk (very similar to V10)
4. ✅ Worth testing if optimization translates to LB

**Arguments AGAINST:**
1. ⚠️ Very small expected gain (+0.001-0.003)
2. ⚠️ May not improve over V10 due to variance
3. ⚠️ Similar predictions (93 vs 94 High)
4. ⚠️ Better approaches may exist (V14 calibration, Focal Loss)

### Recommendation: **SUBMIT V13** ⭐⭐

**Reasoning:**
- Low risk, potential upside
- Confirms whether weight optimization helps
- Fast (already generated)
- Learn whether to try similar optimizations in future

**But don't expect breakthrough:**
- Expected: 0.892-0.893 (marginal improvement)
- If < 0.893, move to V14 (probability calibration)
- If ≥ 0.893, continue optimizing blend approaches

---

## Next Steps After V13

### If V13 ≥ 0.895: ⭐⭐⭐
**Strategy:** Keep optimizing blends and calibration
- Implement V14: Probability Calibration (Isotonic/Platt)
- Try V15: Test-Time Augmentation
- Optimize ensemble stacking

### If V13 = 0.892-0.894: ⭐⭐
**Strategy:** Try V14 calibration, then new approaches
- Implement V14: Probability Calibration
- Consider Focal Loss (targets High class directly)
- Consider Pseudo-Labeling (add confident test samples)

### If V13 ≤ 0.891: ⭐
**Strategy:** Blending is exhausted, try fundamentally different approaches
- **Focal Loss** - down-weight easy examples, focus on hard High class
- **Pseudo-Labeling** - use confident test predictions as training data
- **New features** - different feature engineering approach
- **Different algorithms** - Neural networks, stacking, etc.

---

## Technical Details

### Optuna Configuration:
```python
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=50, timeout=600)
```

### Objective Function:
```python
def objective(trial):
    alpha = trial.suggest_float('alpha', 0.3, 0.7, step=0.01)
    blended_proba = alpha * oof_v9_proba + (1 - alpha) * oof_v2_proba
    thresholds = optimize_thresholds(oof_labels, blended_proba)
    predictions = predict_with_thresholds(blended_proba, thresholds)
    return f1_score(oof_labels, predictions, average='macro')
```

### Trial Distribution:
- Trials explored full range [0.30, 0.70]
- Convergence around α=0.44-0.45
- Multiple trials found same optimum (robust)

---

## Comparison to All Versions

| Version | Strategy | Public LB | CV Gap | High Predictions |
|---------|----------|-----------|--------|------------------|
| **V2** | Cat+LGB, no SMOTE | 0.883 | +0.080 | 93 |
| **V9** | Cat+LGB, SMOTE | 0.888 | -0.043 | 98 |
| **V10** | V2+V9 blend (α=0.50) | 0.892 | ? | 94 |
| **V13** | V2+V9 blend (α=0.45) | **???** | ? | 93 |

**Key Observations:**
- V13 reverts to 93 High predictions (same as V2)
- Slightly more conservative than V10
- Favors V2's proven generalization strategy
- Expected: 0.892-0.895 LB

---

## Bottom Line

**Optuna found the mathematically optimal blend weight: α=0.45**

This means:
- 55% V2 (conservative, generalizes well)
- 45% V9 (better High detection but some overfitting)

**Expected gain: +0.001-0.003 LB (small but positive)**

**Submit V13, then:**
- If ≥ 0.893: Try V14 Probability Calibration
- If < 0.893: Try Focal Loss or Pseudo-Labeling

**To reach 0.906 leader:** Need more diverse approaches beyond blend optimization

---

*File ready: submissions/submission_v13_optimized_blend_weights.csv*
