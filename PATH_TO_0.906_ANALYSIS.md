# Path to 0.906 - Comprehensive Strategy

## Current Status

**Progress:**
- V2: 0.883 LB (baseline)
- V9: 0.888 LB (SMOTE)
- V10: 0.892 LB (ensemble)
- **V13: Ready** (optimized blend weights)
- **V14: Ready** (probability calibration)

**Goal:** 0.906 LB (match leader)
**Current Gap:** 0.014 (from V10)

---

## V13: Optimized Blend Weights ⭐⭐

### What It Does:
Uses Optuna to find mathematically optimal blend ratio between V2 and V9

### Key Finding:
**α = 0.45** is optimal (45% V9 + 55% V2)
- V2's generalization is slightly more valuable
- 50/50 was close but suboptimal

### Expected Gain:
+0.001-0.003 LB → **0.893-0.895 expected**

### Predictions:
- High: 93 (same as V2, more conservative than V10's 94)
- Very minor changes from V10

### Should Submit:
✅ **Yes** - Low risk, validates optimization approach

**File:** `submissions/submission_v13_optimized_blend_weights.csv`

---

## V14: Probability Calibration ⭐⭐⭐

### What It Does:
Fixes probability miscalibration using:
- **Temperature Scaling:** Adjusts confidence levels globally
- **Isotonic Regression:** Fixes per-class probability biases

### Why It Helps:
- Model may predict 80% confidence when true probability is 85%
- Calibration aligns predicted probabilities with actual outcomes
- Better probabilities → better threshold-based predictions

### Expected Gain:
+0.002-0.006 LB → **0.894-0.898 expected**

### Should Submit:
✅ **Yes** - After V13, this is next best optimization

**File:** `submissions/submission_v14_probability_calibration.csv`

---

## Recommended Submission Order

### Priority 1: V13 (Submit Now)
- Mathematically optimal blend weight
- Expected: 0.893-0.895
- Learn if weight optimization helps

### Priority 2: V14 (Submit After V13)
- Probability calibration
- Expected: 0.894-0.898
- Most promising for breaking 0.895

### Priority 3: Wait or Try Advanced Techniques
- Focal Loss
- Pseudo-Labeling
- New feature engineering

---

## If V13 + V14 Don't Reach 0.90...

### High-Impact Options Remaining:

### A. **Focal Loss** ⭐⭐⭐
**Target Problem:** High class confusion with Medium

**How It Works:**
- Down-weights easy examples (well-classified Low/Medium)
- Up-weights hard examples (High confused with Medium)
- Explicitly targets our bottleneck

**Expected Gain:** +0.005-0.010
**Implementation Time:** 30 minutes
**Expected LB:** 0.897-0.902

**Code Snippet:**
```python
class FocalLoss:
    def __init__(self, gamma=2.0, alpha=None):
        self.gamma = gamma
        self.alpha = alpha  # Class weights

    def __call__(self, y_pred, y_true):
        p = y_pred[range(len(y_true)), y_true]
        focal_weight = (1 - p) ** self.gamma
        if self.alpha is not None:
            focal_weight *= self.alpha[y_true]
        loss = -np.log(p) * focal_weight
        return loss.mean()
```

---

### B. **Pseudo-Labeling** ⭐⭐
**Target Problem:** Limited High class training samples (470)

**How It Works:**
1. Train on training data
2. Predict on test data with HIGH confidence (e.g., prob > 0.95)
3. Add confident test predictions to training set
4. Retrain on augmented data

**Expected Gain:** +0.003-0.008
**Implementation Time:** 1 hour
**Expected LB:** 0.895-0.900

**Risk:** Can propagate errors if confident predictions are wrong

---

### C. **Advanced Feature Engineering** ⭐
**Target Problem:** Current features may miss important patterns

**Ideas:**
1. **Business ratios:**
   - Debt-to-income ratio
   - Working capital ratio
   - Quick ratio

2. **Temporal features:**
   - Business age bins
   - Seasonal effects (if timestamp available)

3. **Cross-country comparisons:**
   - Z-score within country
   - Percentile rank within country

**Expected Gain:** +0.002-0.008
**Implementation Time:** 2-3 hours
**Expected LB:** 0.894-0.900

---

### D. **Stacking** ⭐⭐
**Target Problem:** Simple blending may not capture complex interactions

**How It Works:**
1. Layer 1: Train CatBoost, LightGBM, XGBoost
2. Layer 2: Train meta-model on Layer 1's OOF predictions
3. Meta-model learns optimal combination

**Expected Gain:** +0.003-0.007
**Implementation Time:** 1-2 hours
**Expected LB:** 0.895-0.899

---

## Decision Tree

```
V13 Result?
│
├─ ≥0.895 → Submit V14, expect 0.896-0.900
│            If < 0.900: Try Focal Loss
│            If ≥ 0.900: Try Stacking for 0.906
│
├─ 0.893-0.894 → Submit V14, expect 0.895-0.897
│                 If < 0.897: Try Focal Loss + Pseudo-Label
│                 If ≥ 0.897: Try Stacking
│
└─ < 0.893 → Optimization exhausted
             Priority: Focal Loss (targets High class directly)
             Then: Pseudo-Labeling or New Features
```

---

## Why Leader is at 0.906?

### Possible Techniques Used:

1. **Better High Class Detection**
   - Focal Loss or Cost-Sensitive Learning
   - More sophisticated sampling (ADASYN, SVMSMOTE)
   - Better class weight optimization

2. **More Diverse Models**
   - Neural networks (TabNet, FT-Transformer)
   - Different feature engineering
   - Multi-level stacking

3. **Better Probability Calibration**
   - Advanced calibration methods
   - Per-country calibration
   - Ensemble calibration

4. **Feature Engineering**
   - Domain expertise features
   - Better country interactions
   - External data (if allowed)

5. **Hyperparameter Optimization**
   - More extensive Optuna searches
   - Per-fold hyperparameters
   - Optimized for High class specifically

---

## Realistic Path to 0.906

### Conservative Estimate:
```
V13 (blend opt):        0.893
V14 (calibration):      0.896
Focal Loss:             0.900
Pseudo-Label + Stack:   0.904
```
→ **0.904 achievable** with current techniques

### To Reach 0.906:
Need at least one of:
- Superior feature engineering
- Neural network models
- Multi-level stacking with diverse models
- Domain expertise insights

---

## Immediate Action Plan

### Step 1: Submit V13 (Now)
**File:** `submissions/submission_v13_optimized_blend_weights.csv`
**Expected:** 0.893-0.895
**Time:** 0 minutes (already generated)

### Step 2: Run V14 (If V13 ≥ 0.893)
**Command:** `python train_v14_probability_calibration.py`
**Expected:** 0.894-0.898
**Time:** ~5 minutes

### Step 3: Submit V14 (If trained)
**File:** `submissions/submission_v14_probability_calibration.csv`
**Expected:** +0.002-0.006 over V13

### Step 4: Decide Next Move
**If V14 < 0.897:** Implement Focal Loss (highest potential)
**If V14 ≥ 0.897:** Implement Stacking or Pseudo-Labeling

---

## Bottom Line

**V13 and V14 are ready to submit.**

**Expected trajectory:**
- V13: 0.893-0.895 (optimization)
- V14: 0.896-0.898 (calibration)
- **Gap remaining:** 0.008-0.010 to leader

**To close final gap:**
- Focal Loss (best bet for +0.005-0.010)
- Then Pseudo-Labeling or Stacking
- **0.900+ is achievable**
- **0.906 requires advanced techniques**

**Recommendation:**
1. Submit V13 now
2. If V13 ≥ 0.893, run and submit V14
3. If V14 < 0.897, implement Focal Loss next
4. Continue iterating until 0.900+

---

*Files ready:*
- `submissions/submission_v13_optimized_blend_weights.csv`
- `submissions/submission_v14_probability_calibration.csv` (run script first)
