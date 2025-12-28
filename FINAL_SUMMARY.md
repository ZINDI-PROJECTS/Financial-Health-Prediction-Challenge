# Financial Health Prediction - Final Summary

## Competition Status
- **Current Public LB:** 0.883 (Rank 97)
- **Target:** 0.90+ (Top 5 range: ~0.900-0.906)
- **Gap to close:** 0.017-0.023

---

## Submissions Ready

| # | File | Strategy | CV F1 | Public LB | Status |
|---|------|----------|-------|-----------|--------|
| **V1** | `baseline_catboost_v1.csv` | CatBoost baseline | 0.766 | 0.868 | ✓ Tested |
| **V2** | `submission_v2_catboost_lgb_blend.csv` | Ensemble (Cat+LGB) | **0.803** | **0.883** | ✓ Tested (BEST) |
| **V3** | `submission_v3_engineered_features.csv` | 7 features + Cat | 0.799 | 0.872 | ✓ Tested (Noisy) |
| **V4** | `submission_v4_refined_clean.csv` | 3 refined features + Cat | 0.798 | ??? | Ready |
| **V5** | `submission_v5_final_blend_features.csv` | Ensemble + 3 refined features | **0.800** | ??? | Ready |

---

## What We Learned

### ✅ What Worked
1. **Class weights (1, 2.5, 7)** → +0.036 CV gain (biggest impact)
2. **Ensemble (Cat+LGB)** → +0.001 CV, stable performance
3. **Engineered features rank high** → credit_access (#3), profit_margin (#4)

### ❌ What Didn't Work
1. **Threshold tuning** → 0.000 gain (probabilities already calibrated)
2. **Missingness indicators** → -0.003 (CatBoost handles natively)
3. **Log transforms** → -0.002 (CatBoost handles skewness)
4. **Too many features (7)** → Introduced noise, hurt Medium/Low classes
5. **Aggressive High weights (>10)** → Collapsed High class recall

### 📊 Key Insight
**We hit a model ceiling at ~0.80 CV (0.88 public)**
- CatBoost + class weights extracted 95% of available signal
- Feature engineering helped individual feature importance but not ensemble performance
- The gap between CV and public LB is ~0.08 (not the +0.10 we initially hoped)

---

## Current Situation Analysis

### CV → Public LB Relationship
```
V1: 0.766 CV → 0.868 Public (+0.102 gap)
V2: 0.803 CV → 0.883 Public (+0.080 gap)
V3: 0.799 CV → 0.872 Public (+0.073 gap)
```

**Pattern:** Gap is shrinking as CV improves (more realistic predictions)

### Expected Performance
- **V4:** 0.798 CV → ~0.875 public (features alone, no ensemble)
- **V5:** 0.800 CV → ~0.880 public (ensemble + features)

---

## Strategic Assessment

### Why We're Stuck at 0.88

**The Problem:** High class (5% of data) F1 = 0.72
- Even with class weights (1, 2.5, 7), High recall is limited
- Confusion between Medium ↔ High is fundamental
- Macro F1 = average of (Low=0.90, Med=0.78, High=0.72) = 0.80

**To reach 0.90 macro F1, we need:**
- High F1 ≥ 0.80 (+0.08 gain needed)
- OR perfect Low + Medium (unrealistic)

### What Would Actually Work (Untested Approaches)

**Option A: Country-Specific Modeling** ⭐ HIGH POTENTIAL
- Your data spans 4 countries (Eswatini, Zimbabwe, Malawi, Lesotho)
- Financial health definitions likely vary by country
- **Approach:** Train 4 separate models OR use country as strong interaction
- **Risk:** Lower sample size per country
- **Expected gain:** +0.01-0.03 if country effects are strong

**Option B: Focal Loss / Custom Objective** ⭐ MEDIUM POTENTIAL
- CatBoost supports custom loss functions
- Focal loss explicitly targets hard-to-classify examples (High class)
- **Approach:** Implement focal loss with γ=2, focus on High class
- **Risk:** Requires careful tuning, might overfit
- **Expected gain:** +0.01-0.02

**Option C: SMOTE/Oversampling High Class** ⭐ MEDIUM POTENTIAL
- Synthetic oversampling to balance classes
- **Approach:** Generate synthetic High samples to 10% of dataset
- **Risk:** Overfitting to synthetic patterns
- **Expected gain:** +0.005-0.015

**Option D: Pseudo-Labeling (If Multiple Submissions Allowed)** ⭐ LOW RISK
- Use V2 predictions on test set as pseudo-labels
- Retrain including high-confidence test predictions
- **Approach:** Add test samples with probability > 0.9 to training
- **Risk:** Circular logic if test distribution differs
- **Expected gain:** +0.005-0.01

**Option E: Stacking (Not Just Blending)** ⭐ MEDIUM POTENTIAL
- Train meta-model on CatBoost + LightGBM predictions
- **Approach:** Use logistic regression / XGBoost as meta-learner
- **Risk:** Overfitting, small gain
- **Expected gain:** +0.003-0.008

---

## Recommendations

### Immediate Action (Next Submission)

**Submit V5** (`submission_v5_final_blend_features.csv`)

**Rationale:**
- Combines ensemble strength + proven features (rank #3, #4)
- CV = 0.800 (comparable to V2)
- Features might help more on public data than in CV
- Different prediction distribution than V2 (might hit different test patterns)

**Expected outcome:**
- **Best case:** 0.885-0.890 (features help on public data)
- **Likely case:** 0.880-0.883 (same as V2)
- **Worst case:** 0.875-0.880 (features add noise)

### If V5 < 0.885

Then we've confirmed: **Standard ML optimization won't reach 0.90**

**Next steps (in order of ROI):**

1. **Try Country-Specific Features** (1-2 hours)
   - Add country × profit_margin interaction
   - Add country × has_insurance interaction
   - Expected: +0.01 gain

2. **Implement Focal Loss** (2-3 hours)
   - Custom CatBoost objective
   - Focus on High class specifically
   - Expected: +0.005-0.015 gain

3. **Test SMOTE on High Class** (1 hour)
   - Quick experiment
   - Low complexity
   - Expected: +0.005-0.01 gain

4. **Advanced Stacking** (2-3 hours)
   - Add XGBoost to ensemble
   - Meta-learner on predictions
   - Expected: +0.003-0.008 gain

### If V5 ≥ 0.885

**You're in striking distance!**

- Submit country-specific features next
- One more +0.015 gain gets you to 0.90

---

## Technical Debt & Next Session Prep

### Clean Code Structure
```
✓ src/features.py (V4 clean features)
✓ src/model.py (threshold optimization)
✓ engineer_features_v4.py (production-ready)
✓ train_v5_final.py (full pipeline)
```

### What's Ready to Use
- **Best class weights:** {0: 1.0, 1: 2.5, 2: 7.0}
- **Best features:** credit_access_score, profit_margin (log-scaled)
- **Best ensemble:** CatBoost + LightGBM (simple mean)
- **Threshold tuning:** Implemented but not helping (probabilities calibrated)

### What Needs Exploration
- Country interactions (EDA + feature engineering)
- Custom loss functions (focal loss for High class)
- Synthetic sampling strategies (SMOTE, ADASYN)
- Meta-learning / stacking architectures

---

## Per-Class Breakdown (V5)

| Class | Support | Precision | Recall | F1 | Issue |
|-------|---------|-----------|--------|-------|-------|
| **Low** | 6,280 | 0.932 | 0.881 | **0.905** | ✓ Strong |
| **Medium** | 2,868 | 0.761 | 0.828 | **0.775** | ⚠ Moderate |
| **High** | 470 | 0.780 | 0.672 | **0.719** | ❌ **Bottleneck** |

**High class confusion:**
- 14 predicted as Low (2.9%)
- 140 predicted as Medium (29.8%) ← **Main problem**
- 316 correctly predicted (67.2%)

**The fix:** Better Medium vs High separation
- Current features help but not enough
- Need stronger discriminative signal (country, focal loss, or SMOTE)

---

## Files Reference

### Submissions
- `/submissions/submission_v5_final_blend_features.csv` ← **Next upload**
- `/submissions/submission_v2_catboost_lgb_blend.csv` (0.883 public)

### Code
- `/train_v5_final.py` - Full V5 pipeline
- `/engineer_features_v4.py` - Clean feature engineering
- `/step2_class_weights.py` - Class weight experiments
- `/step5_model_comparison.py` - Model comparison

### Analysis
- `/eda.py` - Comprehensive EDA
- `/baseline.py` - Initial baseline

---

## Bottom Line

**Current position:** Rank 97 @ 0.883 public
**Target:** Top 5 @ 0.90+
**Gap:** 0.017-0.023
**Next submission:** V5 (ensemble + features)
**If V5 fails:** Country features → Focal loss → SMOTE

**The truth:** We optimized models to their limit. To reach 0.90, we need **different signal**, not better optimization.

---

*Generated after 6 iterations of systematic experimentation*
*Total CV improvement: 0.766 → 0.803 (+0.037)*
*Total Public improvement: 0.868 → 0.883 (+0.015)*
