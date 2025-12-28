---

## Claude Code Prompt (Phase 1 – Baseline)

> You are working inside this project:
>
> `~/Ml/Financial-Health-Prediction-Challenge`
>
> Follow the plan strictly. Do **not** jump ahead.

### OBJECTIVE

Build a **strong baseline** for the **data.org Financial Health Prediction Challenge**.
Metric = **macro F1**.
Target variable = `Target` ∈ {Low, Medium, High}.

---

### RULES

* No AutoML
* No deep learning
* Use **only provided data**
* Use **StratifiedKFold (5 folds)**
* Fix random seed
* Optimize **macro F1**, not accuracy
* STOP after baseline evaluation

---

### TASKS (DO THESE IN ORDER)

#### 1️⃣ Data Loading

* Load:

  * `data/raw/Train.csv`
  * `data/raw/Test.csv`
* Separate:

  * `X`
  * `y = Target`
* Drop ID from features (keep for submission)
* Print:

  * dataset shape
  * missing values summary
  * class distribution

---

#### 2️⃣ Feature Handling (Minimal & Safe)

* Identify:

  * categorical columns
  * numerical columns
* Do **NOT** one-hot everything
* Prepare pipeline support for:

  * CatBoost (native categoricals)
  * LightGBM/XGBoost (encoded categoricals later)

Save any processed outputs into:

```
data/processed/
```

---

#### 3️⃣ Baseline Model (ONLY ONE MODEL)

Train **CatBoostClassifier** with:

* `loss_function="MultiClass"`
* `eval_metric="TotalF1"`
* class weights enabled
* early stopping
* reasonable defaults (no heavy tuning)

Use **5-fold Stratified CV**
Log:

* F1 per fold
* Mean CV macro F1

---

### STOP CONDITION

After reporting:

* Mean CV macro F1
* Std
* Confusion matrix (aggregated)

**STOP EXECUTION. DO NOT ENSEMBLE. DO NOT SUBMIT.**

---

### OUTPUT

* Clean, runnable code
* No experiments beyond CatBoost
* Clear printed CV results

---

