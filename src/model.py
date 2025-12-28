"""
Model training and evaluation utilities
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from catboost import CatBoostClassifier
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def optimize_thresholds(y_true, y_proba, n_classes=3):
    """
    Find optimal decision thresholds to maximize macro-F1

    Parameters:
    -----------
    y_true : array
        True labels (0, 1, 2)
    y_proba : array
        Predicted probabilities, shape (n_samples, n_classes)
    n_classes : int
        Number of classes

    Returns:
    --------
    thresholds : array
        Optimal thresholds for each class
    """
    def objective(thresholds):
        # Convert probabilities to predictions using thresholds
        predictions = np.argmax(y_proba - thresholds, axis=1)
        # Return negative macro-F1 (we minimize)
        return -f1_score(y_true, predictions, average='macro')

    # Initial thresholds (equal probability)
    init_thresholds = np.zeros(n_classes)

    # Optimize
    result = minimize(
        objective,
        init_thresholds,
        method='Nelder-Mead',
        options={'maxiter': 500}
    )

    return result.x


def predict_with_thresholds(y_proba, thresholds):
    """
    Apply learned thresholds to probability predictions

    Parameters:
    -----------
    y_proba : array
        Predicted probabilities, shape (n_samples, n_classes)
    thresholds : array
        Thresholds for each class

    Returns:
    --------
    predictions : array
        Class predictions
    """
    return np.argmax(y_proba - thresholds, axis=1)


def train_catboost_cv(X, y, categorical_cols, n_folds=5, class_weights=None,
                      iterations=1000, learning_rate=0.05, depth=6, random_state=42):
    """
    Train CatBoost with cross-validation and threshold tuning

    Parameters:
    -----------
    X : DataFrame
        Features
    y : Series
        Target (encoded as 0, 1, 2)
    categorical_cols : list
        List of categorical column names
    n_folds : int
        Number of CV folds
    class_weights : dict, optional
        Manual class weights {0: w0, 1: w1, 2: w2}

    Returns:
    --------
    results : dict
        CV results including F1 scores, thresholds, predictions
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_results = []
    all_y_true = []
    all_y_proba = []
    all_y_pred_raw = []
    all_y_pred_tuned = []

    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION: {n_folds}-Fold Stratified")
    print(f"{'='*80}")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold_idx}/{n_folds}")
        print("-" * 40)

        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Initialize model
        model_params = {
            'loss_function': 'MultiClass',
            'eval_metric': 'TotalF1',
            'iterations': iterations,
            'learning_rate': learning_rate,
            'depth': depth,
            'random_state': random_state,
            'verbose': False,
            'early_stopping_rounds': 50
        }

        # Add class weights
        if class_weights is not None:
            model_params['class_weights'] = class_weights
        else:
            model_params['auto_class_weights'] = 'Balanced'

        model = CatBoostClassifier(**model_params)

        # Train
        model.fit(
            X_train_fold,
            y_train_fold,
            cat_features=categorical_cols,
            eval_set=(X_val_fold, y_val_fold),
            verbose=False
        )

        # Predict probabilities
        y_proba = model.predict_proba(X_val_fold)

        # Raw predictions (argmax)
        y_pred_raw = np.argmax(y_proba, axis=1)

        # Optimize thresholds on validation fold
        thresholds = optimize_thresholds(y_val_fold.values, y_proba)

        # Apply tuned thresholds
        y_pred_tuned = predict_with_thresholds(y_proba, thresholds)

        # Calculate F1 scores
        f1_raw = f1_score(y_val_fold, y_pred_raw, average='macro')
        f1_tuned = f1_score(y_val_fold, y_pred_tuned, average='macro')

        # Per-class F1
        f1_per_class_raw = f1_score(y_val_fold, y_pred_raw, average=None)
        f1_per_class_tuned = f1_score(y_val_fold, y_pred_tuned, average=None)

        print(f"  Macro-F1 (raw):   {f1_raw:.6f}")
        print(f"  Macro-F1 (tuned): {f1_tuned:.6f}")
        print(f"  Improvement:      {f1_tuned - f1_raw:+.6f}")
        print(f"  Thresholds:       {thresholds}")
        print(f"  F1 per class (tuned): Low={f1_per_class_tuned[0]:.4f}, "
              f"Med={f1_per_class_tuned[1]:.4f}, High={f1_per_class_tuned[2]:.4f}")

        # Store results
        fold_results.append({
            'fold': fold_idx,
            'f1_raw': f1_raw,
            'f1_tuned': f1_tuned,
            'thresholds': thresholds,
            'f1_per_class_raw': f1_per_class_raw,
            'f1_per_class_tuned': f1_per_class_tuned,
            'model': model
        })

        all_y_true.extend(y_val_fold.values)
        all_y_proba.append(y_proba)
        all_y_pred_raw.extend(y_pred_raw)
        all_y_pred_tuned.extend(y_pred_tuned)

    # Aggregate results
    all_y_proba = np.vstack(all_y_proba)
    all_y_true = np.array(all_y_true)
    all_y_pred_raw = np.array(all_y_pred_raw)
    all_y_pred_tuned = np.array(all_y_pred_tuned)

    # Global threshold optimization
    global_thresholds = optimize_thresholds(all_y_true, all_y_proba)
    y_pred_global = predict_with_thresholds(all_y_proba, global_thresholds)

    # Final metrics
    mean_f1_raw = np.mean([r['f1_raw'] for r in fold_results])
    std_f1_raw = np.std([r['f1_raw'] for r in fold_results])
    mean_f1_tuned = np.mean([r['f1_tuned'] for r in fold_results])
    std_f1_tuned = np.std([r['f1_tuned'] for r in fold_results])

    f1_global = f1_score(all_y_true, y_pred_global, average='macro')
    f1_per_class_global = f1_score(all_y_true, y_pred_global, average=None)

    # Confusion matrix
    cm = confusion_matrix(all_y_true, y_pred_global)

    print(f"\n{'='*80}")
    print(f"FINAL CV RESULTS")
    print(f"{'='*80}")
    print(f"\nMacro-F1 (raw):        {mean_f1_raw:.6f} ± {std_f1_raw:.6f}")
    print(f"Macro-F1 (tuned):      {mean_f1_tuned:.6f} ± {std_f1_tuned:.6f}")
    print(f"Macro-F1 (global):     {f1_global:.6f}")
    print(f"\nGlobal Thresholds:     {global_thresholds}")
    print(f"\nPer-class F1 (global):")
    print(f"  Low (0):    {f1_per_class_global[0]:.4f}")
    print(f"  Medium (1): {f1_per_class_global[1]:.4f}")
    print(f"  High (2):   {f1_per_class_global[2]:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"         Pred_Low  Pred_Med  Pred_High")
    print(f"True_Low    {cm[0,0]:5d}    {cm[0,1]:5d}     {cm[0,2]:5d}")
    print(f"True_Med    {cm[1,0]:5d}    {cm[1,1]:5d}     {cm[1,2]:5d}")
    print(f"True_High   {cm[2,0]:5d}    {cm[2,1]:5d}     {cm[2,2]:5d}")

    return {
        'fold_results': fold_results,
        'mean_f1_raw': mean_f1_raw,
        'std_f1_raw': std_f1_raw,
        'mean_f1_tuned': mean_f1_tuned,
        'std_f1_tuned': std_f1_tuned,
        'f1_global': f1_global,
        'global_thresholds': global_thresholds,
        'f1_per_class_global': f1_per_class_global,
        'confusion_matrix': cm,
        'all_y_true': all_y_true,
        'all_y_pred_tuned': y_pred_global
    }
