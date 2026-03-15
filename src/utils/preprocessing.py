import numpy as np
import pandas as pd


# ordinal encoding maps
_COLLEGE_TIER_MAP = {'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1}
_RANK_BAND_MAP    = {'Top 100': 4, '100-300': 3, '300+': 2}

# columns to one-hot encode
_ONEHOT_COLS = ['country', 'specialization', 'industry']

# numerical columns to standardize
_NUM_COLS = [
    'cgpa', 'backlogs', 'internship_count',
    'aptitude_score', 'communication_score', 'internship_quality_score',
]


def load_and_preprocess(csv_path):
    """
    Load the student placement CSV and return train/val/test splits.

    Encoding:
      - college_tier         : ordinal (Tier 1=3, Tier 2=2, Tier 3=1)
      - university_ranking_band : ordinal (Top 100=4, 100-300=3, 300+=2)
      - country, specialization, industry : one-hot
      - numerical columns    : standardized (zero mean, unit variance)
      - placement_status     : Placed=1, Not Placed=0

    Returns:
        X_train, X_val, X_test : np.ndarray, float64
        y_train, y_val, y_test : np.ndarray, shape (N, 1), float64
        feature_names          : list of str
    """
    df = pd.read_csv(csv_path)

    # --- target ---
    y = (df['placement_status'] == 'Placed').astype(float).values.reshape(-1, 1)

    # --- ordinal encoding ---
    df['college_tier']            = df['college_tier'].map(_COLLEGE_TIER_MAP)
    df['university_ranking_band'] = df['university_ranking_band'].map(_RANK_BAND_MAP)

    # --- one-hot encoding ---
    df = pd.get_dummies(df, columns=_ONEHOT_COLS, dtype=float)

    # --- drop target column ---
    df = df.drop(columns=['placement_status'])

    # --- standardize numerical columns (fit on train split later) ---
    # collect all feature columns
    feature_cols = df.columns.tolist()

    X = df.values.astype(float)

    # --- split: 70 / 15 / 15 ---
    n      = len(X)
    idx    = np.random.default_rng(42).permutation(n)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # --- standardize numerical cols (fit only on train) ---
    num_indices = [feature_cols.index(c) for c in _NUM_COLS if c in feature_cols]
    mean = X_train[:, num_indices].mean(axis=0)
    std  = X_train[:, num_indices].std(axis=0) + 1e-8

    X_train[:, num_indices] = (X_train[:, num_indices] - mean) / std
    X_val[:,   num_indices] = (X_val[:,   num_indices] - mean) / std
    X_test[:,  num_indices] = (X_test[:,  num_indices] - mean) / std

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def accuracy(y_pred_data, y_true):
    """Binary classification accuracy. y_pred_data is a numpy array of probabilities."""
    preds = (y_pred_data >= 0.5).astype(float)
    return float((preds == y_true).mean())
