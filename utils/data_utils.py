import pandas as pd
import numpy as np

def load_csv(path):
    return pd.read_csv(path)

def remove_unused_column(df, column="Unnamed: 133"):
    if column in df.columns:
        df = df.drop(column, axis=1)
    return df

def detect_columns(df):
    target_candidates = ["disease", "prognosis"]
    target = None
    for col in df.columns:
        if col.lower() in target_candidates:
            target = col
            break
    if target is None:
        raise ValueError("Target column not found")
    features = [c for c in df.columns if c != target]
    return target, features

def add_noise(X, noise_level=0.1):
    X_noisy = X.copy()
    num_features = X.shape[1]

    for i in range(len(X)):
        flip_count = int(noise_level * num_features)

        if flip_count == 0:
            continue

        indices = np.random.choice(num_features, flip_count, replace=False)
        X_noisy[i, indices] = 1 - X_noisy[i, indices]

    return X_noisy