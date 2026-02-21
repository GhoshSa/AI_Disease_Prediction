import pandas as pd

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