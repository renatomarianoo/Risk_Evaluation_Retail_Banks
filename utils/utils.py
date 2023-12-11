"""
Created on 26/11/2023

@author: renato.mariano
"""

from time import time
import pandas as pd
from tqdm import tqdm


def measure_time_elapsed(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.3f} seconds\n")
        return result

    return wrapper


@measure_time_elapsed
def load_df(file_path, encoding=None, n_skip=1):
    df = pd.read_csv(file_path, encoding=encoding, skiprows=lambda x: x % n_skip)
    print(f"The shape of the data is: {df.shape}")
    return df


def extract_features(df):
    """Extracts numeric, categorical, binary, and high-cardinality features from a DataFrame."""
    num_feats = df.select_dtypes(include="number").columns
    cat_feats = df.select_dtypes(include="object").columns

    binary_feats = (df[cat_feats].nunique() <= 2).index[df[cat_feats].nunique() <= 2]
    highcard_feats = (df[cat_feats].nunique() > 2).index[df[cat_feats].nunique() > 2]

    return num_feats, cat_feats, binary_feats, highcard_feats
