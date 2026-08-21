"""
data_loader.py

Loads the California Housing dataset. Checks for a locally cached copy
before fetching from sklearn, and saves a fresh fetch to disk for reuse
on future runs.
"""

import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
from src.config import RAW_DATA_PATH

def load_housing_data(save_path=RAW_DATA_PATH):
    """
    Load the California Housing dataset as a DataFrame.

    If save_path is given and a file already exists there, load from disk
    instead of re-fetching. Otherwise, fetch from sklearn and optionally
    save a copy to save_path for future runs.

    Parameters
    ----------
    save_path : str or None
        File path to cache the raw data as a CSV.

    Returns
    -------
    pandas.DataFrame
        The housing dataset, features and target combined.
    """
    if save_path and os.path.exists(save_path):
        return pd.read_csv(save_path)

    data = fetch_california_housing(as_frame=True)
    df = data.frame

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

    return df