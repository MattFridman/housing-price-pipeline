"""
preprocessing.py

Functions for cleaning and preparing the California Housing dataset for
model training: feature engineering, cap-row filtering, and the
train/test split + scaling step (scaling fit on train only, to avoid
data leakage).
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import TARGET_COL, CAP_THRESHOLD, TEST_SIZE, RANDOM_STATE

def add_derived_features(df):
    """
    Add engineered features to the housing DataFrame.
    
    The California Housing dataset includes information describing the
    average number of bedrooms and separately, the average number of 
    rooms. Combining these both into a ratio makes it easier to read
    and understand the data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing the AveBedrms and AveRooms columns.
        
    Returns
    -------
    pandas.DataFrame
        A copy of the original dataframe with the additional bedroom_ratio 
        column.
    """
    df = df.copy()  # don't mutate the caller's DataFrame
    df['bedroom_ratio'] = df['AveBedrms'] / df['AveRooms']
    return df

def remove_capped_target(df, cap=CAP_THRESHOLD, target_col=TARGET_COL):
    """
    Remove rows where the target variable has hit its artificial upper cap.

    The California Housing dataset's target was capped at during the
    collection process. Any data points at this cap are likely to provide
    unusually results that may negatively impact the model's predictive
    capabilities for the rest of target values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing the target column.
    cap : float, default=CAP_THRESHOLD
        Threshold at or above which rows are considered capped and removed.
    target_col : str, default=TARGET_COL
        Name of the target column to check against the cap.

    Returns
    -------
    pandas.DataFrame
        A filtered copy of df with capped rows removed.
    """
    return df[df[target_col] < cap]

def split_and_scale(df, target_col=TARGET_COL, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Split data into train/test sets, then scale features using a scaler
    fit only on the training set.

    Fitting the scaler on the full dataset before splitting would leak
    information about the test set's distribution into training, producing
    an overly optimistic evaluation. Fitting only on X_train and reusing
    that same fitted scaler to transform X_test keeps the test set from its
    bias.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing features and the target column.
    target_col : str, default=TARGET_COL
        Name of the column to use as the prediction target.
    test_size : float, default=TEST_SIZE
        Proportion of the data to hold out for testing.
    random_state : int, default=RANDOM_STATE
        Seed for reproducible splitting.

    Returns
    -------
    X_train_scaled, X_test_scaled : numpy.ndarray
        Scaled feature arrays for training and testing.
    y_train, y_test : pandas.Series
        Target values corresponding to the train/test splits.
    X_train, X_test : pandas.DataFrame
        Unscaled feature DataFrames, returned alongside the scaled
        versions for inspection or use with models that don't require
        scaling.
    scaler : sklearn.preprocessing.StandardScaler
        The fitted scaler, returned so new data (outside this pipeline)
        can be transformed consistently with the training data.
    """
    X = df.drop(columns=target_col)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test, scaler