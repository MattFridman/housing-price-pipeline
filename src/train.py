"""
train.py

Functions for fitting sklearn regressors and evaluating their
predictive performance using MAE, RMSE, and R².
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_model(model, X_train, y_train):
    """
    Fit a given sklearn regressor on training data and return it.

    Accepts any sklearn-compatible estimator, so the same function can
    be reused across different model types (e.g. LinearRegression,
    RandomForestRegressor) without duplicating fitting logic.

    Parameters
    ----------
    model : sklearn estimator
        A regressor implementing `.fit(X, y)` and `.predict(X)`.
    X_train : pandas.DataFrame or numpy.ndarray
        Training feature data.
    y_train : pandas.Series
        Training target values.

    Returns
    -------
    sklearn estimator
        The same model instance, now fitted on X_train/y_train.
    """
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Return a dict of MAE, RMSE, and R² for a fitted model on test data.

    MAE, RMSE, and R² each summarize a fitted model's prediction error
    differently: MAE weighs every error linearly, RMSE penalizes large
    errors more heavily (since errors are squared before averaging), and
    R² expresses the proportion of the target's variance the model
    explains. Reporting all three avoids relying on any single metric
    that could look better or worse than the model's real-world behavior.

    Parameters
    ----------
    model : sklearn estimator
        A fitted regressor implementing `.predict(X)`.
    X_test : pandas.DataFrame or numpy.ndarray
        Test feature data, held out from training.
    y_test : pandas.Series
        True target values corresponding to X_test.

    Returns
    -------
    dict
        Dictionary with keys 'mae', 'rmse', and 'r2'.
    """
    predictions = model.predict(X_test)
    return {
        'mae': mean_absolute_error(y_test, predictions),
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'r2': r2_score(y_test, predictions),
    }