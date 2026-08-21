"""
evaluate.py

Visualization functions for inspecting a fitted regression model's
prediction errors: predicted-vs-actual and residual plots.
"""

import matplotlib.pyplot as plt


def plot_predicted_vs_actual(y_test, predictions, title='Predicted vs Actual'):
    """
    Plot predicted values against actual values, with a reference
    diagonal representing perfect prediction.

    Points closer to the diagonal indicate more accurate predictions;
    systematic drift away from it (e.g., all points below the line at
    high values) reveals bias that a single aggregate metric like R²
    can hide.

    Parameters
    ----------
    y_test : pandas.Series
        True target values.
    predictions : numpy.ndarray
        Model-predicted values corresponding to y_test.
    title : str, default='Predicted vs Actual'
        Title for the plot.

    Returns
    -------
    None
        Displays the plot; does not return a value.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, predictions, alpha=0.3)
    lims = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    ax.plot(lims, lims, 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    ax.legend()

def plot_residuals(y_test, predictions, title='Residual Plot'):
    """
    Plot model residuals (actual - predicted) against predicted values.

    A well-fit model shows residuals scattered randomly around zero
    with constant spread. A funnel shape (widening spread as predicted
    values increase) or a visible curve indicates the model's error
    is not uniform across the range of predictions.

    Parameters
    ----------
    y_test : pandas.Series
        True target values.
    predictions : numpy.ndarray
        Model-predicted values corresponding to y_test.
    title : str, default='Residual Plot'
        Title for the plot.

    Returns
    -------
    None
        Displays the plot; does not return a value.
    """
    residuals = y_test - predictions
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(predictions, residuals, alpha=0.3)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Residual (Actual - Predicted)')
    ax.set_title(title)