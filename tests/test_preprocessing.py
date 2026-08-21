import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import add_derived_features, remove_capped_target, split_and_scale

def test_add_derived_features_creates_ratio_column():
    fake_df = pd.DataFrame({
        'AveRooms': [6.0, 4.0],
        'AveBedrms': [1.0, 2.0],
    })
    result = add_derived_features(fake_df)
    assert 'bedroom_ratio' in result.columns
    assert result['bedroom_ratio'].iloc[0] == 1.0 / 6.0

def test_remove_capped_target_filters_correctly():
    fake_df = pd.DataFrame({
        'MedHouseVal': [1.0, 3.0, 5.0, 5.5],
    })
    result = remove_capped_target(fake_df, cap=5.0)
    assert len(result) == 2
    assert result['MedHouseVal'].max() < 5.0
    
def test_split_and_scale_row_counts_match():
    fake_df = pd.DataFrame({
        'AveRooms': [3.0, 4.0, 5.0, 6.0],
        'AveBedrms': [1.0, 2.0, 3.0, 4.0],
        'MedHouseVal': [1.0, 3.0, 5.0, 5.5],
    })
    X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test, scaler = split_and_scale(
        fake_df, target_col='MedHouseVal', test_size=0.5, random_state=42
    )
    assert len(X_train) + len(X_test) == len(fake_df)

def test_split_and_scale_scaled_mean_near_zero():
    fake_df = pd.DataFrame({
        'AveRooms': [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        'AveBedrms': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'MedHouseVal': [1.0, 3.0, 5.0, 5.5, 2.0, 4.0],
    })
    X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test, scaler = split_and_scale(
        fake_df, target_col='MedHouseVal', test_size=0.33, random_state=42
    )
    assert abs(X_train_scaled.mean()) < 0.01
    
def test_add_derived_features_does_not_mutate_original():
    fake_df = pd.DataFrame({
        'AveRooms': [3.0, 4.0],
        'AveBedrms': [1.0, 2.0],
    })
    original_columns = list(fake_df.columns)
    add_derived_features(fake_df)
    assert list(fake_df.columns) == original_columns