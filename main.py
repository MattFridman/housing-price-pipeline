"""
main.py

End-to-end entry point: load data, preprocess, train both models,
print evaluation metrics.
"""
from src.data_loader import load_housing_data
from src.preprocessing import add_derived_features, remove_capped_target, split_and_scale
from src.train import train_model, evaluate_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    df = load_housing_data()
    df = add_derived_features(df)
    df = remove_capped_target(df)
    X_train, X_test, y_train, y_test, X_train_raw, X_test_raw, scaler = split_and_scale(df)

    lr = train_model(LinearRegression(), X_train, y_train)
    print("Linear Regression:", evaluate_model(lr, X_test, y_test))

    rf = train_model(RandomForestRegressor(random_state=42), X_train, y_train)
    print("Random Forest:", evaluate_model(rf, X_test, y_test))