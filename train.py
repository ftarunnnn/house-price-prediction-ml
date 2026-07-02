import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# XGBoost import – optional, will raise informative error if not installed
try:
    from xgboost import XGBRegressor
except ImportError as e:
    XGBRegressor = None
    import warnings
    warnings.warn("xgboost is not installed. XGBRegressor will be unavailable.")

from src.preprocessing import clean_data
from src.feature_engineering import one_hot_encode_location
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os
from src.split_data import split_features_target, split_train_test


def load_and_prepare_data(csv_path: str = "data/house_prices.csv",
                          target_column: str = "Price",
                          location_column: str = "Location",
                          numeric_scaler: str = "standard"):
    """Load CSV, clean, engineer features, and split into train/test.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
    scaler : fitted scaler object (StandardScaler or MinMaxScaler) used for numeric features.
    """
    # Load raw data
    df_raw = pd.read_csv(csv_path)
    # Clean
    df_clean = clean_data(df_raw)
    # Feature engineering – one‑hot encode location then scale numeric columns
    df_fe = one_hot_encode_location(df_clean)

    # Scale numeric features and keep scaler for later reuse
    numeric_features = [col for col in ["Area", "Bathrooms", "Age"] if col in df_fe.columns]
    if numeric_features:
        if numeric_scaler == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        df_fe[numeric_features] = scaler.fit_transform(df_fe[numeric_features])
    else:
        scaler = None

    # Split X / y
    X, y = split_features_target(df_fe, target_column=target_column)
    # Train‑test split (default 80/20)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    return X_train, X_test, y_train, y_test, scaler


def train_models(X_train, y_train):
    """Train a collection of regression models.

    Returns a dictionary mapping model names to fitted model objects.
    """
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    }
    if XGBRegressor is not None:
        models["XGBRegressor"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
        )
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def evaluate_models(models: dict, X_test, y_test):
    """Evaluate multiple regression models.

    Returns a DataFrame with MAE, MSE, RMSE, and R² for each model.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import pandas as pd
    import numpy as np

    rows = []
    for name, model in models.items():
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        rows.append({"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})
    # Sort by MAE (lower is better)
    return pd.DataFrame(rows).sort_values(by="MAE")


if __name__ == "__main__":
    # End‑to‑end execution
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
    trained_models = train_models(X_train, y_train)
    performance = evaluate_models(trained_models, X_test, y_test)
    print("Model performance (sorted by MAE):")
    print(performance.to_string(index=False))

    # Identify best model (lowest MAE)
    best_model_name = performance.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save best model
    joblib.dump(best_model, os.path.join("models", "house_model.pkl"))
    print(f"Best model '{best_model_name}' saved to models/house_model.pkl")

    # Save scaler if available
    if scaler is not None:
        joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
        print("Scaler saved to models/scaler.pkl")
    else:
        print("No scaler to save.")

    # Save feature column names so the inference pipeline stays in sync
    import json
    feature_cols_path = os.path.join("models", "feature_columns.json")
    with open(feature_cols_path, "w", encoding="utf-8") as f:
        json.dump(list(X_train.columns), f)
    print(f"Feature columns ({len(X_train.columns)}) saved to {feature_cols_path}")
