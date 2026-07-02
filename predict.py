"""Prediction and input-validation helpers for the House Price app."""
import json
import os

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "house_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_columns.json")

VALID_LOCATIONS = ("Urban", "Semi Urban", "Rural")
AREA_MIN = 500
AREA_MAX = 10000
BEDROOMS_MIN = 1
BEDROOMS_MAX = 10
BATHROOMS_MIN = 1
BATHROOMS_MAX = 6
GARAGE_MIN = 0
GARAGE_MAX = 4
AGE_MIN = 0
AGE_MAX = 100


def load_model():
    return joblib.load(MODEL_PATH)


def load_scaler():
    return joblib.load(SCALER_PATH)


def load_feature_columns():
    with open(FEATURES_PATH, encoding="utf-8") as f:
        return json.load(f)


def validate_inputs(area, bedrooms, bathrooms, garage, age, location):
    """
    Validate user inputs before prediction.

    Raises
    ------
    ValueError
        If any input is outside allowed bounds or has an invalid type/value.
    """
    errors = []

    numeric_checks = [
        ("Area", area, AREA_MIN, AREA_MAX),
        ("Bedrooms", bedrooms, BEDROOMS_MIN, BEDROOMS_MAX),
        ("Bathrooms", bathrooms, BATHROOMS_MIN, BATHROOMS_MAX),
        ("Garage", garage, GARAGE_MIN, GARAGE_MAX),
        ("House Age", age, AGE_MIN, AGE_MAX),
    ]
    for label, value, low, high in numeric_checks:
        if not isinstance(value, (int, float)):
            errors.append(f"{label} must be a number.")
            continue
        if value < low:
            errors.append(f"{label} cannot be less than {low}.")
        elif value > high:
            errors.append(f"{label} cannot be greater than {high}.")

    if location not in VALID_LOCATIONS:
        errors.append(
            f"Location must be one of {', '.join(VALID_LOCATIONS)} (got '{location}')."
        )

    if errors:
        raise ValueError(" ".join(errors))


def preprocess(area, bedrooms, bathrooms, garage, age, location, scaler=None, feature_cols=None):
    """Transform raw inputs into the feature matrix used at training time."""
    validate_inputs(area, bedrooms, bathrooms, garage, age, location)

    scaler = scaler if scaler is not None else load_scaler()
    feature_cols = feature_cols if feature_cols is not None else load_feature_columns()

    scaled = scaler.transform(pd.DataFrame(
        [[area, bathrooms, age]],
        columns=["Area", "Bathrooms", "Age"],
    ))

    row = {col: 0 for col in feature_cols}
    row["Area"] = scaled[0, 0]
    row["Bedrooms"] = bedrooms
    row["Bathrooms"] = scaled[0, 1]
    row["Garage"] = garage
    row["Age"] = scaled[0, 2]
    row["Floors"] = 1
    row["Year Built"] = 2026 - age

    location_col = f"Location_{location}"
    if location_col in row:
        row[location_col] = 1

    if "Condition_Good" in row:
        row["Condition_Good"] = 1

    return pd.DataFrame([row], columns=feature_cols)


def predict_price(
    area,
    bedrooms,
    bathrooms,
    garage,
    age,
    location,
    model=None,
    scaler=None,
    feature_cols=None,
):
    """Validate inputs, preprocess, and return the predicted house price."""
    features = preprocess(
        area, bedrooms, bathrooms, garage, age, location,
        scaler=scaler,
        feature_cols=feature_cols,
    )
    model = model if model is not None else load_model()
    return float(model.predict(features)[0])


def format_inr(amount):
    """Format a price in the Indian numbering system (e.g. ₹78,45,200)."""
    amount = int(round(amount))
    s = str(abs(amount))
    if len(s) <= 3:
        formatted = s
    else:
        formatted = s[-3:]
        s = s[:-3]
        while s:
            formatted = s[-2:] + "," + formatted
            s = s[:-2]
    prefix = "-" if amount < 0 else ""
    return f"{prefix}₹{formatted}"
