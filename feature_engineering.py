import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def engineer_features(df: pd.DataFrame, scaler_type: str = "standard") -> pd.DataFrame:
    """Perform feature engineering on the cleaned housing dataframe.

    * One‑hot encode the ``Location`` column (expected values: "Urban", "Semi Urban", "Rural").
    * Scale numeric features ``Area``, ``Bathrooms``, and ``Age`` using either
      ``StandardScaler`` (zero‑mean, unit‑variance) or ``MinMaxScaler``
      (scaled to [0, 1]).

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe returned from ``clean_data``.
    scaler_type: str, optional
        "standard" for ``StandardScaler`` or "minmax" for ``MinMaxScaler``.
        Default is "standard".

    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features ready for model training.
    """
    # One‑hot encode Location if present
    if "Location" in df.columns:
        location_dummies = pd.get_dummies(df["Location"], prefix="Location")
        df = df.drop(columns=["Location"], errors="ignore")
        df = pd.concat([df, location_dummies], axis=1)

    # Identify numeric columns to scale (Area, Bathrooms, Age if they exist)
    numeric_features = [col for col in ["Area", "Bathrooms", "Age"] if col in df.columns]
    if numeric_features:
        scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])

    return df


def one_hot_encode_location(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the ``Location`` column and any other remaining categorical columns.

    This is a lightweight helper used by ``train.py`` when the scaler is
    managed separately.  For the combined operation use ``engineer_features``.
    """
    if "Location" in df.columns:
        location_dummies = pd.get_dummies(df["Location"], prefix="Location")
        df = df.drop(columns=["Location"], errors="ignore")
        df = pd.concat([df, location_dummies], axis=1)

    # Encode any remaining categorical / object columns (e.g. Condition)
    remaining_cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    if remaining_cat_cols:
        df = pd.get_dummies(df, columns=remaining_cat_cols, drop_first=False)

    # Ensure all boolean dummy columns are int (0/1) for sklearn compatibility
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)

    return df

