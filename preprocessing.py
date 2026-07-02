import pandas as pd
import numpy as np

# Columns that are identifiers and should be removed if present in the dataset
UNWANTED_COLUMNS = ["House ID", "Owner Name", "Address", "Transaction ID"]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the housing dataset.

    Steps performed (as requested in Phase 4):
    1️⃣ **Remove duplicate rows** – ``df.drop_duplicates(inplace=True)``
    2️⃣ **Drop unwanted identifier columns** – any of the columns listed in
       ``UNWANTED_COLUMNS`` that exist in the dataframe are removed.
    3️⃣ **Convert data types**
       * Object columns with a limited set of unique values are cast to
         ``category`` (helps memory & modelling).
       * Columns that look like numbers but are stored as strings are coerced to
         numeric types.
    4️⃣ **Handle missing values**
       * **Numeric columns** – filled with the column **mean** (you can replace
         with median by changing the implementation).
       * **Categorical columns** – filled with the column **mode** (most
         frequent value).
    5️⃣ **Ensure integer‑like columns are proper integer dtype** – if a numeric
       column contains only whole numbers it is cast to ``Int64`` (pandas nullable
       integer type).

    Parameters
    ----------
    df: pd.DataFrame
        Raw dataframe loaded from the CSV.

    Returns
    -------
    pd.DataFrame
        A cleaned dataframe ready for further preprocessing / model training.
    """
    # 1️⃣ Remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 2️⃣ Drop unwanted columns if they exist
    df = df.drop(columns=[col for col in UNWANTED_COLUMNS if col in df.columns], errors="ignore")

    # 3️⃣ Convert data types
    for col in df.select_dtypes(include=["object"]).columns:
        # If the column has relatively few unique values, treat it as categorical
        if df[col].nunique() < 0.5 * len(df):
            df[col] = df[col].astype("category")
        else:
            # Try to coerce to numeric – many datasets store numbers as strings
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # 4️⃣ Handle missing values
    # Numeric columns (int, float, etc.)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())  # mean imputation

    # Categorical columns (object or category)
    categorical_cols = df.select_dtypes(include=["category", "object"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])
            else:
                df[col] = df[col].fillna("Missing")

    # 5️⃣ Ensure integer‑like columns are proper integer dtype
    for col in numeric_cols:
        if pd.api.types.is_float_dtype(df[col]):
            if (df[col] % 1 == 0).all():
                df[col] = df[col].astype("Int64")

    return df
