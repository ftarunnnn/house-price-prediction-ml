import pandas as pd
from sklearn.model_selection import train_test_split

def split_features_target(df: pd.DataFrame, target_column: str = "Price") -> tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned and feature‑engineered dataframe.
    target_column : str, optional
        Name of the column containing the target variable. Default is ``"Price"``.

    Returns
    -------
    X : pd.DataFrame
        Feature dataframe (all columns except the target).
    y : pd.Series
        Target series.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataframe")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform an 80/20 train‑test split (or any proportion you specify).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float, optional
        Fraction of the dataset to allocate to the test set. Default ``0.2`` (80/20).
    random_state : int, optional
        Seed for reproducibility. Default ``42``.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Train‑test split of features and target.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
