"""Generate a realistic synthetic housing dataset for the project."""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 500

locations = np.random.choice(["Urban", "Semi Urban", "Rural"], size=N, p=[0.4, 0.35, 0.25])
areas = np.random.randint(600, 5000, size=N).astype(float)
bedrooms = np.random.randint(1, 7, size=N)
bathrooms = np.random.randint(1, 5, size=N)
floors = np.random.randint(1, 4, size=N)
garage = np.random.randint(0, 3, size=N)
age = np.random.randint(0, 50, size=N).astype(float)
condition = np.random.choice(["Excellent", "Good", "Fair", "Poor"], size=N, p=[0.2, 0.4, 0.3, 0.1])
year_built = (2025 - age).astype(int)

# Realistic price formula (in ₹)
base = 500_000
price = (
    base
    + areas * 3500
    + bedrooms * 200_000
    + bathrooms * 150_000
    + floors * 100_000
    + garage * 250_000
    - age * 50_000
    + np.where(locations == "Urban", 1_500_000, np.where(locations == "Semi Urban", 700_000, 0))
    + np.where(condition == "Excellent", 800_000, np.where(condition == "Good", 400_000, np.where(condition == "Fair", 0, -300_000)))
    + np.random.normal(0, 300_000, size=N)
)
price = np.maximum(price, 500_000).astype(int)

df = pd.DataFrame({
    "House ID": [f"H{i:04d}" for i in range(1, N + 1)],
    "Owner Name": [f"Owner_{i}" for i in range(1, N + 1)],
    "Address": [f"Street {i}" for i in range(1, N + 1)],
    "Transaction ID": [f"T{i:06d}" for i in range(1, N + 1)],
    "Area": areas,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Floors": floors,
    "Garage": garage,
    "Age": age,
    "Location": locations,
    "Condition": condition,
    "Year Built": year_built,
    "Price": price,
})

# Add a few duplicates and missing values for realism
dup_rows = df.sample(15, random_state=7)
df = pd.concat([df, dup_rows], ignore_index=True)
for col in ["Area", "Age", "Bathrooms"]:
    mask = np.random.choice(len(df), size=8, replace=False)
    df.loc[mask, col] = np.nan

os.makedirs("data", exist_ok=True)
df.to_csv("data/house_prices.csv", index=False)
print(f"Dataset saved -> data/house_prices.csv  ({len(df)} rows, {len(df.columns)} columns)")
