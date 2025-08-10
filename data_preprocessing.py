import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Output directory
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

print("📂 Loading raw data...")

# Load raw data
df = pd.read_csv("household_power_consumption.txt", sep=";", na_values="?", low_memory=False)

print(f"✅ Raw data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Combine and convert Date + Time to datetime
df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
df.drop(columns=["Date", "Time"], inplace=True)

# Set index (optional)
df.set_index("DateTime", inplace=False)

# Extract time features
df["hour"] = df["DateTime"].dt.hour
df["day"] = df["DateTime"].dt.day
df["month"] = df["DateTime"].dt.month
df["weekday"] = df["DateTime"].dt.weekday
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

# Drop original datetime column (not needed for model)
df.drop(columns=["DateTime"], inplace=True)
print("🕒 DateTime column processed.")

# Convert all columns to numeric (in case any string-type remains)
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# Feature engineering
df["total_power_wh"] = df["Global_active_power"] * 1000 / 60
df["unmetered_power"] = df["total_power_wh"] - (
    df["Sub_metering_1"] + df["Sub_metering_2"] + df["Sub_metering_3"]
)

# Drop the original target to predict (can be swapped as needed)
target = "Global_active_power"

# Features and target
X = df.drop(columns=[target])
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=42)

# Save outputs
X_train.to_csv(os.path.join(ARTIFACTS_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(ARTIFACTS_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(ARTIFACTS_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(ARTIFACTS_DIR, "y_test.csv"), index=False)

print("✅ Preprocessing complete. Files saved:")
print(f"   - {ARTIFACTS_DIR}/X_train.csv")
print(f"   - {ARTIFACTS_DIR}/X_test.csv")
print(f"   - {ARTIFACTS_DIR}/y_train.csv")
print(f"   - {ARTIFACTS_DIR}/y_test.csv")
