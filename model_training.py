import pandas as pd
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
ARTIFACTS_DIR = "artifacts"
X_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "X_train.csv")
Y_TRAIN_PATH = os.path.join(ARTIFACTS_DIR, "y_train.csv")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "linear_regression_model.pkl")

print("📂 Loading training data...")

# Load training data
X_train = pd.read_csv(X_TRAIN_PATH)
y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()  # convert from DataFrame to Series

print(f"✅ Training data loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features")

# Model initialization
model = LinearRegression()

print("🧠 Training Linear Regression model...")
model.fit(X_train, y_train)
print("✅ Model training complete.")

# Save model
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")

# Optional: Evaluate on training set
y_pred_train = model.predict(X_train)
mae = mean_absolute_error(y_train, y_pred_train)
mse = mean_squared_error(y_train, y_pred_train)
rmse = mse ** 0.5  # Manual RMSE
r2 = r2_score(y_train, y_pred_train)

print("\n📊 Model Evaluation on Training Set:")
print(f"MAE :  {mae:.3f}")
print(f"MSE :  {mse:.3f}")
print(f"RMSE:  {rmse:.3f}")
print(f"R²  :  {r2:.3f}")
