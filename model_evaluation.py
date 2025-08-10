import pandas as pd
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
ARTIFACTS_DIR = "artifacts"
X_TEST_PATH = os.path.join(ARTIFACTS_DIR, "X_test.csv")
Y_TEST_PATH = os.path.join(ARTIFACTS_DIR, "y_test.csv")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "linear_regression_model.pkl")
PREDICTIONS_OUTPUT = os.path.join(ARTIFACTS_DIR, "predictions_vs_actual.csv")

# Load test data
print("📂 Loading preprocessed test data...")

try:
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()  # Convert to Series
    print(f"✅ Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
except Exception as e:
    print(f"❌ Failed to load test data: {e}")
    exit()

# Load model
print("📦 Loading trained model...")
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# Make predictions
print("🔍 Making predictions...")
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5  # Manual RMSE
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("\n📊 Model Evaluation on Test Set:")
print(f"MAE :  {mae:.3f}")
print(f"MSE :  {mse:.3f}")
print(f"RMSE:  {rmse:.3f}")
print(f"R²  :  {r2:.3f}")

# Save predictions vs actuals
df_results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
df_results.to_csv(PREDICTIONS_OUTPUT, index=False)
print(f"\n📝 Predictions saved to: {PREDICTIONS_OUTPUT}")
