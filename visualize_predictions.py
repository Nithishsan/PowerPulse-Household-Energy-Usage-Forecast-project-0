import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the actual vs predicted values
file_path = "artifacts/predictions_vs_actual.csv"
df = pd.read_csv(file_path)

# Scatter Plot: Predicted vs Actual
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Actual', y='Predicted', data=df, alpha=0.3)
plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Total Power (Wh)")
plt.ylabel("Predicted Total Power (Wh)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual Plot
df['Residuals'] = df['Actual'] - df['Predicted']
plt.figure(figsize=(8, 6))
sns.histplot(df['Residuals'], kde=True, bins=50)
plt.title("Distribution of Residuals (Actual - Predicted)")
plt.xlabel("Residuals")
plt.grid(True)
plt.tight_layout()
plt.show()

# Line Plot (first 500 samples)
plt.figure(figsize=(12, 6))
plt.plot(df['Actual'][:500], label='Actual', linewidth=2)
plt.plot(df['Predicted'][:500], label='Predicted', linestyle='--')
plt.title("Actual vs Predicted (First 500 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Total Power (Wh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
