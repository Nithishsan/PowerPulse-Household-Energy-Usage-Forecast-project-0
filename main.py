import pandas as pd
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_csv("household_power_consumption.txt", sep=';', na_values='?', low_memory=False)

# Combine 'Date' and 'Time' into 'DateTime'
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)

# Drop original 'Date' and 'Time'
df.drop(columns=['Date', 'Time'], inplace=True)

# Set 'DateTime' as index
df.set_index('DateTime', inplace=True)

# Info & Head
print(df.info())
print(df.head())

# Missing values
print("\nMissing values per column:\n", df.isnull().sum())

# Summary statistics
print("\nSummary statistics:\n", df.describe())

# Resample to daily average
df_sample = df['Global_active_power'].dropna().resample('D').mean()

# Plot daily average global active power
plt.figure(figsize=(15, 5))
plt.plot(df_sample, label='Daily Avg Global Active Power (kW)')
plt.title('Daily Average Global Active Power')
plt.ylabel('kW')
plt.xlabel('Date')
plt.legend()
plt.tight_layout()
plt.show()

# Optional: Reset index if you want to use this for ML
df_reset = df.reset_index()

# Drop 'DateTime' if needed for modeling
if 'DateTime' in df_reset.columns:
    df_reset.drop(columns=['DateTime'], inplace=True)

# Uncomment to save cleaned data
# df_reset.to_csv("cleaned_household_data.csv", index=False)
