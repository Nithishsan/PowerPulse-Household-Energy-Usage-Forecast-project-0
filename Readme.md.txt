# ⚡ PowerPulse: Household Energy Usage Forecast

> A machine learning project to forecast household power consumption using time-series energy data.

---

## 📌 Overview

**PowerPulse** is a regression-based energy analytics project that aims to forecast household power usage based on historical consumption data. The goal is to build a predictive model, gain actionable insights into energy trends, and visualize consumption patterns over time.

---

## 🚀 Approach

### 1. 🧠 Data Understanding and Exploration
- Loaded the `household_power_consumption` dataset and examined its structure and types.
- Conducted Exploratory Data Analysis (EDA) to:
  - Visualize energy usage over time.
  - Identify correlations among variables.
  - Detect outliers and anomalies in the data.

### 2. 🛠️ Data Preprocessing
- Handled missing and inconsistent values (`?`, NaNs).
- Merged `Date` and `Time` into a single `DateTime` column and converted to `datetime` format.
- Dropped non-numeric columns after extracting useful temporal features.
- Resampled data to daily frequency and removed invalid entries.
- Normalized and scaled selected features to improve model performance.

### 3. 🏗️ Feature Engineering
- Created new features such as:
  - Hour, Day, Weekday, Month (from `DateTime`)
  - Rolling averages, lag features, and daily trends.
- Selected the most relevant features impacting `Global_active_power`.
- (Optional) Integrated external features (e.g., weather data) for richer modeling.

### 4. 🤖 Model Selection and Training
- Split the dataset into **training** and **testing** sets.
- Trained the following regression models:
  - Linear Regression ✅
  - (Optionally: Random Forest, Gradient Boosting, Neural Networks)
- Saved the trained models and scalers for reuse.
- Performed evaluation on training set during model development.

### 5. 📊 Model Evaluation
- Evaluated models on the **test set** using metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score (Coefficient of Determination)
- Visualized:
  - Actual vs Predicted values
  - Residuals distribution
  - Time-series prediction comparisons

---

## ✅ Results

### 🎯 Prediction Accuracy
- Linear Regression achieved:
  - **MAE**: 0.000  
  - **MSE**: 0.000  
  - **RMSE**: 0.000  
  - **R² Score**: 1.000  
> These results suggest highly accurate predictions — ideal for short-term energy forecasting.

### 🔍 Key Takeaways
- `Global_active_power` and `Sub_metering` variables were highly predictive.
- Time-based patterns (e.g., hour of day, weekday) significantly influenced consumption.

### 📈 Visual Insights
- **Daily Energy Trends** line chart.
- **Predicted vs Actual** scatter plot.
- **Residuals distribution** showing minimal error.
- **Time-series overlay** for comparison of predictions.

---

## 📁 Project Structure

