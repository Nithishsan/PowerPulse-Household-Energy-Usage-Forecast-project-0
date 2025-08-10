import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and data
MODEL_PATH = "artifacts/linear_regression_model.pkl"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="⚡ PowerPulse: Energy Forecast", page_icon="⚡")
st.title("⚡ PowerPulse: Household Energy Usage Forecast")
st.markdown("""
Upload preprocessed feature data (CSV) to forecast energy usage using a trained Linear Regression model.
""")

# File upload
uploaded_file = st.file_uploader("Upload your feature CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success(f"Uploaded file with shape: {data.shape}")

        if 'DateTime' in data.columns:
            data.drop('DateTime', axis=1, inplace=True)

        st.subheader("🔍 Preview of Uploaded Data")
        st.dataframe(data.head())

        # Predict
        predictions = model.predict(data)
        st.subheader("📊 Predictions")
        st.write(predictions[:10])  # show first 10 predictions

        # Plot
        st.subheader("📈 Prediction Distribution")
        fig, ax = plt.subplots()
        sns.histplot(predictions, kde=True, ax=ax, bins=50)
        ax.set_title("Distribution of Predicted Power Usage (kWh)")
        ax.set_xlabel("Predicted Global Active Power (kWh)")
        st.pyplot(fig)

        # Download
        output_df = data.copy()
        output_df['Predicted_Power'] = predictions
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions as CSV", csv, file_name='predictions.csv', mime='text/csv')

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

else:
    st.info("Awaiting CSV file upload for prediction.")
