
import streamlit as st
import pandas as pd
import joblib

# Function to load the FRP-confined models and scaler
def load_model_scaler_frp_confined():
    scaler = joblib.load("scaler_Fcc_Scc.pkl")
    model_fcc = joblib.load("best_model_Fcc.pkl")
    model_scc = joblib.load("best_model_Scc.pkl")
    return model_fcc, model_scc, scaler

# Function to predict using CSV file for FRP-confined RAC
def predict_from_csv_frp_confined(df):
    required_columns = ['AT', '%RA', 'MSA', '%W/C', 'H', 'Efrp', '%Rfrp', 'fco', '%Sco', '%RS']
    model_fcc, model_scc, scaler = load_model_scaler_frp_confined()
    X_scaled = scaler.transform(df[required_columns])
    predictions_fcc = model_fcc.predict(X_scaled)
    predictions_scc = model_scc.predict(X_scaled)

    # Calculate strength and strain improvement
    strength_improvement = (predictions_fcc / df['fco']) * 100
    strain_improvement = (predictions_scc / df['%Sco']) * 100

    df['Predicted Fcc (MPa)'] = predictions_fcc
    df['Predicted %Scc'] = predictions_scc
    df['Strength Improvement (%)'] = strength_improvement
    df['Strain Improvement (%)'] = strain_improvement

    return df

# Function to handle manual input predictions for FRP-confined RAC
def predict_from_manual_input_frp_confined(inputs):
    input_data = pd.DataFrame([inputs])
    model_fcc, model_scc, scaler = load_model_scaler_frp_confined()
    X_scaled = scaler.transform(input_data)
    prediction_fcc = model_fcc.predict(X_scaled)
    prediction_scc = model_scc.predict(X_scaled)

    # Calculating Strength and Strain Improvement
    strength_improvement = (prediction_fcc[0] / inputs['fco']) * 100
    strain_improvement = (prediction_scc[0] / inputs['%Sco']) * 100

    st.write(f"**Predicted Ultimate Compressive Strength (Fcc): {prediction_fcc[0]:.2f} MPa**")
    st.write(f"**Predicted Ultimate Axial Strain (%Scc): {prediction_scc[0]:.2f}%**")
    st.write(f"**Strength Improvement for RAC: {strength_improvement:.2f}%**")
    st.write(f"**Strain Improvement for RAC: {strain_improvement:.2f}%**")

# Main App Interface for FRP-confined RAC
st.title("FRP-Confined RAC Prediction App")
st.markdown("Predict the ultimate compressive strength and axial strain of FRP-confined Recycled Aggregate Concrete (RAC).")

# Upload CSV file or enter data manually
st.markdown("### Input Data")
upload_method = st.radio("Choose how to input data:", ('Upload CSV File', 'Enter Manually'))

# For CSV input
if upload_method == 'Upload CSV File':
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            result_df = predict_from_csv_frp_confined(df)
            st.dataframe(result_df)

            # Allow the user to download the prediction results
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='prediction_results.csv',
                mime='text/csv',
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")

# For manual input
elif upload_method == 'Enter Manually':
    inputs = {
        'AT': st.selectbox('Aggregate Type (AT)', options=[0, 1, 2, 3], format_func=lambda x: ['NA', 'RCA', 'RCL', 'RBA'][x]),
        '%RA': st.number_input('Recycled Aggregate Replacement Ratio (%RA) [%]', min_value=0.0),
        'MSA': st.number_input('Maximum Size of Aggregate (MSA) [mm]', min_value=0.0),
        '%W/C': st.number_input('Effective Water-to-Cement Ratio (%W/C)', min_value=0.0),
        'H': st.number_input('Column Height (H) [mm]', min_value=0.0),
        'Efrp': st.number_input('Elastic Modulus of FRP (Efrp) [GPa]', min_value=0.0),
        '%Rfrp': st.number_input('FRP Reinforcement Ratio (%Rfrp) [%]', min_value=0.0),
        'fco': st.number_input('Compressive Strength of Plain Concrete (fco) [MPa]', min_value=0.0),
        '%Sco': st.number_input('Peak Strain of Plain Concrete (%Sco) [%]', min_value=0.0),
        '%RS': st.number_input('FRP Rupture Strain (%RS) [%]', min_value=0.0)
    }
    if st.button("Predict"):
        predict_from_manual_input_frp_confined(inputs)

# Footer
st.markdown("---")
st.markdown("© 2024 (Amira Ahmed, Wu Jin, Mosaad Ali ). All rights reserved.")
st.markdown("Developed by Amira Ahmed. Contact: amira672012@yahoo.com")
