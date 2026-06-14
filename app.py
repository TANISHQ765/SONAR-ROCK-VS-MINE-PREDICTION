import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------
# 1. Page Configuration & Title
# ---------------------------------------------------------
st.set_page_config(page_title="Sonar Rock vs Mine Predictor", layout="centered")
st.title("⚓ Sonar Object Predictor (Rock vs Mine)")
st.write("Input the 60 sonar frequencies below to predict whether the object is a **Rock** or a **Magnetic Mine**.")

# ---------------------------------------------------------
# 2. Model Training (Cached so it only runs once)
# ---------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    # Note: Replace this URL with your local file path if 'sonar data.csv' is on your PC
    url = "https://raw.githubusercontent.com/Anujssw/Sonar-Rock-vs-Mine-Prediction/main/sonar%20data.csv"
    sonar_data = pd.read_csv(url, header=None)
    
    X = sonar_data.drop(columns=60, axis=1)
    Y = sonar_data[60]
    
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
    
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    return model

# Load the trained model
try:
    model = load_and_train_model()
except Exception as e:
    st.error("Could not load dataset automatically. Please ensure you have an active internet connection or local dataset path set up.")

# ---------------------------------------------------------
# 3. User Interface Options
# ---------------------------------------------------------
st.subheader("Choose Input Method")
input_mode = st.radio("How would you like to enter data?", ("Paste Raw Text (Comma Separated)", "Manual Sliders"))

input_data_list = []

if input_mode == "Paste Raw Text (Comma Separated)":
    # Default placeholder text matches the sample from your Colab notebook
    default_text = "0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,0.1460,0.2792,0.3877,0.4992,0.4981,0.4972,0.5607,0.7339,0.8230,0.9173,0.9975,0.9911,0.8240,0.6498,0.5980,0.4862,0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.2930,0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,0.0245,0.0190,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055"
    
    user_string = st.text_area("Paste your 60 comma-separated values here:", default_text, height=150)
    
    try:
        input_data_list = [float(val.strip()) for val in user_string.split(",") if val.strip()]
    except ValueError:
        st.error("Please make sure all values are numbers separated by commas.")

else:
    st.write("Adjust individual frequency metrics:")
    # Creates 60 collapsible/scannable columns for manual inputs if desired
    with st.expander("Expand to fine-tune all 60 features"):
        cols = st.columns(4) # split into 4 visual columns
        for i in range(60):
            with cols[i % 4]:
                val = st.slider(f"Freq {i+1}", min_value=0.0, max_value=1.0, value=0.05, step=0.0001)
                input_data_list.append(val)

# ---------------------------------------------------------
# 4. Prediction Logic
# ---------------------------------------------------------
st.markdown("---")

if st.button("🔍 Run Sonar Diagnostics", type="primary"):
    if len(input_data_list) != 60:
        st.warning(f"The model requires exactly 60 attributes. You currently have provided {len(input_data_list)}.")
    else:
        # Step-for-step prediction code from your Colab file
        input_data_as_numpy_array = np.asarray(input_data_list)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        
        prediction = model.predict(input_data_reshaped)
        
        # Display Results Beautifully
        st.subheader("Diagnostic Output")
        if prediction[0] == 'R':
            st.success("🟢 **The object is a Rock.** It's safe to pass.")
        else:
            st.error("🔴 **⚠️ HAZARD DETECTED: The object is a Mine!**")