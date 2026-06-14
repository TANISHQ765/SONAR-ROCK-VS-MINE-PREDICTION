# SONAR-ROCK-VS-MINE-PREDICTION
An end-to-end Machine Learning project that utilizes a Logistic Regression model to analyze 60 distinct sonar frequency data points. It accurately predicts and classifies underwater objects as either safe Rocks or hazardous Naval Mines. Includes an interactive Streamlit web interface for real-time diagnostics and user-friendly testing.
# ⚓ Sonar Rock vs. Mine Predictor

An interactive Machine Learning solution designed to process underwater sonar frequencies and accurately classify objects as either a **Rock** or a **Magnetic Naval Mine**. Built with Python and deployed with an intuitive web interface.

---

## 🚀 Interactive Live Preview
*Want to see it in action? Paste your sonar data array directly into the app interface to test live classification diagnostics!*

| Sample Data Type | Expected Output | Click to Copy Sample Array |
| :--- | :--- | :--- |
| **Subsurface Rock** | 🟢 Rock (`R`) | `0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,...` |
| **Naval Mine** | 🔴 Mine (`M`) | `0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,...` |

---

## 🛠️ How It Works (The Pipeline)
[60 Sonar Frequencies] ➡️ [NumPy Array Reshaping] ➡️ [Logistic Regression Model] ➡️ [Streamlit UI Output]
1. **Data Ingestion:** The system captures 60 discrete sonar frequency data points scaled between 0.0 and 1.0.
2. **Processing:** Data is converted into a NumPy array and reshaped to `(1, -1)` for single-instance inference.
3. **Classification:** A trained Binary Logistic Regression model predicts the target pattern.
4. **UI Diagnostic:** The Streamlit dashboard immediately reflects a green safety confirmation or a red hazard warning.

---

## 🎯 Key Features

* **Dual Input Modes:** Paste raw comma-separated data directly from your datasets, or fine-tune individual frequencies using 60 interactive manual sliders.
* **Optimized Caching:** Implements `@st.cache_resource` so the underlying machine learning model trains instantly once and stays resident in memory.
* **Responsive Visuals:** Clean, scannable layout using dynamic status alerts (`st.success` / `st.error`) for high-contrast safety feedback.

---

## 💻 Tech Stack & Libraries

* **Language:** Python
* **Machine Learning:** Scikit-Learn
* **Data Processing:** NumPy, Pandas
* **Web UI Framework:** Streamlit

* Install requirements:
pip install streamlit numpy pandas scikit-learn

Launch the interactive application:
streamlit run app.py

📈 Model Performance
Algorithm: Logistic Regression (optimized for linear binary classification patterns in sonar arrays).

Dataset Splitting: Stratified training split to ensure uniform class distribution of rocks and mines across datasets.
### 💡 Quick Tips for your GitHub Repository:
* Make sure to replace `YOUR_USERNAME` in the clone command with your actual GitHub username.
* Once you commit this, GitHub will automatically render the markdown into a beautiful, scannable documentation page.


