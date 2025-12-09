import streamlit as st
import numpy as np
import pickle
import pandas as pd

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="ASD Prediction App",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS FOR PROFESSIONAL LOOK
# -----------------------------
st.markdown("""
<style>

.title-container {
    background-color: #3A7CA5;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}

.input-card {
    padding: 18px;
    border: 1px solid #C9D6DF;
    border-radius: 12px;
    background-color: #F7FBFC;
    margin-bottom: 10px;
}

.result-box {
    padding: 18px;
    border-radius: 12px;
    font-size: 20px;
    text-align: center;
    font-weight: bold;
}

.success-box {
    background-color: #C8F7C5;
    color: #1B5E20;
    border: 2px solid #66BB6A;
}

.error-box {
    background-color: #F8D7DA;
    color: #721C24;
    border: 2px solid #E57373;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER SECTION
# -----------------------------
st.markdown("""
<div class="title-container">
    <h1>🧠 Autism Spectrum Disorder (ASD) Prediction</h1>
    <p>AI-assisted healthcare tool to analyze ASD Indicators</p>
   
</div>
""", unsafe_allow_html=True)

st.write("### 🩺 Please fill in the details below to get an ASD prediction.")

# -----------------------------
# LOAD MODEL & ENCODERS
# -----------------------------
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# -----------------------------
# INPUT FORM (TWO COLUMNS)
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.subheader("📝 Screening Questions")
    A_scores = {}
    for i in range(1, 11):
        A_scores[f"A{i}_Score"] = st.selectbox(f"A{i}_Score", [0, 1])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.subheader("👤 Personal Information")

    gender = st.selectbox("Gender", ["m", "f"])
    ethnicity = st.selectbox("Ethnicity", encoders["ethnicity"].classes_)
    jaundice = st.selectbox("Jaundice history", ["yes", "no"])
    autism = st.selectbox("Family Autism History", ["yes", "no"])
    country = st.selectbox("Country", encoders["contry_of_res"].classes_)
    used_app_before = st.selectbox("Used App Before", ["yes", "no"])
    relation = st.selectbox("Relation", encoders["relation"].classes_)
    age = st.number_input("Age", min_value=1, max_value=120, value=18)
    result = st.number_input("Result Score", min_value=0, max_value=50, value=10)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# ENCODING HELPER
# -----------------------------
def encode_value(column, value):
    if column in encoders:
        return encoders[column].transform([value])[0]
    return value

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("🔍 Predict ASD"):
    input_dict = {
        **A_scores,
        "gender": encode_value("gender", gender),
        "ethnicity": encode_value("ethnicity", ethnicity),
        "jaundice": encode_value("jaundice", jaundice),
        "austim": encode_value("austim", autism),
        "contry_of_res": encode_value("contry_of_res", country),
        "used_app_before": encode_value("used_app_before", used_app_before),
        "relation": encode_value("relation", relation),
        "age": age,
        "result": result
    }

    input_data = np.array(list(input_dict.values())).reshape(1, -1)
    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.markdown("<div class='result-box error-box'>🚨 The person **HAS ASD**</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-box success-box'>✨ The person **does NOT have ASD**</div>", unsafe_allow_html=True)
