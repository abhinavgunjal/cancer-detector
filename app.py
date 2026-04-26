import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from models.dnn import build_dnn

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Cancer Predictor", layout="wide")

# =========================
# LOAD DATA + FEATURES
# =========================
df = pd.read_csv("data/lung_cancer.csv")
feature_names = df.drop('PULMONARY_DISEASE', axis=1).columns.tolist()

# ✅ Correct placement (no NameError)
st.write("App features:", len(feature_names))

# =========================
# LOAD MODEL (MATCHED)
# =========================
model = build_dnn(len(feature_names))
model.load_weights("model.weights.h5")

# =========================
# LOAD NORMALIZATION
# =========================
mean = np.load("mean.npy")
std = np.load("std.npy")
std[std == 0] = 1

# =========================
# PREDICTION FUNCTION
# =========================
def predict(data):
    full_data = np.array(mean.copy())

    for i, col in enumerate(feature_names):
        if col in data:
            full_data[i] = data[col]

    full_data = (full_data - mean) / std
    full_data = full_data.reshape(1, -1)

    prob = model.predict(full_data, verbose=0)[0][0]
    prob = 0.1 + 0.8 * prob

    if prob > 0.7:
        label = "High Risk"
    elif prob > 0.4:
        label = "Moderate Risk"
    else:
        label = "Low Risk"

    reasons = []
    if data.get("SMOKING"): reasons.append("smoking")
    if data.get("EXPOSURE_TO_POLLUTION"): reasons.append("pollution exposure")
    if data.get("BREATHING_ISSUE"): reasons.append("breathing issues")
    if data.get("CHEST_TIGHTNESS"): reasons.append("chest tightness")
    if data.get("LONG_TERM_ILLNESS"): reasons.append("chronic illness")
    if data.get("IMMUNE_WEAKNESS"): reasons.append("weak immunity")

    explanation = (
        f"This assessment is influenced by {', '.join(reasons)}."
        if reasons else "No major risk factors detected."
    )

    return label, prob, explanation


# =========================
# UI HEADER
# =========================
st.title("AI Cancer Predictor")
st.caption("Clinical lung cancer risk assessment system")

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([2,1])

# =========================
# INPUT SECTION
# =========================
with col1:

    age = st.slider("Age", 10, 90, 40)

    smoking = st.selectbox("Smoking", ["No","Yes"])
    alcohol = st.selectbox("Alcohol Consumption", ["No","Yes"])

    anxiety = st.selectbox("Mental Stress", ["No","Yes"])
    pollution = st.selectbox("Pollution Exposure", ["No","Yes"])
    illness = st.selectbox("Chronic Disease", ["No","Yes"])

    yellow_fingers = st.selectbox("Finger Discoloration", ["No","Yes"])
    energy = st.selectbox("Low Energy", ["No","Yes"])
    immunity = st.selectbox("Weak Immunity", ["No","Yes"])
    breathing = st.selectbox("Breathing Issue", ["No","Yes"])
    throat = st.selectbox("Throat Discomfort", ["No","Yes"])
    stress_immune = st.selectbox("Stress Immune Effect", ["No","Yes"])
    chest = st.selectbox("Chest Tightness", ["No","Yes"])

    gender = st.selectbox("Gender", ["Female","Male"])
    oxygen = st.slider("Oxygen Saturation", 80, 100, 95)
    family = st.selectbox("Family History", ["No","Yes"])
    smoking_family = st.selectbox("Smoking Family History", ["No","Yes"])

    if st.button("Assess Risk"):

        input_data = {
            "AGE": age,
            "GENDER": 1 if gender == "Male" else 0,
            "SMOKING": 1 if smoking == "Yes" else 0,
            "FINGER_DISCOLORATION": 1 if yellow_fingers == "Yes" else 0,
            "MENTAL_STRESS": 1 if anxiety == "Yes" else 0,
            "EXPOSURE_TO_POLLUTION": 1 if pollution == "Yes" else 0,
            "LONG_TERM_ILLNESS": 1 if illness == "Yes" else 0,
            "ENERGY_LEVEL": 1 if energy == "Yes" else 0,
            "IMMUNE_WEAKNESS": 1 if immunity == "Yes" else 0,
            "BREATHING_ISSUE": 1 if breathing == "Yes" else 0,
            "ALCOHOL_CONSUMPTION": 1 if alcohol == "Yes" else 0,
            "THROAT_DISCOMFORT": 1 if throat == "Yes" else 0,
            "CHEST_TIGHTNESS": 1 if chest == "Yes" else 0,
            "OXYGEN_SATURATION": oxygen,
            "FAMILY_HISTORY": 1 if family == "Yes" else 0,
            "SMOKING_FAMILY_HISTORY": 1 if smoking_family == "Yes" else 0,
            "STRESS_IMMUNE": 1 if stress_immune == "Yes" else 0
        }

        label, prob, explanation = predict(input_data)

        st.session_state.result = label
        st.session_state.prob = prob
        st.session_state.explanation = explanation


# =========================
# OUTPUT SECTION
# =========================
with col2:

    if "result" in st.session_state:
        st.subheader(st.session_state.result)
        st.progress(int(st.session_state.prob * 100))
        st.write(f"Risk Score: {st.session_state.prob:.2f}")
        st.info(st.session_state.explanation)
    else:
        st.write("Enter details and click Assess Risk")


# =========================
# FOOTER
# =========================
st.caption("⚠️ This tool is for educational purposes only.")
