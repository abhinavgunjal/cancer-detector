import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Cancer Predictor", layout="wide")

# =========================
# APPLE DARK CSS
# =========================
st.markdown("""
<style>
body {
    background-color: #0b0b0c;
    color: #f5f5f7;
}

.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 32px;
    font-weight: 600;
}

.subtitle {
    color: #a1a1a6;
    margin-bottom: 25px;
}

.section {
    font-size: 14px;
    font-weight: 600;
    margin-top: 18px;
    margin-bottom: 5px;
    color: #d1d1d6;
}

/* Cards */
.card {
    background: #1c1c1e;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.05);
}

/* Result card */
.result-card {
    background: #1c1c1e;
    padding: 25px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Colors */
.low { color: #30d158; }
.medium { color: #ffd60a; }
.high { color: #ff453a; }

/* Progress bar */
.stProgress > div > div {
    background-color: #0a84ff !important;
}

/* Inputs */
.stSelectbox div[data-baseweb="select"] {
    background-color: #1c1c1e;
    border-radius: 10px;
}

.stSlider > div {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("model.keras")
mean = np.load("mean.npy")
std = np.load("std.npy")
std[std == 0] = 1

df = pd.read_csv("data/lung_cancer.csv")
feature_names = df.drop('PULMONARY_DISEASE', axis=1).columns.tolist()

# =========================
# PREDICT FUNCTION
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
        color = "high"
    elif prob > 0.4:
        label = "Moderate Risk"
        color = "medium"
    else:
        label = "Low Risk"
        color = "low"

    reasons = []
    if data["SMOKING"]: reasons.append("smoking")
    if data["EXPOSURE_TO_POLLUTION"]: reasons.append("pollution exposure")
    if data["BREATHING_ISSUE"]: reasons.append("breathing issues")
    if data["CHEST_TIGHTNESS"]: reasons.append("chest tightness")
    if data["LONG_TERM_ILLNESS"]: reasons.append("chronic illness")
    if data["IMMUNE_WEAKNESS"]: reasons.append("weak immunity")

    explanation = (
        f"This assessment is influenced by {', '.join(reasons)}."
        if reasons else "No significant risk indicators were identified."
    )

    return label, prob, explanation, color


# =========================
# HEADER
# =========================
st.markdown('<div class="title">AI Cancer Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Clinical lung cancer risk assessment system</div>', unsafe_allow_html=True)

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([2,1])

# =========================
# INPUT PANEL
# =========================
with col1:

    st.markdown('<div class="section">Basic Information</div>', unsafe_allow_html=True)
    age = st.slider("Age", 10, 90, 40)

    st.markdown('<div class="section">Lifestyle</div>', unsafe_allow_html=True)
    smoking = st.selectbox("Smoking", ["No","Yes"])
    alcohol = st.selectbox("Alcohol Consumption", ["No","Yes"])

    st.markdown('<div class="section">Medical & Behavioral</div>', unsafe_allow_html=True)
    anxiety = st.selectbox("Mental Stress", ["No","Yes"])
    pollution = st.selectbox("Pollution Exposure", ["No","Yes"])
    illness = st.selectbox("Chronic Disease", ["No","Yes"])

    st.markdown('<div class="section">Symptoms</div>', unsafe_allow_html=True)
    yellow_fingers = st.selectbox("Finger Discoloration", ["No","Yes"])
    energy = st.selectbox("Low Energy", ["No","Yes"])
    immunity = st.selectbox("Weak Immunity", ["No","Yes"])
    breathing = st.selectbox("Breathing Issue", ["No","Yes"])
    throat = st.selectbox("Throat Discomfort", ["No","Yes"])
    swallowing = st.selectbox("Stress Immune Effect", ["No","Yes"])
    chest = st.selectbox("Chest Tightness", ["No","Yes"])

    st.markdown('<div class="section">Additional Factors</div>', unsafe_allow_html=True)
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
            "STRESS_IMMUNE": 1 if swallowing == "Yes" else 0
        }

        label, prob, explanation, color = predict(input_data)

        st.session_state.result = label
        st.session_state.prob = prob
        st.session_state.explanation = explanation
        st.session_state.color = color


# =========================
# RESULT PANEL
# =========================
with col2:

    st.markdown("### Result")

    if "result" in st.session_state:

        st.markdown(f"""
        <div class="result-card">
            <h2 class="{st.session_state.color}">{st.session_state.result}</h2>
            <p style="color:#a1a1a6;">Risk Score: {st.session_state.prob:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(st.session_state.prob * 100))
        st.info(st.session_state.explanation)

    else:
        st.markdown("No prediction yet")

# =========================
# FOOTER
# =========================
st.caption("⚠️ For educational use only. Not a medical diagnosis.")
