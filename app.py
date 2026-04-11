import streamlit as st
import pandas as pd
import pickle
import os

# --- Page Config ---
st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* Base */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #f0f4f8;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    /* Hero Header */
    .hero {
        background: linear-gradient(135deg, #0f2942 0%, #1a4a7a 60%, #1e6091 100%);
        border-radius: 20px;
        padding: 48px 52px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 260px; height: 260px;
        background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 30%;
        width: 180px; height: 180px;
        background: radial-gradient(circle, rgba(100,180,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        color: #a8d4f5;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 6px 14px;
        border-radius: 20px;
        margin-bottom: 16px;
    }
    .hero h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        color: white;
        margin: 0 0 12px 0;
        line-height: 1.15;
    }
    .hero p {
        color: rgba(255,255,255,0.65);
        font-size: 1rem;
        margin: 0;
        font-weight: 300;
        max-width: 480px;
    }
    .hero-icon {
        position: absolute;
        right: 52px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 80px;
        opacity: 0.15;
    }

    /* Stat Pills */
    .stats-row {
        display: flex;
        gap: 12px;
        margin-bottom: 28px;
        flex-wrap: wrap;
    }
    .stat-pill {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 22px;
        flex: 1;
        min-width: 140px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stat-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f2942;
        font-family: 'DM Serif Display', serif;
    }
    .stat-label {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-top: 2px;
    }

    /* Card */
    .card {
        background: white;
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .card-title {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 16px;
    }

    /* Multiselect styling */
    .stMultiSelect > div > div {
        border-radius: 10px !important;
        border-color: #e2e8f0 !important;
        background: #f8fafc !important;
    }
    .stMultiSelect > div > div:focus-within {
        border-color: #1a4a7a !important;
        box-shadow: 0 0 0 3px rgba(26,74,122,0.1) !important;
    }

    /* Predict Button */
    .stButton > button {
        background: linear-gradient(135deg, #0f2942, #1a4a7a) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 40px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        font-family: 'DM Sans', sans-serif !important;
        width: 100% !important;
        height: auto !important;
        letter-spacing: 0.5px;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(15,41,66,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(15,41,66,0.4) !important;
    }

    /* Result Box */
    .result-box {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 1.5px solid #86efac;
        border-radius: 16px;
        padding: 28px 32px;
        margin-top: 16px;
    }
    .result-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #16a34a;
        margin-bottom: 8px;
    }
    .result-disease {
        font-family: 'DM Serif Display', serif;
        font-size: 2rem;
        color: #14532d;
        margin: 0;
    }
    .result-note {
        margin-top: 12px;
        font-size: 0.82rem;
        color: #4ade80;
        color: #15803d;
    }

    /* Warning Box */
    .warn-box {
        background: #fffbeb;
        border: 1.5px solid #fcd34d;
        border-radius: 12px;
        padding: 16px 20px;
        color: #92400e;
        font-size: 0.9rem;
        margin-top: 12px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #cbd5e1;
        font-size: 0.78rem;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
models_path = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def load_model():
    with open(os.path.join(models_path, "rf_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(models_path, "symptoms_list.pkl"), "rb") as f:
        symptoms_list = pickle.load(f)
    return model, symptoms_list

model, symptoms_list = load_model()

display_symptoms = sorted([s.strip() for s in symptoms_list])
symptom_map = {s.strip(): s for s in symptoms_list}

# --- Hero ---
st.markdown("""
<div class="hero">
    <div class="hero-badge">🩺 AI Powered</div>
    <h1>MediPredict</h1>
    <p>Select your symptoms and our machine learning model will predict the most likely condition.</p>
    <div class="hero-icon">⚕️</div>
</div>
""", unsafe_allow_html=True)

# --- Stats Row ---
st.markdown(f"""
<div class="stats-row">
    <div class="stat-pill">
        <div class="stat-number">{len(symptoms_list)}</div>
        <div class="stat-label">Symptoms</div>
    </div>
    <div class="stat-pill">
        <div class="stat-number">41</div>
        <div class="stat-label">Diseases</div>
    </div>
    <div class="stat-pill">
        <div class="stat-number">99%</div>
        <div class="stat-label">Accuracy</div>
    </div>
    <div class="stat-pill">
        <div class="stat-number">4920</div>
        <div class="stat-label">Records Trained</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Input Card ---
st.markdown('<div class="card"><div class="card-title">Step 1 — Select Your Symptoms</div>', unsafe_allow_html=True)

selected_display = st.multiselect(
    label="",
    options=display_symptoms,
    placeholder="🔍  Search and select symptoms...",
)

if selected_display:
    st.markdown(f"<p style='color:#64748b; font-size:0.85rem; margin-top:8px;'>✅ {len(selected_display)} symptom(s) selected</p>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- Predict Card ---
st.markdown('<div class="card"><div class="card-title">Step 2 — Get Prediction</div>', unsafe_allow_html=True)

if st.button("🔬  Analyse Symptoms"):
    if not selected_display:
        st.markdown('<div class="warn-box">⚠️ Please select at least one symptom before predicting.</div>', unsafe_allow_html=True)
    else:
        with st.spinner("Analysing symptoms..."):
            input_vector = pd.DataFrame(0, index=[0], columns=symptoms_list)
            for symptom in selected_display:
                original = symptom_map.get(symptom)
                if original and original in input_vector.columns:
                    input_vector[original] = 1

            prediction = model.predict(input_vector)[0]
            proba = model.predict_proba(input_vector).max() * 100

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Predicted Condition</div>
            <p class="result-disease">{prediction}</p>
            <div class="result-note">
                Based on {len(selected_display)} symptom(s)
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- Disclaimer ---
st.markdown("""
<div class="footer">
    ⚠️ This tool is for educational purposes only and is not a substitute for professional medical advice.<br>
    Always consult a qualified healthcare provider for diagnosis and treatment.
</div>
""", unsafe_allow_html=True)