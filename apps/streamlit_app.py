import sys
import streamlit as st
from pathlib import Path

# ── Resolve Paths ──────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent   # app/
ROOT = HERE.parent                       # project root
SRC  = ROOT / "src"

# Make src/ importable
sys.path.append(str(SRC))
from predict import predict_calories

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Calorie Burn Predictor",
    page_icon  = "🔥",
    layout     = "centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }

        .result-card {
            background: linear-gradient(135deg, #ff6a00, #ee0979);
            border-radius: 16px;
            padding: 30px;
            text-align: center;
            color: white;
            margin-top: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }

        .result-card h1 {
            font-size: 3.5rem;
            margin: 0;
            font-weight: 800;
        }

        .result-card p {
            font-size: 1.1rem;
            margin: 5px 0 0 0;
            opacity: 0.9;
        }

        .tip-box {
            background-color: #fff8e1;
            border-left: 5px solid #FFA500;
            border-radius: 8px;
            padding: 12px 16px;
            margin-top: 20px;
            font-size: 0.95rem;
            color: #555;
        }

        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #ff6a00, #ee0979);
            color: white;
            border: none;
            padding: 14px;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 700;
            cursor: pointer;
            transition: opacity 0.3s;
        }

        .stButton > button:hover {
            opacity: 0.88;
        }
    </style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🔥 Calorie Burn Predictor")
st.markdown("Enter your workout details below and find out how many **calories you burned!**")
st.markdown("---")


# ── Sidebar Inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("👤 Personal Details")

gender = st.sidebar.selectbox(
    "Gender",
    options=["male", "female"],
    format_func=lambda x: "Male 👨" if x == "male" else "Female 👩"
)

age = st.sidebar.slider(
    "Age (years)",
    min_value=10,
    max_value=80,
    value=25,
    step=1
)

height = st.sidebar.slider(
    "Height (cm)",
    min_value=120.0,
    max_value=220.0,
    value=170.0,
    step=0.5
)

weight = st.sidebar.slider(
    "Weight (kg)",
    min_value=30.0,
    max_value=150.0,
    value=70.0,
    step=0.5
)

st.sidebar.header("🏋️ Workout Details")

duration = st.sidebar.slider(
    "Duration (minutes)",
    min_value=1.0,
    max_value=120.0,
    value=30.0,
    step=1.0
)

heart_rate = st.sidebar.slider(
    "Heart Rate (bpm)",
    min_value=60.0,
    max_value=200.0,
    value=100.0,
    step=1.0
)

body_temp = st.sidebar.slider(
    "Body Temperature (°C)",
    min_value=36.0,
    max_value=42.5,
    value=40.0,
    step=0.1
)


# ── Input Summary Table ────────────────────────────────────────────────────────
st.subheader("📋 Your Input Summary")

col1, col2 = st.columns(2)
with col1:
    st.metric("Gender",        gender.capitalize())
    st.metric("Age",           f"{age} yrs")
    st.metric("Height",        f"{height} cm")
    st.metric("Weight",        f"{weight} kg")

with col2:
    st.metric("Duration",      f"{duration} min")
    st.metric("Heart Rate",    f"{heart_rate} bpm")
    st.metric("Body Temp",     f"{body_temp} °C")

st.markdown("---")


# ── Predict Button ─────────────────────────────────────────────────────────────
if st.button("🔥 Predict My Calorie Burn"):

    user_input = {
        "Gender"     : gender,
        "Age"        : age,
        "Height"     : height,
        "Weight"     : weight,
        "Duration"   : duration,
        "Heart_Rate" : heart_rate,
        "Body_Temp"  : body_temp
    }

    with st.spinner("⚙️ Calculating your calorie burn..."):
        result = predict_calories(user_input)

    # ── Result Card ────────────────────────────────────────────────────────────
    st.markdown(f"""
        <div class="result-card">
            <p>🔥 Estimated Calories Burned</p>
            <h1>{result} kcal</h1>
            <p>Based on your workout and personal details</p>
        </div>
    """, unsafe_allow_html=True)

    # ── Motivational Tip Based on Result ──────────────────────────────────────
    if result < 100:
        tip = "💧 Good start! Stay hydrated and try increasing your workout duration."
    elif result < 250:
        tip = "💪 Nice effort! A consistent routine will help you burn even more."
    elif result < 450:
        tip = "🏃 Great workout! You're in the fat-burning zone. Keep it up!"
    else:
        tip = "🚀 Incredible session! Elite-level calorie burn. Make sure to recover well!"

    st.markdown(f'<div class="tip-box">💡 <strong>Tip:</strong> {tip}</div>', unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><sub>Built with ❤️ using Streamlit & Scikit-learn</sub></center>",
    unsafe_allow_html=True
)

