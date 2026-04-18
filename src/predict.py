import joblib
import pandas as pd
from pathlib import Path

# ── Resolve Paths ──────────────────────────────────────────────────────────────
HERE       = Path(__file__).resolve().parent   # src/
ROOT       = HERE.parent                       # project root
MOD_DIR    = ROOT / "models"

MODEL_PATH  = MOD_DIR / "model.pkl"
SCALER_PATH = MOD_DIR / "scaler.pkl"

# ── Feature Order (must match training exactly) ────────────────────────────────
FEATURE_COLS = ["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

# ── Step 1: Load Model & Scaler ────────────────────────────────────────────────
def load_artifacts():
    """
    Loads the trained model and fitted scaler from disk.
    Returns: model, scaler
    """
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Model loaded   → {MODEL_PATH.name}")
    print(f"✅ Scaler loaded  → {SCALER_PATH.name}")
    return model, scaler


# ── Step 2: Preprocess Input ───────────────────────────────────────────────────
def preprocess_input(user_input: dict, scaler) -> pd.DataFrame:
    """
    Accepts raw user input as a dictionary, encodes and scales it.
    
    Expected keys:
        Gender     : str  → "male" or "female"
        Age        : int
        Height     : float (cm)
        Weight     : float (kg)
        Duration   : float (minutes)
        Heart_Rate : float (bpm)
        Body_Temp  : float (°C)

    Returns: scaled DataFrame ready for model prediction
    """
    df = pd.DataFrame([user_input])

    # Encode Gender exactly as done in data_prep.py
    df["Gender"] = df["Gender"].map({"male": 0, "female": 1})

    # Enforce correct feature column order
    df = df[FEATURE_COLS]

    # Scale using the fitted scaler
    df_scaled = pd.DataFrame(
        scaler.transform(df),
        columns=FEATURE_COLS
    )

    return df_scaled


# ── Step 3: Predict ────────────────────────────────────────────────────────────
def predict_calories(user_input: dict) -> float:
    """
    End-to-end prediction function.
    Accepts a raw input dict, preprocesses it, and returns predicted calories.

    Example input:
        {
            "Gender"     : "male",
            "Age"        : 25,
            "Height"     : 175.0,
            "Weight"     : 70.0,
            "Duration"   : 30.0,
            "Heart_Rate" : 100.0,
            "Body_Temp"  : 40.5
        }

    Returns: float → predicted calories burned
    """
    model, scaler    = load_artifacts()
    df_scaled        = preprocess_input(user_input, scaler)
    prediction       = model.predict(df_scaled)[0]
    prediction       = round(float(prediction), 2)

    print(f"\n🔮 Predicted Calories Burned : {prediction} kcal")
    return prediction


# ── Run Directly (Self-Test) ───────────────────────────────────────────────────
if __name__ == "__main__":

    print("\n" + "="*55)
    print("            RUNNING PREDICTION SELF-TEST")
    print("="*55)

    # Sample test inputs
    test_cases = [
        {
            "label"      : "Young male, moderate workout",
            "Gender"     : "male",
            "Age"        : 25,
            "Height"     : 175.0,
            "Weight"     : 70.0,
            "Duration"   : 30.0,
            "Heart_Rate" : 100.0,
            "Body_Temp"  : 40.5
        },
        {
            "label"      : "Older female, long workout",
            "Gender"     : "female",
            "Age"        : 45,
            "Height"     : 162.0,
            "Weight"     : 65.0,
            "Duration"   : 60.0,
            "Heart_Rate" : 120.0,
            "Body_Temp"  : 41.0
        },
        {
            "label"      : "Young female, short workout",
            "Gender"     : "female",
            "Age"        : 22,
            "Height"     : 158.0,
            "Weight"     : 55.0,
            "Duration"   : 15.0,
            "Heart_Rate" : 85.0,
            "Body_Temp"  : 39.8
        },
    ]

    print("\n── Test Results ───────────────────────────────────────")
    for case in test_cases:
        label = case.pop("label")   # remove label before passing to model
        print(f"\n  🧪 Test Case : {label}")
        print(f"     Input    : {case}")
        result = predict_calories(case)
        print(f"     Result   : {result} kcal")

    print("\n" + "="*55)
    print("✅ Prediction self-test complete!")
    print("="*55 + "\n")

