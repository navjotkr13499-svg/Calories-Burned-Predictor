import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ── Resolve Paths ──────────────────────────────────────────────────────────────
# Script lives at: src/data_prep.py
# So we go up one level to reach the project root
HERE     = Path(__file__).resolve().parent   # src/
ROOT     = HERE.parent                       # project root
RAW_DIR  = ROOT / "data" / "raw"
MOD_DIR  = ROOT / "models"                  # where scaler will be saved

# Create models/ directory if it doesn't exist yet
MOD_DIR.mkdir(exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
TARGET_COL   = "Calories"
SCALER_PATH  = MOD_DIR / "scaler.pkl"

# ── Step 1: Load Raw Data ──────────────────────────────────────────────────────
def load_raw_data() -> pd.DataFrame:
    """
    Loads and merges exercise.csv and calories.csv on User_ID.
    Drops User_ID after merging as it is not a feature.
    """
    calories = pd.read_csv(RAW_DIR / "calories.csv")
    exercise = pd.read_csv(RAW_DIR / "exercise.csv")

    df = pd.merge(exercise, calories, on="User_ID")
    df = df.drop(columns=["User_ID"])

    print(f"✅ Data loaded & merged   : {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ── Step 2: Clean & Encode ─────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering and encoding steps:
      - Encodes Gender (male → 0, female → 1)
      - Validates no missing values remain
    Returns the cleaned DataFrame.
    """
    df = df.copy()

    # Encode Gender
    df["Gender"] = df["Gender"].map({"male": 0, "female": 1})

    # Sanity check: confirm no nulls after encoding
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        raise ValueError(f"❌ Found {null_count} missing values after preprocessing!")

    print(f"✅ Preprocessing complete  : Gender encoded, no missing values")
    return df


# ── Step 3: Split ──────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    """
    Splits the DataFrame into X/y train/test sets.
    Returns: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"✅ Train/test split done   : Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ── Step 4: Scale ──────────────────────────────────────────────────────────────
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fits a StandardScaler on X_train and transforms both X_train and X_test.
    Saves the fitted scaler to models/scaler.pkl for reuse in prediction/app.
    Returns: X_train_scaled, X_test_scaled (as DataFrames)
    """
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()

    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    # Save scaler for later use in predict.py and streamlit_app.py
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Scaling complete        : Scaler saved → {SCALER_PATH}")

    return X_train_scaled, X_test_scaled


# ── Master Pipeline Function ───────────────────────────────────────────────────
def run_data_prep():
    """
    Runs the full data preparation pipeline end-to-end.
    Returns: X_train_scaled, X_test_scaled, y_train, y_test
    """
    print("\n" + "="*55)
    print("         STARTING DATA PREPARATION PIPELINE")
    print("="*55)

    df                             = load_raw_data()
    df                             = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled  = scale_features(X_train, X_test)

    print("="*55)
    print("✅ Data preparation complete!")
    print("="*55 + "\n")

    return X_train_scaled, X_test_scaled, y_train, y_test


# ── Run Directly ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = run_data_prep()

    print("── Final Output Shapes ──")
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  y_train : {y_train.shape}")
    print(f"  y_test  : {y_test.shape}")

