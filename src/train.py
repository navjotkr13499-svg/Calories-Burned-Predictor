import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import our data prep pipeline
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from data_prep import run_data_prep

# ── Resolve Paths ──────────────────────────────────────────────────────────────
HERE      = Path(__file__).resolve().parent   # src/
ROOT      = HERE.parent                       # project root
MOD_DIR   = ROOT / "models"
MOD_DIR.mkdir(exist_ok=True)

MODEL_PATH = MOD_DIR / "model.pkl"

# ── Step 1: Evaluate Any Model ─────────────────────────────────────────────────
def evaluate(name: str, model, X_test, y_test) -> dict:
    """
    Runs predictions and prints MAE, RMSE, and R² for a given model.
    Returns a dict of metrics.
    """
    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    print(f"\n  📊 {name}")
    print(f"     MAE  : {mae:.4f}")
    print(f"     RMSE : {rmse:.4f}")
    print(f"     R²   : {r2:.4f}")

    return {"name": name, "model": model, "mae": mae, "rmse": rmse, "r2": r2}


# ── Step 2: Train All Models ───────────────────────────────────────────────────
def train_all(X_train, X_test, y_train, y_test) -> dict:
    """
    Trains Linear Regression and Random Forest models.
    Evaluates both and returns the best one based on R² score.
    """
    models = {
        "Linear Regression" : LinearRegression(),
        "Random Forest"     : RandomForestRegressor(
                                n_estimators=100,
                                random_state=42,
                                n_jobs=-1
                              ),
    }

    results = []

    print("\n── Training & Evaluation ──────────────────────────────")
    for name, model in models.items():
        print(f"\n  🔧 Training {name}...")
        model.fit(X_train, y_train)
        result = evaluate(name, model, X_test, y_test)
        results.append(result)

    # ── Pick Best Model by R² ──────────────────────────────────────────────────
    best = max(results, key=lambda x: x["r2"])
    return best


# ── Step 3: Save Best Model ────────────────────────────────────────────────────
def save_model(model, path: Path):
    """Saves the trained model to disk using joblib."""
    joblib.dump(model, path)
    print(f"\n  💾 Best model saved → {path}")


# ── Master Training Pipeline ───────────────────────────────────────────────────
def run_training():
    """
    Runs the full training pipeline end-to-end:
      1. Data preparation
      2. Train all models
      3. Select best model
      4. Save best model
    """
    print("\n" + "="*55)
    print("           STARTING TRAINING PIPELINE")
    print("="*55)

    # Step 1: Prepare data
    X_train, X_test, y_train, y_test = run_data_prep()

    # Step 2 & 3: Train and pick best
    best = train_all(X_train, X_test, y_train, y_test)

    # Step 4: Save best model
    save_model(best["model"], MODEL_PATH)

    print("\n" + "="*55)
    print(f"  🏆 Best Model  : {best['name']}")
    print(f"  📈 R²          : {best['r2']:.4f}")
    print(f"  📉 MAE         : {best['mae']:.4f}")
    print(f"  📉 RMSE        : {best['rmse']:.4f}")
    print("="*55)
    print("✅ Training pipeline complete!\n")


# ── Run Directly ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_training()

