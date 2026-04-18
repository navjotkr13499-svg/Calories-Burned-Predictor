import pandas as pd
from pathlib import Path

# ── Resolve paths relative to THIS script's location ──────────────────────────
# Script lives at: data/sample/create_sample.py
# So we go up two levels to reach the project root
HERE       = Path(__file__).resolve().parent          # data/sample/
ROOT       = HERE.parent.parent                       # project root
RAW_DIR    = ROOT / "data" / "raw"
SAMPLE_DIR = HERE                                     # save right next to this script

# ── Load raw data ──────────────────────────────────────────────────────────────
calories = pd.read_csv(RAW_DIR / "calories.csv")
exercise = pd.read_csv(RAW_DIR / "exercise.csv")

# ── Merge on User_ID, then drop it (not a feature) ────────────────────────────
df = pd.merge(exercise, calories, on="User_ID")
df = df.drop(columns=["User_ID"])

# ── Create a small reproducible sample (50 rows) ──────────────────────────────
sample = df.sample(n=50, random_state=42).reset_index(drop=True)

# ── Save to data/sample/sample_calories.csv ───────────────────────────────────
out_path = SAMPLE_DIR / "sample_calories.csv"
sample.to_csv(out_path, index=False)

print(f"✅ Sample created : {sample.shape}")
print(f"📁 Saved to       : {out_path}")
print(f"\nFirst 3 rows:")
print(sample.head(3))
