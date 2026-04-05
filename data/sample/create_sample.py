import pandas as pd
import numpy as np

# Load full data
calories = pd.read_csv('data/raw/calories.csv')
exercise = pd.read_csv('data/raw/exercise.csv')

# Merge on user_id
df = pd.merge(exercise, calories, on='User_ID')

# Drop User_ID (privacy) and anonymize
df = df.drop('User_ID', axis=1)

# Create small sample (50 rows, random)
sample = df.sample(n=50, random_state=42).reset_index(drop=True)

# Save anonymized sample
sample.to_csv('data/sample/sample_calories.csv', index=False)

print(f"Sample created: {sample.shape}")
print("\nFirst 3 rows:")
print(sample.head(3))
