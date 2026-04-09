import pandas as pd
from pathlib import Path
import os

# Get absolute path
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "experiments" / "datasets"

df = pd.read_csv(DATA_DIR / "Electric_Production.csv", header=0)
print("Head:")
print(df.head())
print("\nDTypes:")
print(df.dtypes)
print("\nShape:", df.shape)
print("\nFirst column:")
print(df.iloc[:5, 0].values)
print("Type of first value:", type(df.iloc[0, 0]))
