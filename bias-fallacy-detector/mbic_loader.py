import pandas as pd

# Load the main labeled dataset
df = pd.read_excel("data/labeled_dataset.xlsx")

# See available columns
print("Columns:", df.columns)
print("\nSample rows:")
print(df.head())
