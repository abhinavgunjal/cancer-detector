import pandas as pd

df = pd.read_csv("data/lung_cancer.csv")

print("Columns:\n", df.columns)
print("\nFirst 5 rows:\n")
print(df.head())