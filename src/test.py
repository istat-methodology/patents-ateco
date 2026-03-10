import pandas as pd

df = pd.read_csv("data/processed/patents_ateco_predictions_test.csv")
print("righe totali:", len(df))
print("row_id unici:", df["row_id"].nunique())
print("duplicati:", len(df) - df["row_id"].nunique())
