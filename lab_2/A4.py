import pandas as pd
import os

def explore_dataset(df):
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Data types:\n{df.dtypes.value_counts()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    numeric_cols = df.select_dtypes(include='number').columns
    print(f"\nNumeric columns ({len(numeric_cols)}):")
    print(df[numeric_cols].describe())
    return df.dtypes, df.isnull().sum()

def main():
    file_path = "Lab Session Data.xlsx"
    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
    datatypes, missing = explore_dataset(df)
    total_missing = missing.sum()
    print(f"Total records: {df.shape[0]}")
    print(f"Total missing: {total_missing}")
    print(f"Missing %: {total_missing/(df.shape[0]*df.shape[1])*100:.2f}%")
if __name__ == "__main__":
    main()
