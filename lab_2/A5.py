import pandas as pd
import numpy as np
import os

def jaccard_coefficient(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    denom = f01 + f10 + f11
    return f11 / denom if denom > 0 else 0

def main():
    file_path = "Lab Session Data.xlsx"
    df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
    binary_df = df.select_dtypes(include=np.number).fillna(0).astype(int)
    if len(binary_df) >= 2:
        v1, v2 = binary_df.iloc[0].values, binary_df.iloc[1].values
        jc = jaccard_coefficient(v1, v2)
        print(f"Row 0 vs Row 1:")
        print(f"Vector 1 ones: {np.sum(v1)}")
        print(f"Vector 2 ones: {np.sum(v2)}")
        print(f"Jaccard Coefficient: {jc:.4f}")
    else:
        print("⚠️ Insufficient data")
if __name__ == "__main__":
    main()
