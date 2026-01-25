import pandas as pd
import numpy as np
import os

def simple_matching_coefficient(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    
    total = f00 + f01 + f10 + f11
    if total == 0: 
        return 1.0
    return (f11 + f00) / total

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: 
        return 1.0
    return dot / (norm1 * norm2)

def main():
    filename = "Lab Session Data.xlsx"
    df = pd.read_excel(filename, sheet_name="thyroid0387_UCI")
    binary_df = df.select_dtypes(include=np.number).fillna(0).astype(int)
    
    if len(binary_df) >= 2:
        v1, v2 = binary_df.iloc[0].values, binary_df.iloc[1].values
        print(f"Row 0 ones: {np.sum(v1)}")
        print(f"Row 1 ones: {np.sum(v2)}")
        
        smc = simple_matching_coefficient(v1, v2)
        cos_sim = cosine_similarity(v1, v2)
        
        print(f"Simple Matching Coefficient: {smc:.4f}")
        print(f"Cosine Similarity: {cos_sim:.4f}")
    else:
        print("⚠️ Not enough data")
if __name__ == "__main__":
    main()
