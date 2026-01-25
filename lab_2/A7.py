import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def jaccard_coefficient(v1, v2):
    both_1 = np.sum((v1 == 1) & (v2 == 1))
    only1 = np.sum((v1 == 1) & (v2 == 0)) + np.sum((v1 == 0) & (v2 == 1))
    if both_1 + only1 == 0:
        return 1.0
    return both_1 / (both_1 + only1)

def simple_matching_coefficient(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    
    total = f00 + f01 + f10 + f11
    return (f11 + f00) / total if total > 0 else 1.0

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    return dot / (norm1 * norm2)

def similarity_matrices(data):
    n = data.shape[0]
    jc = np.zeros((n, n))
    smc = np.zeros((n, n))
    cos = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            jc[i,j] = jaccard_coefficient(data[i], data[j])
            smc[i,j] = simple_matching_coefficient(data[i], data[j])
            cos[i,j] = cosine_similarity(data[i], data[j])
    
    return jc, smc, cos

def main():
    filename = "Lab Session Data.xlsx"
    df = pd.read_excel(filename, sheet_name="thyroid0387_UCI")
    binary_df = df.select_dtypes(include=np.number).fillna(0).astype(int)
    if len(binary_df) >= 20:
        first20 = binary_df.iloc[:20].values
        jc_mat, smc_mat, cos_mat = similarity_matrices(first20)
        
        print("imilarity matrices computed (20x20)")
        print(f"Jaccard range: {jc_mat.min():.3f} - {jc_mat.max():.3f}")
        print(f"Simple Matching range: {smc_mat.min():.3f} - {smc_mat.max():.3f}")
        print(f"Cosine range: {cos_mat.min():.3f} - {cos_mat.max():.3f}")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.heatmap(jc_mat, annot=False, cmap='coolwarm', ax=axes[0], 
                   vmin=0, vmax=1, cbar_kws={'label': 'Similarity'})
        axes[0].set_title('Jaccard Coefficient')  
        sns.heatmap(smc_mat, annot=False, cmap='coolwarm', ax=axes[1], 
                   vmin=0, vmax=1, cbar_kws={'label': 'Similarity'})
        axes[1].set_title('Simple Matching')  
        sns.heatmap(cos_mat, annot=False, cmap='coolwarm', ax=axes[2], 
                   vmin=-1, vmax=1, cbar_kws={'label': 'Similarity'})
        axes[2].set_title('Cosine Similarity')  
        plt.tight_layout()
        plt.show()
    else:
        print("Need 20+ rows")
if __name__ == "__main__":
    main()
