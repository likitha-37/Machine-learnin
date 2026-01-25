import numpy as np
import pandas as pd
import os

def load_purchase_data(file_path):
	df = pd.read_excel(file_path, sheet_name="Purchase data")
	X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
	y = df['Payment (Rs)'].values
	print(f"Purchase data loaded: {len(df)} customers")
	return X, y

def calculate_rank(matrix):
	return np.linalg.matrix_rank(matrix)

def calculate_cost_using_pinv(X, y):
	X_pinv = np.linalg.pinv(X)
	cost = X_pinv @ y
	return cost

def main():
	file_path = "Lab Session Data.xlsx"        
	X, y = load_purchase_data(file_path)
	rank = calculate_rank(X)
	cost = calculate_cost_using_pinv(X, y)
	print(f"Matrix X shape: {X.shape}")
	print(f"Matrix rank: {rank}")
	print(f"Product costs:")
	print(f"Candies (#): ₹{cost[0]:.2f}")
	print(f"Mangoes (Kg): ₹{cost[1]:.2f}")
	print(f"Milk Packets (#): ₹{cost[2]:.2f}")
if __name__ == "__main__":
	main()
