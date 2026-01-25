import pandas as pd
import numpy as np
def classify_customer(candies, mangoes, milk, payment):
	if payment > 200:
		return "RICH"
	else:
		return "POOR"

def main():
 	df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")
	print("\nCustomer | Candies | Mango | Milk | Payment | Class")
	rich_count = 0
	for i, row in df.iterrows():
		customer_class = classify_customer(
          	row['Candies (#)'], 
            	row['Mangoes (Kg)'], 
            	row['Milk Packets (#)'], 
            	row['Payment (Rs)']
        	)
        if customer_class == "RICH":
            	rich_count += 1
        print(f"{i+1:8} | {row['Candies (#)']:7} | {row['Mangoes (Kg)']:5} | "
              f"{row['Milk Packets (#)']:4} | ₹{row['Payment (Rs)']:4} | {customer_class}")
    	total = len(df)
    	print(f"\nRESULTS:")
    	print(f"RICH customers: {rich_count}/{total}")
   	 print(f"POOR customers: {total-rich_count}/{total}")
    	print(f"RICH %: {rich_count/total*100:.1f}%")
if __name__ == "__main__":
    main()
