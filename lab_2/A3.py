import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

def calc_mean_var_numpy(data):
    return np.mean(data), np.var(data)

def calc_mean_var_manual(data):
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return mean, variance

def measure_time(func, data, runs=10):
    times = []
    for _ in range(runs):
        start = time.time()
        func(data)
        times.append(time.time() - start)
    return sum(times) / runs

def wednesday_avg(df):
    wed_days = df[df['Day'] == 'Wednesday']
    return wed_days['Price'].mean() if len(wed_days) > 0 else np.nan

def april_avg(df):
    april_days = df[df['Month'] == 'Apr']
    return april_days['Price'].mean() if len(april_days) > 0 else np.nan

def loss_probability(chg_col):
    clean_changes = chg_col.dropna()
    losses = clean_changes[clean_changes < 0]
    return len(losses) / len(clean_changes) if len(clean_changes) > 0 else 0

def main():
    filename = "Lab Session Data.xlsx"
    df = pd.read_excel(filename, sheet_name="IRCTC Stock Price")
    prices = df['Price'].dropna().values
    print(f"Dataset: {len(df)} trading days")
    mean_np, var_np = calc_mean_var_numpy(prices)
    mean_manual, var_manual = calc_mean_var_manual(prices)
    time_np = measure_time(calc_mean_var_numpy, prices)
    time_manual = measure_time(calc_mean_var_manual, prices)
    wed_avg = wednesday_avg(df)
    apr_avg = april_avg(df)
    loss_prob = loss_probability(df['Chg%'])
    wed_count = len(df[df['Day'] == 'Wednesday'])
    apr_count = len(df[df['Month'] == 'Apr'])
    print("\nRESULTS:")
    print(f"NumPy Mean: ₹{mean_np:.2f}, Var: {var_np:.0f}")
    print(f"Manual Mean: ₹{mean_manual:.2f}, Var: {var_manual:.0f}")
    print(f"NumPy: {time_np*1e6:.0f}μs, Manual: {time_manual*1e6:.0f}μs")
    if np.isnan(wed_avg):
        print(f"Wednesday ({wed_count} days): N/A")
    else:
        print(f"Wednesday ({wed_count} days): ₹{wed_avg:.0f}")
    print(f"April ({apr_count} days): ₹{apr_avg:.0f}")
    print(f"P(Loss): {loss_prob:.1%}")
    plt.figure(figsize=(12, 6))
    plt.scatter(df['Day'], df['Chg%'], alpha=0.7, s=50)
    plt.title('IRCTC: Daily Change by Day')
    plt.xlabel('Day')
    plt.ylabel('Change %')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()
