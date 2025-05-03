import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os 

def parse_args():
    parser = argparse.ArgumentParser(description="Generate charts from CSV data.")
    parser.add_argument("--program_version", type=str, required=True, help="Path to the CSV file containing time metrics.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()    
    metrics_path = f"../{args.program_version}/output/store_time_metrics.csv"
    speedup_path = f"../{args.program_version}/output/charts/store_speedup.png"
    efficiency_path = f"../{args.program_version}/output/charts/store_efficiency.png"

    os.makedirs(os.path.dirname(speedup_path), exist_ok=True)
    os.makedirs(os.path.dirname(efficiency_path), exist_ok=True)

    df = pd.read_csv(metrics_path)

    # Group by both Data Size and Num Trees
    grouped = df.groupby(["Data Size", "Num Trees"])

    # Plot Speedup
    plt.figure(figsize=(10, 6))
    for (data_size, num_trees), group in grouped:
        data_size = round(data_size/1000000, 3)
        plt.plot(group["Threads"], group["Speedup"], label=f"{data_size}M - {num_trees}")
    plt.title("Speedup vs Threads")
    plt.xlim(1)
    plt.ylim(1)
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Data Size - Num Trees")
    plt.tight_layout()
    plt.savefig(speedup_path)

    # Plot Efficiency
    plt.figure(figsize=(10, 6))
    for (data_size, num_trees), group in grouped:
        data_size = round(data_size/1000000, 3)
        plt.plot(group["Threads"], group["Efficiency"], label=f"{data_size}M - {num_trees}")
    plt.title("Efficiency vs Threads")
    plt.xlim(1)
    plt.ylim(0, 1)
    plt.xlabel("Threads")
    plt.ylabel("Efficiency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Data Size - Num Trees")
    plt.tight_layout()
    plt.savefig(efficiency_path)

    print("Charts saved at \n \t", speedup_path, "\n\t", efficiency_path)