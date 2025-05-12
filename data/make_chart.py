import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os 

def parse_args():
    parser = argparse.ArgumentParser(description="Generate charts from CSV data.")
    parser.add_argument("--program_version", type=str, required=True, help="Path to the CSV file containing time metrics.")
    return parser.parse_args()

def update_metrics(df, group, data_size, num_trees):
    sequential_time = df.loc[
        (df["Data Size"] == data_size) &
        (df["Num Trees"] == num_trees) &
        (df["Threads"] == 1),
        "Total Time"
    ].values[0]

    for _, row in group.iterrows():
        threads = row["Threads"]
        total_time = row["Total Time"]
        speedup = sequential_time / total_time
        efficiency = speedup / threads

        df.loc[
            (df["Data Size"] == data_size) &
            (df["Num Trees"] == num_trees) &
            (df["Threads"] == threads),
            ["Speedup", "Efficiency"]
        ] = speedup, efficiency

    df.to_csv(metrics_path, index=False)


if __name__ == "__main__":
    args = parse_args() 
    if args.program_version not in ["openmp", "mpi", "openmp_mpi"]:
        raise ValueError("Invalid program version. Choose from 'openmp', 'mpi', or 'openmp_mpi'.")   
    metrics_path = f"../{args.program_version}/output/store_time_metrics.csv"
    speedup_path = f"../{args.program_version}/output/charts/store_speedup.png"
    efficiency_path = f"../{args.program_version}/output/charts/store_efficiency.png"
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Time metrics file not found: {metrics_path}")
    xlabel = "Threads" if args.program_version == "openmp" else "Processes"
    os.makedirs(os.path.dirname(speedup_path), exist_ok=True)
    os.makedirs(os.path.dirname(efficiency_path), exist_ok=True)

    df = pd.read_csv(metrics_path)

    # Group by both Data Size and Num Trees
    grouped = df.groupby(["Data Size", "Num Trees"])

    # Plot Speedup
    plt.figure(figsize=(10, 6))
    for (data_size, num_trees), group in grouped:
        group = group.sort_values(xlabel)
        if (group["Speedup"] == -1).any() or (group["Efficiency"] == -1).any():
            update_metrics(df, group, data_size, num_trees)
            df = pd.read_csv(metrics_path)
            grouped = df.groupby(["Data Size", "Num Trees"])
            continue
        data_size = round(data_size/1e6, 3)
        plt.plot(group[xlabel], group["Speedup"], marker='o', label=f"{data_size}M - {num_trees}")
    plt.xlim(1)
    plt.ylim(1, 25)
    plt.xticks(group[xlabel].unique(), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(f"Number of {xlabel}", fontsize=16)
    plt.ylabel("Speedup", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.8, linewidth=1.2)
    plt.legend(title="Data Size - Num Trees", fontsize=14, title_fontsize=16)
    plt.tight_layout()
    plt.savefig(speedup_path)

    # Plot Efficiency
    plt.figure(figsize=(10, 6))
    for (data_size, num_trees), group in grouped:
        group = group.sort_values(xlabel)
        data_size = round(data_size/1000000, 3)
        plt.plot(group[xlabel], group["Efficiency"], marker='o', label=f"{data_size}M - {num_trees}")
    plt.xlim(1)
    plt.ylim(0, 1.2)
    plt.xticks(group[xlabel].unique(), fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(f"Number of {xlabel}", fontsize=16)
    plt.ylabel("Efficiency", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.8, linewidth=1.2)
    plt.legend(title="Data Size - Num Trees", fontsize=14, title_fontsize=16)
    plt.tight_layout()
    plt.savefig(efficiency_path)

    print("Charts saved at \n \t", speedup_path, "\n\t", efficiency_path)