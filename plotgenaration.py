import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def plot_benchmark_results():
    # 1. Find the most recently created JSON file
    list_of_files = glob.glob('benchmark_results/*.json') 
    if not list_of_files:
        print("Error: No JSON file found in 'benchmark_results' folder.")
        return
    
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Reading data: {latest_file}")

    with open(latest_file, 'r') as f:
        data = json.load(f)

    # 2. Parse the Data
    problem_sizes = sorted(list(set(r['problem_size'] for r in data['results'])))
    avg_times = []
    avg_energies = []
    std_times = []

    for size in problem_sizes:
        # Get all samples of the relevant size
        samples = [r for r in data['results'] if r['problem_size'] == size]
        
        # Get the times
        times = [r['spinnaker']['time_spent'] for r in samples]
        # Get the energies (MaxCut values)
        energies = [r['spinnaker']['best_value'] for r in samples]

        avg_times.append(np.mean(times))
        std_times.append(np.std(times))
        avg_energies.append(np.mean(energies))

    # 3. Draw the Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Time Performance ---
    ax1.errorbar(problem_sizes, avg_times, yerr=std_times, fmt='-o', color='b', capsize=5, label='SpiNNaker2')
    ax1.set_title('SpiNNaker2: Problem Size vs Time')
    ax1.set_xlabel('Problem Size (Number of Neurons)')
    ax1.set_ylabel('Solution Time (seconds)')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # We can narrow the y-axis a bit to emphasize that time stays constant
    min_y = min(avg_times) * 0.9
    max_y = max(avg_times) * 1.1
    ax1.set_ylim(min_y, max_y)
    
    # --- Plot 2: Energy (MaxCut Value) ---
    ax2.plot(problem_sizes, avg_energies, '-s', color='r', label='Best Energy')
    ax2.set_title('SpiNNaker2: Problem Size vs Found Energy')
    ax2.set_xlabel('Problem Size (Number of Neurons)')
    ax2.set_ylabel('Average MaxCut Value (Energy)')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('spinnaker_benchmark_plot.png')
    print("Plot saved: spinnaker_benchmark_plot.png")
    plt.show()

if __name__ == "__main__":
    plot_benchmark_results()
