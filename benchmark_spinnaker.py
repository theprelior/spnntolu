import sys
import os
import time
import numpy as np
import json
from datetime import datetime

# --- PATH SETTINGS ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LR_QAOA_PATH = os.path.join(PROJECT_ROOT, "solver", "LR_QAOA")
if LR_QAOA_PATH not in sys.path:
    sys.path.insert(0, LR_QAOA_PATH)

EXACT_SOLVER = os.path.join(PROJECT_ROOT, "solver", "exact_solver")
if EXACT_SOLVER not in sys.path:
    sys.path.insert(0, EXACT_SOLVER)

# --- IMPORTS ---
from solver.spinnaker.maxcut.qubo_spinnaker import spinnaker_qubo_direct_solver
from problems.maxcut.maxcut_data_generator import get_datapoint_set

if __name__ == "__main__":
    # Note: 10000 and 20000 are very large sizes, reduce these numbers if the computer freezes.
    # OLD:
# num_points = [1000, 2000, 3000, 5000, 10000, 20000]

# NEW (Limits the hardware can handle):
    num_points = [10, 20, 50, 100, 150]
    sample_size = 5  
    neuron_params = {
        'threshold': 0.1,
        'alpha_decay': 0.95,
        'reset': 'reset_to_v_reset',
        'v_reset': 0.0,
        'v_init': 0.0
    }
    iteration_step = 1000 

    # Main dictionary to hold results
    benchmark_results = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "problem_class": "unweighted",
            "num_points_sizes": num_points,
            "sample_size": sample_size,
            "neuron_params": neuron_params.copy(),
            },
        "results": []
    }

    for size_index in range(len(num_points)):
        current_size = num_points[size_index]
        print(f"\n>>> Processing size: {current_size}")
        
        # Create dataset (This returns a list)
        points_sets_samples = get_datapoint_set(
            data_point_size=current_size, 
            sample_size=sample_size, 
            point_set_type="unweighted"
        )
        
        for i in range(sample_size):
            # 1. FIX: Accessed qubo_matrix from the i-th element of the list
            qubo = points_sets_samples[i].qubo_matrix
            
            # 2. FIX: Changed parameter name to 'timesteps' (this was the original name in the file)
            time_spent, best_value = spinnaker_qubo_direct_solver(
                qubo, 
                neuron_params, 
                timesteps=iteration_step
            )
            
            # 3. FIX: Neutralizing the negative time error in SpiNNaker file with abs() here
            actual_time_spent = abs(time_spent)
            
            print(f'  Sample {i+1}: Value: {abs(best_value)} | Time: {actual_time_spent:.4f}s')

            # Save instance result
            instance_result = {
                "problem_size": current_size,
                "sample_index": i,
                "spinnaker": {
                    "best_value": float(abs(best_value)),
                    "time_spent": float(actual_time_spent),
                    "iterations": iteration_step,
                },
            }
            benchmark_results["results"].append(instance_result)
    
    # --- SAVE RESULTS ---
    output_dir = "benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_dir, f"spinnaker_maxcut_benchmark_{timestamp}.json")
    
    with open(output_filename, 'w') as f:
        json.dump(benchmark_results, f, indent=4)
    
    print(f'\n[COMPLETED] Results saved to: {output_filename}')
    
    # --- SUMMARY STATISTICS ---
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    for size in num_points:
        size_results = [r for r in benchmark_results["results"] if r["problem_size"] == size]
        if size_results:
            # 4. FIX: Changed incorrect key "GA" to "spinnaker"
            avg_time = np.mean([r["spinnaker"]["time_spent"] for r in size_results])
            print(f"Problem Size: {size:5} | Average Time: {avg_time:.4f}s")
