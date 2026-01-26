import logging
import os
import sys
import numpy as np
import time 

from spinnaker2 import snn, hardware

# --- IP SETTING ---
BOARD_IP = "192.168.1.17"

def spinnaker_qubo_direct_solver(qubo_matrix, neuron_params, timesteps=1000):
    '''
    Solves QUBO on SpiNNaker2 SpiNNcloud48NodeBoard.
    Memory optimization: Only 'spikes' are recorded.
    '''
    n_neurons = qubo_matrix.shape[0]
    
    # --- SCALING ---
    HARDWARE_LIMIT = 15.0
    max_val = np.max(np.abs(qubo_matrix))
    if max_val > HARDWARE_LIMIT:
        scale_factor = HARDWARE_LIMIT / max_val
        # print(f"[Info] Scaling applied (Factor: {scale_factor:.4f})")
        qubo_matrix = np.round(qubo_matrix * scale_factor)
    
    # 1. Matrix Preparation
    indices = np.nonzero(qubo_matrix)
    weights = qubo_matrix[indices]
    delays = np.ones_like(weights).astype(int)
    
    conns_qubo = np.column_stack((indices[0], indices[1], weights, delays))
    
    process_start = time.time()

    # 2. Population
    # CRITICAL FIX: Using only record=["spikes"] instead of record=["spikes", "v"]
    # Disabled voltage recording as it consumes too much RAM. Spikes are sufficient for energy calculation.
    qubo_pop = snn.Population(
        size=n_neurons,
        neuron_model="qubo_neuron",
        params=neuron_params,
        name="qubo_pop",
        record=["spikes"] 
    )
    
    # 3. Connections
    proj_qubo = snn.Projection(pre=qubo_pop, post=qubo_pop, connections=conns_qubo)

    # 4. Network
    net = snn.Network("Benchmark_Network")
    net.add(qubo_pop, proj_qubo)

    # 5. Hardware Connection
    # print(f"\n[SpiNNaker2] Connecting: {BOARD_IP}")
    try:
        hw = hardware.SpiNNcloud48NodeBoard(eth_ip=BOARD_IP)
    except:
        hw = hardware.SpiNNcloud48NodeBoard()

    # Run
    # print(f"[SpiNNaker2] Running simulation ({timesteps} steps)...")
    hw.run(net, timesteps, debug=False)
    
    time_spent = time.time() - process_start

    # 6. Results
    spikes_dict = qubo_pop.get_spikes()
    
    spike_matrix = np.zeros((n_neurons, timesteps))
    for neuron_id, times in spikes_dict.items():
        t_indices = np.array(times).astype(int)
        t_indices = t_indices[t_indices < timesteps]
        if len(t_indices) > 0:
            spike_matrix[neuron_id, t_indices] = 1
    
    # Energy Calculation
    energy_per_time = []
    # Sparse check to speed up matrix multiplication
    for t in range(timesteps):
        state = spike_matrix[:, t]
        if np.any(state):
            energy = state.T @ qubo_matrix @ state
            energy_per_time.append(energy)
        else:
            energy_per_time.append(0)
            
    if len(energy_per_time) > 0:
        best_value = np.max(np.abs(energy_per_time))
        # Scale back (To report the actual value)
        if max_val > HARDWARE_LIMIT:
             best_value = best_value / scale_factor
    else:
        best_value = 0.0

    # print(f"[SpiNNaker2] Bitti. Süre: {time_spent:.4f}s | Max Enerji: {best_value}")

    return time_spent, best_value
