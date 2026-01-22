# --- REQUIRED LIBRARIES ---
import logging
logging.basicConfig(level=logging.INFO)  # INFO to hide unnecessary details

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set file paths (adjust to your folder structure)
current_dir = os.getcwd()
src_dir = os.path.normpath(os.path.join(current_dir, './'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Your helper files
from qubo_utils import *
from qubo_visualization import *
from evaluations import *

# SpiNNaker2 Libraries
from spinnaker2 import snn, hardware, helpers
from spinnaker2.experiment_backends import BackendSettings
from spinnaker2.experiment_backends.backend_settings import ROUTING

# --- 1. SETTINGS (CHECK THIS PART) ---
S2IP = "192.168.1.17"  # <--- PUT YOUR BOARD'S IP HERE!
HW_REF = hardware.SpiNNaker2(S2IP)

print(f"Connecting to SpiNNaker2 Board at {S2IP}...")

# --- 2. DEFINE GRAPH (YOUR CUSTOM EXAMPLE) ---
# You can manually define your adjacency matrix here.
# Example: 5-node "House" shape (Square + Roof)
# Square edges: 0-1, 1-2, 2-3, 3-0; Roof edges: 2-4, 3-4
A_custom = np.array([
    [0, 1, 0, 1, 0], # Node 0
    [1, 0, 1, 0, 0], # Node 1
    [0, 1, 0, 1, 1], # Node 2 (Roof)
    [1, 0, 1, 0, 1], # Node 3 (Roof)
    [0, 0, 1, 1, 0]  # Node 4 (Roof top)
])

print("\n--- Adjacency Matrix ---")
print(A_custom)

# Draw the graph (will continue after closing)
print("Drawing graph...")
try:
    draw_graph_from_matrix(A_custom)
except:
    print("Error drawing graph, continuing...")

# --- 3. CONVERT TO QUBO MATRIX ---
Q = get_Q_from_A(A_custom)
print("\n--- Computed QUBO (Q) Matrix ---")
print(Q)

# --- 4. SET UP NEURAL NETWORK (SPINNAKER) ---

# Neuron Parameters (tweak for better results)
# alpha_decay: Decay rate (good range 0.9-0.99)
# threshold: Firing threshold
neuron_params = {
    'threshold': 0.1, 
    'alpha_decay': 0.95, 
    'reset': 'reset_to_v_reset', 
    'v_reset': 0.0,
    'v_init': 0.0
}

# Prepare Synapses
# Convert non-zero Q values to connection list
indices = np.nonzero(Q)
# Columns: [Source neuron, Target neuron, Weight, Delay]
conns_qubo = np.column_stack((indices[0], indices[1], Q[indices], np.zeros_like(Q[indices])))

# Show in Pandas for inspection
conns_qubo_df = pd.DataFrame(conns_qubo, columns=["pre-neuron", "post-neuron", "weight", "delay"])
print("\n--- Neuron Connections ---")
print(conns_qubo_df.head())  # Show first 5 connections

# Create Population
# "qubo_neuron" is your custom neuron model
qubo_pop = snn.Population(
    size=Q.shape[0], 
    neuron_model="qubo_neuron", 
    params=neuron_params,
    record=['spikes', 'v']  # Record spikes and voltage
)

# Connect neurons (recurrent network)
proj_qubo = snn.Projection(pre=qubo_pop, post=qubo_pop, connections=conns_qubo)

# Package the network
net = snn.Network("QUBO Network")
net.add(qubo_pop, proj_qubo)

# --- 5. RUN THE EXPERIMENT ---
timesteps = 1000  # Number of simulation steps (1ms x 1000 = 1 second)
print(f"\nStarting simulation ({timesteps} steps)...")

backend = snn.SpiNNaker2Backend(net, HW_REF)
# Routing settings (on-chip communication)
backend_settings = BackendSettings(routing=ROUTING.PLACE_AND_ROUTE)

# Run!
backend.run(duration=timesteps, backend_settings=backend_settings)

print("Simulation complete! Fetching data...")

# --- 6. GET RESULTS AND VISUALIZE ---
results = qubo_pop.get_data()
spikes = results['spikes']   # Spike times
voltages = results['v']      # Voltage traces

# Compute energy over time
n_neurons = Q.shape[0]
spike_matrix = np.zeros((n_neurons, timesteps))

# Convert spike list to matrix
for neuron_id, spike_times in enumerate(spikes):
    times = np.array(spike_times).astype(int)
    times = times[times < timesteps]
    spike_matrix[neuron_id, times] = 1

# Determine neuron states (0 or 1) from spikes
states_over_time = spike_matrix  # Simplified: spike = 1, no spike = 0

# Compute energy for each timestep
energy_per_time = [compute_qubo_energy(Q, states_over_time[:, t]) for t in range(timesteps)]

# Find best state (minimum energy)
min_energy = np.min(energy_per_time)
min_energy_idx = np.argmin(energy_per_time)
best_state = states_over_time[:, min_energy_idx]

print(f"\n--- RESULTS ---")
print(f"Minimum Energy: {min_energy}")
print(f"Found Solution Vector: {best_state}")

# Visualization 1: Spike plot + Energy plot
print("Drawing result plots...")
plot_spikes_energy(spike_matrix, energy_per_time, mode='minimize')  # MaxCut is a minimization problem (Ising model)

# Visualization 2: Draw partitioned graph (colored)
plt.figure()
draw_partitioned_graph(A_custom, best_state)
plt.title(f"MaxCut Result (Energy: {min_energy})")
plt.show()

print("Done.")
