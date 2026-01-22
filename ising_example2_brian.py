from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# 1. PARAMETER TRANSFER (Simulating data coming from Qifeng's GPU code)
# -----------------------------------------------------------------------
# N represents the number of spins (neurons). 
# Using 100 for simulation speed; GPU code handles 10,000+.
N = 100 
rng = np.random.default_rng(42)

# Coupling Matrix J (Symmetric, zero diagonal) 
# J[i,j] > 0: Neurons encourage each other (Ferromagnetic)
# J[i,j] < 0: Neurons inhibit each other (Anti-ferromagnetic)
J = rng.normal(0, 1, size=(N, N))
J = 0.5 * (J + J.T)
np.fill_diagonal(J, 0)

# External Field h (Local bias for each spin)
h = rng.normal(0, 0.1, size=(N,))

# -----------------------------------------------------------------------
# 2. NEURAL MODEL DEFINITION (Mimicking SpiNNaker2 qubo_neuron logic)
# -----------------------------------------------------------------------
tau = 10*ms
v_threshold = 0.5*mV
v_reset = 0*mV

# Membrane equation: dv/dt represents the accumulation of synaptic inputs
eqs = '''
dv/dt = -v/tau : volt (unless refractory)
'''

# Create a group of N neurons
G = NeuronGroup(N, eqs, threshold='v > v_threshold', reset='v = v_reset', 
                refractory=1*ms, method='exact')

# -----------------------------------------------------------------------
# 3. SYNAPTIC MAPPING (Mapping J Matrix to Physical Connections)
# -----------------------------------------------------------------------
# Instead of matrix multiplication, we use a Synaptic network.
S = Synapses(G, G, 'w : volt', on_pre='v += w')
S.connect(condition='i != j')

# Map the J matrix values directly to synaptic weights
S.w = J[S.i, S.j] * 0.1 * mV 

# -----------------------------------------------------------------------
# 4. INITIALIZATION & ANNEALING (Starting the search for Global Minimum)
# -----------------------------------------------------------------------
# Initialize voltage with the local field h and some random noise
G.v = h * mV
G.v += (np.random.rand(N) - 0.5) * mV # Initial stochastic push

# -----------------------------------------------------------------------
# 5. EXECUTION & DATA COLLECTION
# -----------------------------------------------------------------------
spikemon = SpikeMonitor(G)
statemon = StateMonitor(G, 'v', record=True)

# Run the simulation
run(200*ms)

# -----------------------------------------------------------------------
# 6. VISUALIZATION & ANALYSIS
# -----------------------------------------------------------------------
plt.figure(figsize=(10, 8))

# Subplot 1: Raster Plot (Visualizing the Aksynchronous search)
plt.subplot(2, 1, 1)
plt.plot(spikemon.t/ms, spikemon.i, '.k', markersize=2)
plt.title('Asynchronous Ising Solver (Spike-based Interaction)')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron (Spin) ID')

# Subplot 2: Solution Activity (Ateşleme yoğunluğu = Spin state)
plt.subplot(2, 1, 2)
spike_counts = spikemon.count
plt.bar(range(N), spike_counts)
plt.title('Final Activity Distribution (Candidate Solution)')
plt.xlabel('Neuron ID')
plt.ylabel('Spike Count')

plt.tight_layout()
plt.savefig('ising_simulation_results.png')
print("Simulation complete. Results saved as 'ising_simulation_results.png'.")
