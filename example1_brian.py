from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# 1. Memory Pattern (5x5 'A' Letter Template)
# 1: Firing neuron, -1: Silent neuron
pattern = np.array([
    -1,  1,  1,  1, -1,
     1, -1, -1, -1,  1,
     1,  1,  1,  1,  1,
     1, -1, -1, -1,  1,
     1, -1, -1, -1,  1
]).flatten()

N = len(pattern)

# 2. Hebbian Weight Matrix
W = np.outer(pattern, pattern)
np.fill_diagonal(W, 0)
W = W * 0.01  # Scale for Brian2 units

# 3. Neuron Model (similar to LIF model on SpiNNaker2)
tau = 10*ms
v_rest = 0*mV
v_threshold = 0.5*mV
v_reset = 0*mV

eqs = '''
dv/dt = -(v - v_rest) / tau : volt (unless refractory)
'''

# Create neuron population
G = NeuronGroup(N, eqs, threshold='v > v_threshold', reset='v = v_reset', 
                refractory=2*ms, method='exact')

# 4. Recurrent Connections (Synapses)
S = Synapses(G, G, 'w : volt', on_pre='v += w')
S.connect(condition='i != j')
# Load weights from Hebbian matrix
S.w = W[S.i, S.j] * volt

# 5. Noisy Input
# Corrupt the original pattern (flip some 1s to -1 and -1s to 1)
noisy_pattern = pattern.copy()
noisy_pattern[1:4] = -1  # Erase top part
noisy_pattern[10:15] = -1  # Erase middle line

# Deliver the noisy pattern as "spikes" to the network
start_spikes = np.where(noisy_pattern == 1)[0]
P = SpikeGeneratorGroup(N, start_spikes, np.zeros(len(start_spikes))*ms)
S_in = Synapses(P, G, on_pre='v += 1.0*mV')
S_in.connect(j='i')

# 6. Recording and Running
statemon = StateMonitor(G, 'v', record=True)
spikemon = SpikeMonitor(G)

run(100*ms)

# 7. Visualization
plt.figure(figsize=(12, 5))

# Raster Plot (Spike Activity)
plt.subplot(1, 2, 1)
plt.plot(spikemon.t/ms, spikemon.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.title('Neuron Firing Pattern (Raster Plot)')

# Final State (Average activity in the last 10ms)
plt.subplot(1, 2, 2)
final_activity = np.zeros(N)
for i in range(N):
    final_activity[i] = len(np.where(spikemon.i == i)[0])

plt.imshow(final_activity.reshape(5, 5), cmap='binary')
plt.title('Recalled Pattern (5x5)')
plt.colorbar(label='Spike Count')

plt.tight_layout()
plt.savefig('hopfield_result.png')  # Save the figure
print("Figure saved as 'hopfield_result.png'!")
