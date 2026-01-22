from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# 1. Graph: 4 nodes, square
N = 4
Q = np.array([
    [0,1,0,1],
    [1,0,1,0],
    [0,1,0,1],
    [1,0,1,0]
])

# 2. LIF-like neuron
tau = 10*ms
v_rest = 0*mV
v_threshold = 0.5*mV
v_reset = 0*mV

sigma = 0.1*mV/ms  # Corrected units!

eqs = '''
dv/dt = -(v - v_rest)/tau + sigma*xi : volt
'''

G = NeuronGroup(N, eqs, threshold='v>v_threshold', reset='v=v_reset', method='euler')

# 3. Connections (inhibitory for Max-Cut)
S = Synapses(G, G, on_pre='v -= 1*mV')
for i in range(N):
    for j in range(N):
        if Q[i,j] != 0 and i!=j:
            S.connect(i=i, j=j)

# 4. Recording
spikemon = SpikeMonitor(G)
statemon = StateMonitor(G, 'v', record=True)

# 5. Run
run(100*ms)

# 6. Plot
plt.figure(figsize=(6,4))
plt.plot(spikemon.t/ms, spikemon.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Max-Cut Spikes')
plt.show()
