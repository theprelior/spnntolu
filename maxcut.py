#%pip install networkx
import logging
logging.basicConfig(level=logging.DEBUG)

import os
import sys

import numpy as np
import pandas as pd

current_dir = os.getcwd()
src_dir = os.path.normpath(os.path.join(current_dir, './'))
data_dir = os.path.normpath(os.path.join(current_dir, './data/maxcut/'))
if src_dir not in sys.path:
    sys.path.append(src_dir)


#from qubo_spinnaker_multirun import *

from qubo_utils import *
from qubo_visualization import *
from evaluations import *
from spinnaker2 import snn, hardware, helpers
from spinnaker2.experiment_backends import BackendSettings
from spinnaker2.experiment_backends.backend_settings import ROUTING
from spinnaker2.experiment_backends import ExperimentBackendType


S2IP = "192.168.1.17" # SpiNNaker2 Board IP Adress. TODO: update


Q = np.load(os.path.join(data_dir, 'toy5_Q.npy'))
print(Q)
draw_graph_from_matrix(Q)



neuron_params = {'threshold': 0.1, 'alpha_decay': 0.95, 'reset': 'reset_to_v_reset', 'v_reset': 0.0}


indices = np.nonzero(Q)
conns_qubo = np.column_stack((indices[0], indices[1], Q[indices], np.zeros_like(Q[indices])))
conns_qubo_df = pd.DataFrame(conns_qubo, columns=["pre-neuron", "post-neuron", "weight", "delay"])
print(conns_qubo_df)


qubo_pop = snn.Population(size=Q.shape[0], neuron_model="qubo_neuron", params=neuron_params,
                          record=['spikes'])


proj_qubo = snn.Projection(pre=qubo_pop, post=qubo_pop, connections=conns_qubo)


net = snn.Network("QUBO Network")
net.add(qubo_pop, proj_qubo)



# hw = hardware.SpiNNaker2Chip(eth_ip=S2IP)
# hw = hardware.SpiNNcloud48NodeBoard(stm_ip=S2IP)
# hw = hardware.SpiNNcloud48NodeBoard(stm_ip="192.168.1.2")

hw = hardware.SpiNNcloud48NodeBoard()



timesteps = 300
hw.run(net, timesteps, debug=True)



spike_times = qubo_pop.get_spikes()
spike_matrix = helpers.spike_times_to_matrix(spike_times, n_timesteps=timesteps)


energy_per_time = [compute_qubo_energy(Q, spike_matrix.T[i]) for i in  range(spike_matrix.T.shape[0])]
max_energy_per_time = max(energy_per_time)
index_best_solution = int(np.argmax(energy_per_time))


plot_spikes_energy(spike_matrix, energy_per_time, experiment_dir=None, cmap='black')


best_solution = spike_matrix.T[index_best_solution]
draw_partitioned_graph(Q, best_solution)



A = np.load(os.path.join(data_dir, 'toy8_Q.npy'))
print(A)
draw_graph_from_matrix(A)
Q = get_Q_from_A(A)
print(Q)
indices = np.nonzero(Q)
conns_qubo = np.column_stack((indices[0], indices[1], Q[indices], np.zeros_like(Q[indices])))
conns_qubo_df = pd.DataFrame(conns_qubo, columns=["pre-neuron", "post-neuron", "weight", "delay"])
print(conns_qubo_df)
neuron_params = {'threshold': 0.15, 'alpha_decay': 0.5, 'reset': 'reset_to_v_reset', 'v_reset': 0.0}
qubo_pop = snn.Population(size=Q.shape[0], neuron_model="qubo_neuron", params=neuron_params,
                          record=['spikes'])
proj_qubo = snn.Projection(pre=qubo_pop, post=qubo_pop, connections=conns_qubo)
net = snn.Network("QUBO Network")
net.add(qubo_pop, proj_qubo)

# hw = hardware.SpiNNaker2Chip(eth_ip=S2IP)
hw = hardware.SpiNNcloud48NodeBoard(eth_ip=S2IP)

timesteps = 300
hw.run(net, timesteps, debug=False)

spike_times = qubo_pop.get_spikes()
spike_matrix = helpers.spike_times_to_matrix(spike_times, n_timesteps=timesteps)
energy_per_time = [compute_qubo_energy(A, spike_matrix.T[i]) for i in  range(spike_matrix.T.shape[0])]
max_energy_per_time = max(energy_per_time)
index_best_solution = int(np.argmax(energy_per_time))



plot_spikes_energy(spike_matrix, energy_per_time, experiment_dir=None, cmap='black')



best_solution = spike_matrix.T[index_best_solution]
draw_partitioned_graph(A, best_solution)




n = 50
upper_tri = np.triu(np.random.randint(0, 2, size=(n, n)), k=1)
A = upper_tri + upper_tri.T
#print(A)
draw_graph_from_matrix(A)
Q = get_Q_from_A(A)


neuron_params = {'threshold': 0.5, 'alpha_decay': 0.75, 'reset': 'reset_to_v_reset', 'v_reset': 0.0}
#print(Q)
indices = np.nonzero(Q)
conns_qubo = np.column_stack((indices[0], indices[1], Q[indices], np.zeros_like(Q[indices])))
conns_qubo_df = pd.DataFrame(conns_qubo, columns=["pre-neuron", "post-neuron", "weight", "delay"])
print(conns_qubo_df)
qubo_pop = snn.Population(size=Q.shape[0], neuron_model="qubo_neuron", params=neuron_params,
                          record=['spikes'])
proj_qubo = snn.Projection(pre=qubo_pop, post=qubo_pop, connections=conns_qubo)
net = snn.Network("QUBO Network")
net.add(qubo_pop, proj_qubo)

# hw = hardware.SpiNNaker2Chip(eth_ip=S2IP)
hw = hardware.SpiNNcloud48NodeBoard(eth_ip=S2IP)

timesteps = 300
hw.run(net, timesteps, debug=False)

spike_times = qubo_pop.get_spikes()
spike_matrix = helpers.spike_times_to_matrix(spike_times, n_timesteps=timesteps)
energy_per_time = [compute_qubo_energy(Q, spike_matrix.T[i]) for i in  range(spike_matrix.T.shape[0])]
max_energy_per_time = max(energy_per_time)
index_best_solution = int(np.argmax(energy_per_time))




plot_spikes_energy(spike_matrix, energy_per_time, experiment_dir=None, cmap='black', ceil=True)