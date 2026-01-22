import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# --- 1. SETUP (Data Preparation) ---
def create_random_graph(n_nodes=6, probability=0.6):
    np.random.seed(42)  # Produce the same graph each time
    A = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.random() < probability:
                A[i, j] = 1
                A[j, i] = 1
    return A

def get_Q_from_A(A):
    Q = A.copy()
    for i in range(len(A)):
        row_sum = np.sum(A[i])
        Q[i, i] = -row_sum 
    return Q

# --- 2. ANNEALING-ENABLED NEURON ---
class AnnealingNeuron:
    def __init__(self, neuron_id, alpha=0.9, threshold=0.1):
        self.id = neuron_id
        self.voltage = 0.0
        self.alpha = alpha
        self.threshold = threshold
        self.spiked = 0
        self.refractory_timer = 0

    def update(self, input_current, temperature):
        # Skip update if in refractory period
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            self.spiked = 0
            return 0
        
        # Update voltage (Decay + Input)
        self.voltage = self.voltage * self.alpha + input_current
        
        # --- CRITICAL PART: ADD NOISE (ANNEALING) ---
        # If temperature is high, shake the voltage randomly!
        if temperature > 0:
            noise = np.random.normal(0, temperature)
            self.voltage += noise

        # Spike check
        if self.voltage >= self.threshold:
            self.spiked = 1
            self.voltage = 0.0  # Reset voltage
            self.refractory_timer = 3  # Rest for a bit
        else:
            self.spiked = 0
        
        return self.spiked

# --- 3. SIMULATION ---
num_neurons = 6
A = create_random_graph(num_neurons)
Q = get_Q_from_A(A)
neurons = [AnnealingNeuron(i) for i in range(num_neurons)]

steps = 200
spike_history = np.zeros((num_neurons, steps))
temperature_history = []

# --- ANNEALING PARAMETERS ---
initial_temp = 5.0  # High starting temperature (lots of noise)
cooling_rate = 0.97 # Cooling rate (3% per step)
current_temp = initial_temp

print("Simulation starting...")

for t in range(steps):
    temperature_history.append(current_temp)
    
    current_spikes = [n.spiked for n in neurons]
    
    for i in range(num_neurons):
        input_signal = 0
        
        # Bias (self preference)
        input_signal += Q[i, i] * 0.5 
        
        # Inhibition from neighbors
        for j in range(num_neurons):
            if i != j and current_spikes[j] == 1:
                # If neighbor spiked, suppress me (-2.0)
                input_signal -= Q[i, j] * 2.0 

        # Update neuron with temperature
        neurons[i].update(input_signal, current_temp)
        spike_history[i, t] = neurons[i].spiked
    
    # Cool down a bit each step
    current_temp *= cooling_rate

# --- 4. VISUALIZE RESULTS ---
plt.figure(figsize=(14, 5))

# Temperature plot
plt.subplot(1, 3, 1)
plt.plot(temperature_history, color='orange', linewidth=2)
plt.title("Temperature (Annealing) Evolution")
plt.xlabel("Time")
plt.ylabel("Noise Level")
plt.grid(True)

# Spike plot
plt.subplot(1, 3, 2)
for i in range(num_neurons):
    times = np.where(spike_history[i] == 1)[0]
    plt.scatter(times, [i]*len(times), marker='|', s=100)
plt.title("Neuron Spikes")
plt.xlabel("Time")
plt.ylabel("Neuron ID")

# Graph result (color groups)
# Look at activity over the last 50 steps
final_activity = np.sum(spike_history[:, -50:], axis=1)
mean_act = np.mean(final_activity)

node_colors = []
for act in final_activity:
    if act > mean_act:
        node_colors.append('red')   # High activity (Group A)
    else:
        node_colors.append('blue')  # Low activity (Group B)

plt.subplot(1, 3, 3)
G = nx.from_numpy_array(A)
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, node_color=node_colors, with_labels=True, font_color='white', node_size=800, edge_color='gray')
plt.title("Result: Red vs Blue")

plt.tight_layout()
plt.show()
