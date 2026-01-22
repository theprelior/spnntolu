import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# --- 1. SETUP ---
def create_random_graph(n_nodes=6, probability=0.6):
    np.random.seed(42)  # Fixed seed for reproducibility
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
        # Refractory period
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            self.spiked = 0
            return 0
        
        # Update voltage
        self.voltage = self.voltage * self.alpha + input_current
        
        # --- NEW: ANNEALING (Add noise) ---
        # Higher temperature -> more randomness (noise)
        # This allows the neuron to "accidentally" spike or stay silent
        if temperature > 0:
            noise = np.random.normal(0, temperature)
            self.voltage += noise

        # Spike check
        if self.voltage >= self.threshold:
            self.spiked = 1
            self.voltage = 0.0
            self.refractory_timer = 3  # Rest a bit longer
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

# ANNEALING SETTINGS
initial_temp = 5.0  # Very high at start (lots of noise)
cooling_rate = 0.97 # Cool by 3% per step

print("Simulation starting...")

current_temp = initial_temp

for t in range(steps):
    temperature_history.append(current_temp)
    current_spikes = [n.spiked for n in neurons]
    
    for i in range(num_neurons):
        input_signal = 0
        # Bias
        input_signal += Q[i, i] * 0.5 
        
        # Signals from neighbors (Inhibition)
        for j in range(num_neurons):
            if i != j and current_spikes[j] == 1:
                input_signal -= Q[i, j] * 2.0 

        # Update neuron (pass temperature)
        neurons[i].update(input_signal, current_temp)
        spike_history[i, t] = neurons[i].spiked
    
    # Cooling
    current_temp *= cooling_rate

# --- 4. VISUALIZATION ---
plt.figure(figsize=(14, 5))

# Temperature Plot
plt.subplot(1, 3, 1)
plt.plot(temperature_history, color='orange')
plt.title("Temperature (Annealing) Evolution")
plt.xlabel("Time")
plt.ylabel("Temperature (Noise Level)")

# Spike Plot
plt.subplot(1, 3, 2)
for i in range(num_neurons):
    times = np.where(spike_history[i] == 1)[0]
    plt.scatter(times, [i]*len(times), marker='|', s=100)
plt.title("Neuron Spikes")
plt.xlabel("Time")

# Graph Result
final_activity = np.sum(spike_history[:, -50:], axis=1)  # Last 50 steps
mean_act = np.mean(final_activity)
# Assign colors: above average -> Red, below average -> Blue
node_colors = ['red' if act > mean_act else 'blue' for act in final_activity]

plt.subplot(1, 3, 3)
G = nx.from_numpy_array(A)
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, node_color=node_colors, with_labels=True, font_color='white', node_size=800)
plt.title("Resulting Groups")

plt.tight_layout()
plt.savefig('annealing_result.png')
print("Completed.")
