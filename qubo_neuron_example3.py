import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# --- 1. SETTINGS ---
num_neurons = 6
steps = 200
initial_temp = 5.0      # Starting temperature (high noise)
cooling_rate = 0.97     # Each step reduces temperature by 3%

# Create a random graph (fully connected for maximum chaos)
# Seed fixed for reproducibility
np.random.seed(42) 
A = np.ones((num_neurons, num_neurons)) - np.eye(num_neurons)  # Fully connected
Q = A.copy()

# Prepare QUBO matrix (for MaxCut: diagonals negative, others positive)
for i in range(len(A)):
    row_sum = np.sum(A[i])
    Q[i, i] = -row_sum 

# --- 2. ADVANCED QUBO NEURON ---
class AnnealingNeuron:
    def __init__(self, neuron_id):
        self.id = neuron_id
        self.voltage = 0.0
        self.alpha = 0.9      # Voltage decay factor
        self.threshold = 0.1
        self.spiked = 0
        self.refractory_timer = 0

    def update(self, input_current, current_temperature):
        # Refractory period
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            self.spiked = 0
            return 0
        
        # 1. Update voltage (decay + new input)
        self.voltage = self.voltage * self.alpha + input_current
        
        # 2. ANNEALING (Add noise proportional to temperature)
        # High temperature -> high noise -> neuron behaves randomly
        # Low temperature -> neuron behaves more deterministically
        noise = np.random.normal(0, current_temperature)
        self.voltage += noise

        # 3. Spike decision
        if self.voltage >= self.threshold:
            self.spiked = 1
            self.voltage = 0.0      # Discharge
            self.refractory_timer = 2 # Rest for a few steps
        else:
            self.spiked = 0
        
        return self.spiked

# --- 3. SIMULATION ---
neurons = [AnnealingNeuron(i) for i in range(num_neurons)]
spike_history = np.zeros((num_neurons, steps))
temp_history = []  # To plot temperature evolution

current_temp = initial_temp

print("Simulation starting...")

for t in range(steps):
    current_spikes = [n.spiked for n in neurons]
    
    # Record and decay temperature
    temp_history.append(current_temp)
    current_temp *= cooling_rate
    
    for i in range(num_neurons):
        input_signal = 0
        
        # Bias (neuron’s own tendency)
        input_signal += Q[i, i] * 0.1 
        
        # Signal from neighbors (Inhibition)
        for j in range(num_neurons):
            if i != j and current_spikes[j] == 1:
                # Neurons dislike each other (MaxCut), so subtract signal
                input_signal -= Q[i, j] * 0.5 

        # Update neuron (pass current temperature)
        neurons[i].update(input_signal, current_temp)
        spike_history[i, t] = neurons[i].spiked

# --- 4. VISUALIZATION ---
plt.figure(figsize=(12, 8))

# Plot 1: Temperature over time
plt.subplot(2, 2, 1)
plt.plot(temp_history, color='orange', linewidth=2)
plt.title("Temperature (Annealing) Evolution")
plt.ylabel("Noise Level")
plt.xlabel("Time")
plt.grid(True)

# Plot 2: Neuron spikes
plt.subplot(2, 2, 2)
for i in range(num_neurons):
    times = np.where(spike_history[i] == 1)[0]
    plt.scatter(times, [i]*len(times), marker='|', s=50)
plt.title("Neuron Spikes")
plt.ylabel("Neuron ID")
plt.xlabel("Time (Chaos first, order later)")

# Plot 3: Final groups (MaxCut solution)
# Determine groups based on last 50 steps of activity
final_activity = np.sum(spike_history[:, -50:], axis=1)
threshold_activity = np.mean(final_activity)

# High activity -> Red, low activity -> Blue
colors = ['red' if x > threshold_activity else 'blue' for x in final_activity]

plt.subplot(2, 1, 2)
G = nx.from_numpy_array(A)
pos = nx.circular_layout(G)
nx.draw(G, pos, node_color=colors, with_labels=True, font_color='white', node_size=800, edge_color='gray')
plt.title("Result: MaxCut Solution (Different colors = successful cut)")

plt.tight_layout()
plt.show()
