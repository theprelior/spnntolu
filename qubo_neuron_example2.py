import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# --- STEP 1: HELPER FUNCTIONS  ---

def create_random_graph(n_nodes=5, probability=0.5):
    """Create a random adjacency matrix."""
    # Make a symmetric matrix (graph is undirected)
    A = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.random() < probability:
                A[i, j] = 1
                A[j, i] = 1
    return A

def get_Q_from_A(A):
    """Convert adjacency matrix A to QUBO matrix Q (from your original file)."""
    Q = A.copy()
    for i in range(len(A)):
        # Diagonal: negative sum of the row
        row_sum = np.sum(A[i])
        Q[i, i] = -row_sum 
    return Q

# --- STEP 2: VIRTUAL HARDWARE  ---

class NetworkNeuron:
    def __init__(self, neuron_id, alpha=0.9, threshold=0.1):
        self.id = neuron_id
        self.voltage = 0.0
        self.alpha = alpha
        self.threshold = threshold
        self.spiked = 0
        self.refractory_timer = 0  # Rest period after firing

    def update(self, input_current):
        # Do nothing if neuron is in refractory period
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            self.spiked = 0
            return 0
        
        # Decay voltage and add new input
        self.voltage = self.voltage * self.alpha + input_current
        
        # Add noise (important for annealing effect!)
        noise = np.random.normal(0, 0.05) 
        self.voltage += noise

        if self.voltage >= self.threshold:
            self.spiked = 1
            self.voltage = 0.0  # Reset
            self.refractory_timer = 2  # Rest for 2 steps
        else:
            self.spiked = 0
        
        return self.spiked

# --- STEP 3: RUN THE EXPERIMENT ---

# A) Create the graph
num_neurons = 6
A = create_random_graph(num_neurons)
Q = get_Q_from_A(A)

print("Adjacency Matrix (A):\n", A)
print("\nQUBO Matrix (Q - neurons liking/disliking each other):\n", Q)

# B) Create neurons
neurons = [NetworkNeuron(i) for i in range(num_neurons)]

# C) Simulation loop
steps = 100
spike_history = np.zeros((num_neurons, steps))

print("\n--- Simulation Starts ---")

# For each time step:
for t in range(steps):
    current_spikes = [n.spiked for n in neurons]  # Previous spike states
    
    for i in range(num_neurons):
        # Compute total input to neuron i
        # Formula: Input = sum(Other neuron state * Q[i, j])
        input_signal = 0
        
        # Bias (diagonal of Q, neuron’s own tendency)
        input_signal += Q[i, i] * 0.5  # Bias effect
        
        # Signals from other neurons (off-diagonal)
        for j in range(num_neurons):
            if i != j and current_spikes[j] == 1:
                # Receive signal proportional to Q weight
                # Note: For MaxCut we usually want anti-correlation,
                # so positive Q means penalty -> neurons should inhibit each other
                input_signal += Q[i, j] * 2.0 

        neurons[i].update(input_signal)
        spike_history[i, t] = neurons[i].spiked

# --- STEP 4: VISUALIZATION ---

plt.figure(figsize=(12, 6))

# Spike plot (which neuron fired when?)
plt.subplot(1, 2, 1)
for i in range(num_neurons):
    times = np.where(spike_history[i] == 1)[0]
    plt.scatter(times, [i]*len(times), marker='|', s=100)
plt.title("Neuron Firing Times")
plt.ylabel("Neuron ID")
plt.xlabel("Time")
plt.yticks(range(num_neurons))

# Resulting graph (Red vs Blue groups)
# Neurons that fired more in the last 10 steps -> Group 1 (Red), others -> Group 0 (Blue)
final_activity = np.sum(spike_history[:, -20:], axis=1)
group_colors = ['red' if x > np.mean(final_activity) else 'blue' for x in final_activity]

plt.subplot(1, 2, 2)
G = nx.from_numpy_array(A)
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color=group_colors, with_labels=True, font_color='white', node_size=800)
plt.title("Result: Red and Blue Groups (MaxCut)")

plt.tight_layout()
plt.show()
