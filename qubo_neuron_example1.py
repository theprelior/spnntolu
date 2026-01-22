import numpy as np
import matplotlib.pyplot as plt

class ToyQuboNeuron:
    def __init__(self, name, alpha=0.9, threshold=1.0):
        self.name = name
        self.voltage = 0.0      # Accumulated energy inside the neuron
        self.alpha = alpha      # Decay factor
        self.threshold = threshold
        self.spiked = 0         # Current state (0 or 1)

    def step(self, input_current):
        # 1. Decay the old voltage a bit
        self.voltage = self.voltage * self.alpha
        
        # 2. Add incoming signal
        self.voltage += input_current
        
        # 3. Check if threshold is crossed
        if self.voltage >= self.threshold:
            self.spiked = 1             # Fire! (state becomes 1)
            self.voltage = 0.0          # Reset voltage
            print(f"{self.name} FIRED! (Sending signal)")
        else:
            self.spiked = 0             # Silent (state 0)
        
        return self.spiked

# --- SIMULATION ---

# Create two neurons
n1 = ToyQuboNeuron("Neuron A")
n2 = ToyQuboNeuron("Neuron B")

# Connection weight (W): Make it negative so they inhibit each other.
# If Neuron A fires, it will reduce Neuron B's voltage.
w_inhibition = -2.0 

# Constant input (bias): Both actually "want" to fire
bias_current = 0.6 

history_v1 = []
history_v2 = []

print("--- Simulation Starts ---")
for t in range(20):
    print(f"\nTime {t}:")
    
    # Neurons look at EACH OTHER'S PREVIOUS state
    input_to_n1 = bias_current + (n2.spiked * w_inhibition)
    input_to_n2 = bias_current + (n1.spiked * w_inhibition)
    
    # Step the neurons forward
    s1 = n1.step(input_to_n1)
    s2 = n2.step(input_to_n2)
    
    # Record for plotting
    history_v1.append(n1.voltage)
    history_v2.append(n2.voltage)
    
    print(f"   States -> A: {s1}, B: {s2}")
    print(f"   Voltages -> A: {n1.voltage:.2f}, B: {n2.voltage:.2f}")

# Plot the results
plt.figure(figsize=(10, 4))
plt.plot(history_v1, label='Neuron A Voltage', marker='o')
plt.plot(history_v2, label='Neuron B Voltage', marker='x')
plt.axhline(y=1.0, color='r', linestyle='--', label='Threshold')
plt.title('Conflict Between Two QUBO Neurons (MaxCut)')
plt.xlabel('Time Step')
plt.ylabel('Voltage')
plt.legend()
plt.grid(True)
plt.show()
