import numpy as np
import matplotlib.pyplot as plt

# --- STEP 1: Define the Problem (The Graph) ---
# Let's define a simple Triangle Graph (Nodes 0, 1, 2 are all connected)
# Ideally, we want to split them, but we can't perfectly split a triangle into two groups!
# This is a "frustrated" system.

# Adjacency Matrix (A)
# 1 means connected, 0 means not connected
A = np.array([
    [0, 1, 1], # Node 0 connects to 1 and 2
    [1, 0, 1], # Node 1 connects to 0 and 2
    [1, 1, 0]  # Node 2 connects to 0 and 1
])

print("Adjacency Matrix (A):\n", A)

# --- STEP 2: Convert to QUBO Matrix (The Math) ---

def get_Q_from_A(A):
    Q = A.copy()
    for i in range(len(A)):
        # In your file, Q[i,i] is the negative sum of the row weights
        row_sum = np.sum(A[i])
        Q[i, i] = -row_sum
    return Q

Q = get_Q_from_A(A)
print("\nQUBO Matrix (Q):\n", Q)

# --- STEP 3: The Energy Function ---
# f(x) = x.T * Q * x
def calculate_energy(x, Q):
    # x is a vector of binary variables (0 or 1)
    return x.T @ Q @ x

# --- STEP 4: The Solver (Simulated Annealing) ---
# This mimics what the Hopfield network/Annealer does
def simple_annealing_solver(Q, steps=100):
    n_nodes = Q.shape[0]
    
    # Random initial state (e.g., [0, 1, 0])
    current_state = np.random.randint(2, size=n_nodes)
    current_energy = calculate_energy(current_state, Q)
    
    print(f"\nInitial State: {current_state}, Energy: {current_energy}")
    
    best_state = current_state.copy()
    best_energy = current_energy
    
    # Annealing Loop
    for i in range(steps):
        # 1. Flip a random bit (change a 0 to 1, or 1 to 0)
        new_state = current_state.copy()
        node_to_flip = np.random.randint(n_nodes)
        new_state[node_to_flip] = 1 - new_state[node_to_flip]
        
        # 2. Calculate new energy
        new_energy = calculate_energy(new_state, Q)
        
        # 3. Decision Rule (Metropolis)
        # If energy is lower, accept it. 
        # If energy is higher, accept it with a small probability (Simulated Temperature)
        if new_energy < current_energy:
            current_state = new_state
            current_energy = new_energy
        
        # Keep track of best
        if current_energy < best_energy:
            best_energy = current_energy
            best_state = current_state.copy()

    return best_state, best_energy

# --- Run the Solver ---
solution_state, solution_energy = simple_annealing_solver(Q)

print("\n--- Final Result ---")
print(f"Solution Vector x: {solution_state}")
print(f"Minimum Energy: {solution_energy}")

# Visualization (Simple Text)
group_0 = [i for i, x in enumerate(solution_state) if x == 0]
group_1 = [i for i, x in enumerate(solution_state) if x == 1]
print(f"Group A Nodes: {group_0}")
print(f"Group B Nodes: {group_1}")