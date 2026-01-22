import numpy as np


def get_number_of_cuts(Q, neuron_states, mode='maximize'):
    """
    Calculates the number of cuts in a QUBO problem based on the given Q matrix and neuron states.

    Parameters:
    Q (numpy.ndarray): The Q matrix representing the QUBO problem.
    neuron_states (numpy.ndarray): The array of neuron states.

    Returns:
    float: The number of cuts in the QUBO problem.

    """
    n_cuts = np.sum(np.sum(Q*(1 - np.outer(neuron_states, neuron_states))))/4

    return n_cuts


def get_mis(Q, neuron_states, mode='maximize'):
    """
    Calculates the maximum independent set (MIS) of a graph represented by a Q matrix.

    Parameters:
    Q (numpy.ndarray): The Q matrix representing the graph.
    neuron_states (numpy.ndarray): The states of the neurons in the graph.

    Returns:
    int: The size of the maximum independent set.

    """
    n = len(neuron_states)
    for i in range(n):
        if neuron_states[i] == 1:
            for j in range(i + 1, n):
                if neuron_states[j] == 1 and Q[i, j] != 0:
                    return 0

    return np.sum(neuron_states == 1)


def compute_qubo_energy(Q, states, mode='maximize'):
    """
    Compute the energy of a QUBO solution.

    Parameters:
        Q (numpy.ndarray): The QUBO matrix.
        x (numpy.ndarray): The QUBO solution vector.
        optimization (str, optional): The optimization type. Default is 'maximize'.

    Returns:
        float or numpy.ndarray: The computed energy of the QUBO solution.

    Raises:
        ValueError: If the optimization parameter is not 'minimize' or 'maximize'.
    """

    # Validate the optimization parameter
    if mode not in ['minimize', 'maximize']:
        raise ValueError("optimization parameter must be 'minimize' or 'maximize'")

    # change -1 states to 0
    x = states.copy()
    x[x == -1] = 0

    # Compute the solution
    energy = x.T @ Q @ x

    # Adjust sign based on optimization
    if mode == 'maximize':
        energy = -energy

    # Check if x is essentially 1D (either truly 1D or a 2D array with one dimension equal to 1)
    is_essentially_1d = x.ndim == 1 or 1 in x.shape

    # Return the appropriate shape
    return energy.item() if is_essentially_1d else np.diag(energy)


EVALUATION_FUNCTIONS = {
    "maxcut": get_number_of_cuts,
    "mis": get_mis
}
