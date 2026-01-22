import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qubo_utils import *


def plot_spikes_energy(spike_matrix, energy_per_time, experiment_dir=None, cmap='viridis', ceil=False, mode='maximize'):
    """
    """

    fig, ax = plt.subplots(nrows=2, sharex=True)

    spike_matrix = np.where(spike_matrix==-1, 0, spike_matrix)
    spike_times = [np.where(row)[0] for row in spike_matrix]
    if cmap == 'black':
        cmap = mcolors.ListedColormap(['black'])
    else:
        cmap = getattr(plt.cm, cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, len(spike_times))]

    ax[0].eventplot(spike_times, colors=colors)
    ax[1].plot(range(spike_matrix.shape[1]), energy_per_time, c='k')

    if mode == 'maximize':
        max_index = np.argmax(energy_per_time)
        ax[1].text(max_index, max(energy_per_time), 'Max: {}'.format(max(energy_per_time)), fontsize=8, color='red',
                   bbox=dict(facecolor='white', alpha=0.5))
    else:
        min_index = np.argmin(energy_per_time)
        ax[1].text(min_index, min(energy_per_time), 'Min: {}'.format(min(energy_per_time)), fontsize=8, color='red',
                   bbox=dict(facecolor='white', alpha=0.5))

    if ceil:
        ax[1].set_ylim(0, math.ceil(max(energy_per_time)/100)*100)

    ax[0].spines[['right', 'top']].set_visible(False)
    ax[1].spines[['right', 'top']].set_visible(False)

    ax[0].set_ylabel('QUBO')
    ax[1].set_ylabel('Energy')
    ax[1].set_xlabel('Timesteps x 100')

    if experiment_dir is not None:
        plt.savefig(experiment_dir.joinpath('spikes_energy'), dpi=300)
        plt.clf()
        plt.cla()
        plt.close()
    else:
        plt.show()


def plot_multirun_energy(results_dir):
    """
    Plot mean and standard deviation of energy_per_time from multiple runs
    """

    _, all_energy = np.array(extract_energies(results_dir))
    mean_energies = np.mean(all_energy, axis=0)
    std_energies = np.std(all_energy, axis=0)

    plt.plot(mean_energies, label='Mean Energy')
    plt.fill_between(range(len(mean_energies)), mean_energies - std_energies,
                     mean_energies + std_energies, color='gray', alpha=0.5,
                     label='Standard Deviation')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Mean and Std Dev of Energy over Time')
    plt.legend()
    plt.show()


def plot_average_maximum_energy(results_dir):
    max_energies, _ = np.array(extract_energies(results_dir))

    # Calculate mean and standard deviation
    mean_value = np.mean(max_energies)
    std_value = np.std(max_energies)

    # Create bar plot
    plt.bar(['Mean'], [mean_value], yerr=[std_value], color='blue', capsize=10, width=0.02)

    print('Mean value: {}'.format(mean_value))
    print('Standard Deviation: {}'.format(std_value))
    # Set title and labels
    plt.title('Average maximum energy over {} runs'.format(len(max_energies)))
    plt.ylabel('Threshold: 0.1')
    plt.show()


def plot_JO_bitstrings_as_events(bitstrings):
    num_variables = len(bitstrings[0])

    # Convert bitstrings to a list of event times
    events = [[] for _ in range(num_variables)]
    for index, bitstring in enumerate(bitstrings):
        for bit_index, bit in enumerate(bitstring):
            if bit == 1:
                events[bit_index].append(index)  # Event time is the index of the bitstring

    # Create an event plot
    plt.eventplot(events, orientation='horizontal')
    plt.show()


def draw_graph_from_matrix(matrix):
    G = nx.Graph()
    num_nodes = len(matrix)

    # Add nodes to the graph
    for node in range(1, num_nodes + 1):
        G.add_node(node)

    # Add edges to the graph based on the connectivity matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            if matrix[i][j] == 1:
                G.add_edge(i + 1, j + 1)

    # Draw the graph
    pos = nx.spring_layout(G)  # You can choose a different layout if you prefer
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=12)
    plt.title("Graph from Connectivity Matrix")
    plt.show()


def draw_partitioned_graph(A, solution):
    G = nx.Graph()
    num_nodes = len(A)

    # Add nodes to the graph
    for node in range(1, num_nodes + 1):
        G.add_node(node)

    # Add edges to the graph based on the adjacency matrix
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if A[i, j] == 1:
                G.add_edge(i + 1, j + 1)  # Adjust indices for 1-based node numbering

    pos = {}
    left_y = 0
    right_y = 0

    for i, group in enumerate(solution, start=1):
        if group == 0:
            # Place this node on the left
            pos[i] = (-1, left_y)
            left_y += 1
        else:
            # Place this node on the right
            pos[i] = (1, right_y)
            right_y += 1

    # Prepare node colors
    color_map = ['red' if group == 0 else 'blue' for group in solution]

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=500, font_size=16)
    plt.show()
