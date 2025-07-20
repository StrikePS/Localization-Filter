# Pranjal Sinha

# # plot_utils.py

import numpy as np
import matplotlib.pyplot as plt

def plot_belief_evolution(belief_history):
    """
    Plots the belief evolution as a 2D colored field.
    Each row is the belief distribution at one time step.
    """
    belief_array = np.array(belief_history)

    plt.figure(figsize=(10, 6))
    plt.imshow(belief_array, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Belief')

    # Set x-ticks with labels starting from 1 instead of 0
    num_positions = belief_array.shape[1]
    tick_positions = np.arange(num_positions)
    tick_labels = tick_positions + 1  # Start labels from 1

    plt.xticks(ticks=tick_positions, labels=tick_labels)

    plt.xlabel('World Position (Cell Index)')
    plt.ylabel('Time Step')
    plt.title('Belief Evolution Over Time')
    plt.tight_layout()
    plt.show()
