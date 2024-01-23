import numpy as np
import matplotlib.pyplot as plt
from tools import generate_rand_config, bit_flip

def energy(config: np.ndarray) -> float:
    return

def metropolis(config: np.ndarray, j: int, T: float) -> np.ndarray:
    '''
    Inputs:
    config: the current configuration of a Trotter slice
    j: index of the spin to flip
    T: temperature of the system
    '''
    # Flip the spin at index j
    config_new = bit_flip(config)
    return

def pre_anneal(initial_config: np.ndarray) -> np.ndarray:
    return

def quantum_anneal(configurations: np.ndarray) -> np.ndarray:
    return

def find_solution(configurations: np.ndarray) -> (np.ndarray, float):
    return


if __name__ == "__main__":
    '''
    Test on milestone example

    Maximum Independent Set of graph below

    0 --- 0
     \   /|
       0  |
      /  \|
    0     0

    '''
    M = np.matrix([[0, 1, 1, 0, 0],
                   [0, 0, 1, 0, 1],
                   [0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])
    n_spins = M.shape[0]
    n_trotter_slices = 10
    T = 0
    L_start = 0
    L_end = 0
    dL = 0

    # Generate the initial configuration of n_spins random spins
    random_config = generate_rand_config(n_spins)

    # Perform a classical preannealing
    pre_annealed_config = pre_anneal(random_config)

    # Copy the prepared configuration P times
    initial_configs = np.tile(pre_annealed_config, (n_trotter_slices,1))

    # Simulate quantum annealing 
    final_configs = quantum_anneal(initial_configs)

    # Take the lowest energy replica amongst the P Trotter slices
    solution_config, solution_energy = find_solution(final_configs)

    print(f'Solution Energy: {solution_energy}')
    print('Solution Config: ', solution_config)
    




