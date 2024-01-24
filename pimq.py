import numpy as np
import matplotlib.pyplot as plt
from tools import generate_rand_config, bit_flip
from mis import construct_H_ising as H_ising

def J_perp(P: int, T: float, G: float) -> float:
    return -1/2 * P * T * np.log(np.tanh(G/(P*T)))

def energy(config: np.ndarray, H_ising: np.ndarray, G: float) -> float:
    '''
    '''
    H = H_ising
    return np.dot(config,H*config)

def metropolis(config: np.ndarray, j: int, T: float, G: float) -> np.ndarray:
    '''
    Inputs:
    config: the current configuration of a Trotter slice
    j: index of the spin to flip
    T: temperature of the system
    G: current magnetic field strength

    Returns:
    config with spin at index j flipped if the energy
    of the new configuration is less
    than that of the old configuration, or if 
    e^(dE/T) > U_random.
    
    Otherwise, returns config
    '''
    # Flip the spin at index j
    config_new = bit_flip(config, j)

    # Calculate the difference in energy of the original and new configs
    dE = energy(config) - energy(config_new)

    # Define the random variable
    u_random = np.random.randint(2)

    # If dE is positive or config_new satisfies the metropolis condition
    # return the new config, otherwise return the old config
    if dE > 0 or np.exp(dE/T) > u_random:
        return config_new
    else:
        return config
    
def pre_anneal(initial_config: np.ndarray) -> np.ndarray:
    return

def quantum_anneal(configs: np.ndarray, G_start: float, G_end: float,
                    dG: float) -> np.ndarray:
    '''
    Inputs:
    configs: the n_spins x P system to anneal
    G_start: the initial magnetic field strength
    G_end: the final magnetic field strength
    dG: the value with which to increment G

    Returns: 
    The P configurations after simulated quantum annealing with
    quantum Monte Carlo
    '''
    # Define n_spins as the number of qubits
    n_spins = len(configs[0])

    # Simulate over the L values
    for G in range(G_start, G_end, dG):
        for config, k in enumerate(configs):
            for j in range(n_spins):
                configs[k] = metropolis(config, j, T, G)

    return configs

def find_solution(configs: np.ndarray, G_end: float) -> (np.ndarray, float):
    '''
    Inputs:
    configs: the P configurations after quantum annealing
    G_end: the final magnetic field strength
    Returns: 
    The configuration with the lowest energy
    '''
    E_min = np.inf
    config_min = configs[0]
    for config in configs:
        E = energy(config, H_ising, G_end)
        if E < E_min:
            E_min = E
            config_min = config
    
    return config_min, E_min


if __name__ == "__main__":
    '''
    Test on milestone example

    Find the Maximum Independent Set of graph below

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
    P = 30
    T = 0.05
    G_start = 3
    G_end = 1e-8
    dG = 0

    # Generate the initial configuration of n_spins random spins
    random_config = generate_rand_config(n_spins)

    # Perform a classical preannealing
    pre_annealed_config = pre_anneal(random_config)

    # Copy the prepared configuration P times
    initial_configs = np.tile(pre_annealed_config, (P,1))

    # Simulate quantum annealing 
    final_configs = quantum_anneal(initial_configs)

    # Take the lowest energy replica amongst the P Trotter slices
    solution_config, solution_energy = find_solution(final_configs)

    print('Solution Energy: ', solution_energy)
    print('Solution Config: ', solution_config)
    




