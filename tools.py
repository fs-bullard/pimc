import numpy as np
from numpy import random

def generate_rand_config(n_spins: int) -> np.ndarray:
    '''
    Returns a random spin configuration
    for a system of n_spins qubits
    '''
    rng = random.RandomState()
    return np.array([rng.randint(0, 2) for _ in range(n_spins)])

def bit_flip(config: np.ndarray, j: int) -> np.ndarray:
    '''
    Returns config with the spin at element j 
    flipped
    i.e. if config[j] = 0, returns config_new[j] = 1
    or if config[j] = 1, returns config_new[j] = 0
    '''
    config[j] = config[j] ^ 1
    return config
