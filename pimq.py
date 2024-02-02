import numpy as np
import matplotlib.pyplot as plt
from tools import generate_rand_config, bit_flip

def classical_energy(J: np.ndarray, config: np.ndarray) -> float:
    '''
    Inputs:
    J: the coupling field strengths 
    config: configuration of the system

    Returns:
    the classical energy of the system in configuration config (neglects
    field term)
    '''
    # print("Calculating classical energy")
    # print(J_perp)
    N = len(config)

    # Calculate first sum - could make more efficient
    s_1 = 0 

    for i in range(N):
        for j in range(N):
            s_1 += J[i, j] * config[i] * config[j]

    # Calculate second sum
    s_2 = 0
    # Initialise h as an array of kappas (kappa = 0.5 here)
    h = np.ones((N)) * 0.5
    # Generate h 
    for k in range(N):
        for j in range(N):
            h[k] += -(J[k, j] + J[j, k])

    for i, h_i in enumerate(h):
        s_2 += h_i*config[i]
    
    return (s_1 + s_2)

def get_J_perp(P: int, T: float, G: float) -> float:
    return -1/2 * P * T * np.log(np.tanh(G/(P*T)))

def quantum_energy(J: np.ndarray, configs: np.ndarray, k:int, G: float, 
           T: float) -> float:
    '''
    Inputs:
    J: the coupling field strengths 
    configs: the P configurations of the system
    k: the index of the Trotter slice of interest
    G: magnetic field strength
    T: temperature of the system

    Returns:
    the quantum energy of the system in configuration config (includes field
    term)
    '''
    # print("Calculating quantum energy")
    config = configs[k]
    P = len(configs)

    J_perp = get_J_perp(P, T, G)
    # print(J_perp)
    N = len(config)

    # Calculate first sum - could make more efficient
    s_1 = 0 

    for i in range(N):
        for j in range(N):
            s_1 += J[i, j] * config[i] * config[j]

    s_2 = 0
    # Initialise h as an array of kappas (kappa = 0.5 here)
    h = np.ones((N)) * 0.5
    # Generate h 
    for k in range(N):
        for j in range(N):
            h[k] += -(J[k, j] + J[j, k])

    for i, h_i in enumerate(h):
        s_2 += h_i*config[i]

    # Calculate third sum - might need to change modulus if missing last slice
    s_3 = 0
    config_adj = configs[(k+1) % P]
    for i in range(N):
        s_3 += config[i]*config_adj[i]
    
    return (s_1 + s_2 + J_perp*s_3)

def metropolis(J: np.ndarray, configs: np.ndarray, k: int, j: int, T: float, 
               G: float) -> np.ndarray:
    '''
    Inputs:
    J: coupling field strengths
    configs: the P current configurations of the system
    k: Trotter index of interest
    j: index of the spin to flip
    T: temperature of the system
    G: current magnetic field strength
    P: Trotter number

    Returns:
    config with spin at index j flipped if the energy
    of the new configuration is less
    than that of the old configuration, or if 
    e^(dE/T) > U_random.
    
    Otherwise, returns config
    '''
    P = len(configs)
    # print(f"Metropolis G:{G} P:{P} k:{k} j:{j} T:{T}")

    config = configs[k]

    # Copy the configs into a new array
    configs_new = configs.copy()
    # Flip the spin at index j
    configs_new[k] = bit_flip(configs_new[k], j)

    # Calculate the difference in energy of the original and new configs
    if P == 1:
        dE = classical_energy(J, configs[0]) - classical_energy(J, 
                                                                configs_new[0])
    else:
        dE = quantum_energy(J, configs, k, G, T
                            ) - quantum_energy(J, configs_new, k, G, T)
    # print('dE: ', dE)

    # Define the random variable
    u_random = np.random.randint(2)

    # If dE is positive or config_new satisfies the metropolis condition
    # return the new config, otherwise return the old config
    if dE > 0 or np.exp(dE/T) > u_random:
        return configs_new
    else:
        return configs
    
def pre_anneal(J: np.ndarray, initial_config: np.ndarray, T_start: float,
               T_end: float, T_step: float) -> np.ndarray:
    '''
    Inputs:
    J: coupling field strengths
    initial_config: the n_spins system to anneal
    T_start: the preannleaning start temperature
    T_end: the ambient temperature at the end of preannealing
    T_step: the pre annealing stepsize

    Returns:
    The configuration after the classial pre-annealing
    '''
    config = np.expand_dims(initial_config, axis=0)
    for T in np.linspace(T_start, T_end, int((T_start - T_end)/T_step)):
        for j in range(len(config[0])):
            # Set field term to 0 for classical annealing - could write a
            # faster classical metropolis function
            # Repeat 100 times per spin
            for _ in range(100):
                config = metropolis(J, config, 0, j, T, 0) 

    return config

def quantum_anneal(J: np.ndarray, configs: np.ndarray, G_start: float, 
                   G_end: float, G_step: float, T:float) -> np.ndarray:
    '''
    Inputs:
    J: coupling field strengths
    configs: the n_spins x P system to anneal
    G_start: the initial magnetic field strength
    G_end: the final magnetic field strength
    G_step: the value with which to increment G
    T: temperature of the system
    P: Trotter number

    Returns: 
    The P configurations after simulated quantum annealing with
    quantum Monte Carlo
    '''
    # Define n_spins as the number of qubits
    n_spins = len(configs[0])

    # Simulate over the L values
    for G in np.linspace(G_start, G_end, int((G_start - G_end)/G_step)):
        for k in range(len(configs)):
            for j in range(n_spins):
                configs = metropolis(J, configs, k, j, T, G)

    return configs

def find_solution(J: np.ndarray, configs: np.ndarray, G_end: float, 
                  T:float) -> (np.ndarray, float):
    '''
    Inputs:
    J: coupling field strengths
    configs: the P configurations after quantum annealing
    G_end: the final magnetic field strength
    T: temperature of the system

    Returns: 
    The configuration with the lowest energy
    '''
    E_min = np.inf
    config_min = configs[0]
    for k in range(len(configs)):
        E = quantum_energy(J, configs, k, G_end, T)
        if E < E_min:
            E_min = E
            config_min = configs[k]
    
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
    print('-'*50)
    print("Testing quantum annealing")
    J = np.matrix([[0, 1, 1, 0, 0],
                   [0, 0, 1, 0, 1],
                   [0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])
        
    n_spins = J.shape[0]
    P = 30
    T_start = 3.0
    T_end = 0.05
    T_step = 0.05
    G_start = 3
    G_end = 1e-8
    G_step = 1e-3
    print(quantum_energy(J, np.array([[1, 0, 0, 1, 1]]), 0, G_end, T_end))
    print('-'*60)
    # Generate the initial configuration of n_spins random spins
    random_config = generate_rand_config(n_spins)
    print(f"Random config: {random_config}")

    # Perform a classical preannealing
    print('-'*50)
    print("Simulating Classical Preannealing")
    pre_annealed_config = pre_anneal(J, random_config, T_start, T_end, T_step)
    print('-'*50)
    print(f"Preannealed config: {pre_annealed_config}")

    # Copy the prepared configuration P times
    initial_configs = np.tile(pre_annealed_config, (P,1))

    # Set temperature for quantum annealing
    T = T_end

    # Simulate quantum annealing 
    print('-'*50)
    print("Simulating Quantum Annealing")
    final_configs = quantum_anneal(J, initial_configs, G_start, G_end, G_step, T)

    print(final_configs)

    # Take the lowest energy replica amongst the P Trotter slices
    solution_config, solution_energy = find_solution(J, final_configs, G_end, T)
    
    print('Intial Config: ', random_config)
    print('Preannealed Config: ', pre_annealed_config[0])
    print('Solution Energy: ', solution_energy)
    print('Solution Config: ', solution_config)
    print(np.array_equal(solution_config, np.array([1, 0, 0, 1, 1])))
