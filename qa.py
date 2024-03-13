import numpy as np
import tsplib95
from copy import deepcopy
import random

from tsp import (
    load_tsp_instance, classical_energy_tsp, generate_random_tour, 
    build_weight_dict, tour_to_list, tour_valid, 
    tour_to_matrix, quantum_energy_tsp, quantum_permute
    ) 

def metropolis(T: float, dE: float) -> bool:
    """Applies the metropolis condition

    Args:
        T (float): temperature
        dE (float): E - new_E

    Returns:
        bool
    """
    if dE > 0:
        return True
    else:
        u = np.random.uniform()
        if np.exp(dE/T) > u:
            return True
        else:
            return False

def get_J_perp(P: int, T: float, G: float) -> float:
    """Calculates the coupling between the Trotter slices

    J_perp = -PT/2 log(tanh(G/PT))

    Args:
        P (int): Trotter number
        T (float): Temperature
        G (float): Field strength Gamma

    Returns:
        float: coupling in Trotter axis
    """
    return (-P * T / 2) * np.log(np.tanh(G/(P*T)))

def quantum_annealing(
        problem: tsplib95.models.StandardProblem, 
        G_schedule: list[float], 
        T: float, 
        P: int = 30
    ) -> tuple[np.ndarray, int]:
    """Simulates quantum annealing to solve problem. Uses a linear annealing schedule.

    Args:
        problem (tsplib95.models.StandardProblem): the problem of interest
        T_schedule (list[float]): the annealing schedule
        T (float): the temperture of the system
        P (int): the number of Trotter slices to use

    Returns:
        np.ndarray: tour obtained by annealing process
        int: Ising energy of the final tour
    """
    print('-'*50, 'Quantum Annealing', '-'*50)
    print(f"Problem: {problem.name}, N={problem.dimension}")
    print(f'Annealing Steps: {len(G_schedule)}')
    print(f'Trotter Number: ', P)

    # Generate initial tour
    N = problem.dimension
    tour = generate_random_tour(N)

    # Copy over P Trotter slices
    tours = np.tile(tour, (P, 1, 1))

    weights = build_weight_dict(problem)

    # Set initial energy to infinity so first permutation is always accepted
    E =  np.inf

    # Loop through annealing temperatures
    for G in G_schedule:
        # print(f'G: {G}')

        # Calculate J_perp
        J_perp = get_J_perp(P, T, G)

        for _ in range(N):
            for slice in range(P):
                new_tours = quantum_permute(tours, slice) 
                
                # Calculate energy of new system
                new_E = quantum_energy_tsp(weights, new_tours, J_perp)

                # apply Metropolis condition
                dE = E - new_E
                if metropolis(T, dE):
                    tours = new_tours
                    E = new_E

                # print(f'E: {E}')
    
    # Find the Trotter slice with the lowest classical energy
    E = classical_energy_tsp(weights, tours[0])
    tour = tours[0]

    for slice in tours[1:]:
        E_slice = classical_energy_tsp(weights, slice)
        if E_slice < E:
            E = E_slice
            tour = slice

    assert tour_valid(tour), "New tour is invalid"
    return tour, E


if __name__ == "__main__":
    # Load problem and optimal tour
    problem_filepath = 'tsplib/berlin52.tsp' 
    problem = load_tsp_instance(problem_filepath)
    opt_filepath = 'tsplib/berlin52.opt.tour'
    opt = load_tsp_instance(opt_filepath)
    opt_tour = tour_to_matrix(opt.tours[0])
    
    weights = build_weight_dict(problem)

    # Set optimal solution
    opt_energy = classical_energy_tsp(weights, opt_tour)
    print("Optimal energy:", opt_energy)

    # Run quantum annealing on problem
    G_0 = 300
    G_f = 0.001
    PT = 50
    P = 10

    annealing_steps = 200
    G_schedule = np.linspace(G_0, G_f, annealing_steps)

    annealed_tour, E = quantum_annealing(problem, G_schedule, PT/P, P)

    print([i + 1 for i in tour_to_list(annealed_tour)])
    print('Energy: ', E)
    print('Energy / opt * 100:', (E/opt_energy - 1) * 100)
    print("Optimal energy:", opt_energy)
