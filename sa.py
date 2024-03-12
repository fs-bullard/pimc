import numpy as np
import tsplib95
from copy import deepcopy
import random

from tsp import (
    load_tsp_instance, classical_energy_tsp, generate_random_tour, 
    build_weight_dict, tour_to_list, tour_valid, 
    tour_to_matrix, permute
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

def simulated_annealing(
        problem: tsplib95.models.StandardProblem, 
        T_schedule: list[float]
    ) -> tuple[np.ndarray, int]:
    """Simulates thermal annealing to solve problem. Uses a linear annealing schedule.

    Args:
        problem (tsplib95.models.StandardProblem): the problem of interest
        T_schedule (list[float]): the annealing schedule

    Returns:
        np.ndarray: tour obtained by annealing process
        int: Ising energy of the final tour
    """
    print('-'*50, 'Simulating Annealing', '-'*50)
    print(f"Problem: {problem.name}, N={problem.dimension}")
    print(f'Annealing Steps: {len(T_schedule)}')

    # Generate initial tour
    N = problem.dimension
    tour = generate_random_tour(N)
    
    # or, try with a list tour
    # tour = tour_to_matrix([1, 29, 3, 26, 5, 12, 9, 6, 28, 24, 8, 27, 16, 23, 7, 25, 11, 19, 13, 10, 15, 22, 14, 17, 18, 4, 20, 2, 21])

    weights = build_weight_dict(problem)

    # Calculate energy of this tour
    E = classical_energy_tsp(weights, tour)

    # Loop through annealing temperatures
    for T in T_schedule:
        # print(f'T: {T}')

        for _ in range(N):
            # Generate a permutation of tour TODO: return dE from this to save time
            new_tour = permute(tour) 
            # print(sorted(tour_to_list(new_tour)))
            # assert tour_valid(new_tour), 'Invalid Tour'

            # Calculate energy of new system
            new_E = classical_energy_tsp(weights, new_tour)

            # apply metropolis condition
            dE = E - new_E
            if metropolis(T, dE):
                tour = new_tour
                E = new_E

        # print(f'E: {E}')
        # if random.randint(0, 10) == 5:
        #     print(tour_to_list(tour))

    # assert(tour_valid(tour)), "New tour is invalid"
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

    # Run simulated annealing on problem
    T_0 = 100
    T_f = 0.001
    annealing_steps = 1000
    T_schedule = np.linspace(T_0, T_f, annealing_steps)

    annealed_tour, E = simulated_annealing(problem, T_schedule)

    print([i + 1 for i in tour_to_list(annealed_tour)])
    print('Energy: ', E)
    print('Energy / opt * 100:', (E/opt_energy - 1) * 100)
    print("Optimal energy:", opt_energy)
