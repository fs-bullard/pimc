import numpy as np
import tsplib95
from copy import deepcopy
import random

from tsp import load_tsp_instance, classical_energy_tsp, generate_random_tour, two_opt_move, build_weight_dict
from tsp import build_neighbours_dict, tour_to_list, tour_valid, tour_to_matrix, generate_M_random_neighbours

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

def simulated_annealing(problem: tsplib95.models.StandardProblem, 
                        T_schedule: list[float]) -> np.ndarray:
    """Simulates thermal annealing to solve problem. Uses a linear annealing schedule.

    Args:
        problem (tsplib95.models.StandardProblem): the problem of interest
        T_schedule (list[float]): the annealing schedule

    Returns:
        np.ndarray: tour obtained by annealing process
    """
    print('-'*50, 'Simulating Annealing', '-'*50)
    print(f"Problem: {problem.name}, N={problem.dimension}")

    # Generate initial tour
    N = problem.dimension
    tour = generate_random_tour(N)

    # or, try with a list tour
    # tour = tour_to_matrix([1, 29, 3, 26, 5, 12, 9, 6, 28, 24, 8, 27, 16, 23, 7, 25, 11, 19, 13, 10, 15, 22, 14, 17, 18, 4, 20, 2, 21])

    weights = build_weight_dict(problem)
    neighbours_dict = build_neighbours_dict(N, weights)

    for T in T_schedule:
        print(f'T: {T}')

        # Loop through each city in current tour
        tour_list = tour_to_list(tour)

        # Copy the original tour for this annealing step
        current_tour = deepcopy(tour)

        for i, cur in enumerate(tour_list):
            # print(i, '/', N)
            prev = tour_list[i - 1]
            next = tour_list[(i + 1)%N]
            
            assert(tour_valid(current_tour)), "Current tour is invalid"

            # Calculate energy of this tour
            E = classical_energy_tsp(weights, tour)

            # Generate the neighbourhood of the next city in the tour
            # neighbours = neighbours_dict[cur]
            # neighbours = generate_M_random_neighbours(N, prev, next, N-2)

            # Loop through each of the next city's neighbours
            for neighbour in random.shuffle(tour_list[i - 5:(i + 5)%N]):
            # for neighbour in neighbours:
                # If prev, cur or next are in neighbours ignore them
                if neighbour in [prev, cur, next]:
                    continue

                # Find the city before neighbour
                before_neighbour = tour_list[tour_list.index(neighbour) - 1]

                # Make a two-opt move
                # print(tour_to_list(tour))
                new_tour = two_opt_move(deepcopy(tour), cur, next, neighbour, before_neighbour)
                # print(tour_to_list(new_tour))
                # return

                # Calculate energy of new system
                new_E = classical_energy_tsp(weights, new_tour)

                # apply metropolis condition
                dE = E - new_E
                if metropolis(T, dE):
                    current_tour = new_tour
                    E = new_E

        tour = current_tour
        print(f'E: {E}')
                    
    return tour, E


if __name__ == "__main__":
    # Load problem and optimal tour
    problem_filepath = 'tsplib/bays29.tsp' 
    problem = load_tsp_instance(problem_filepath)
    opt_filepath = 'tsplib/bays29.opt.tour'
    opt = load_tsp_instance(opt_filepath)
    opt_tour = tour_to_matrix(opt.tours[0])
    
    weights = build_weight_dict(problem)

    # Set optimal solution
    opt_energy = classical_energy_tsp(weights, opt_tour)
    print("Optimal energy:", opt_energy)

    # Run simulated annealing on problem
    T_0 = 100
    T_f = 0.001
    annealing_steps = 100
    T_schedule = np.linspace(T_0, T_f, annealing_steps)

    annealed_tour, E = simulated_annealing(problem, T_schedule)

    print([i + 1 for i in tour_to_list(annealed_tour)])
    print('Energy:', E)
    print("Optimal energy:", opt_energy)
