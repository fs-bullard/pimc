import numpy as np
import tsplib95
from copy import deepcopy

from tsp import load_tsp_instance, classical_energy_tsp, generate_random_tour, two_opt_move
from tsp import generate_M_neighbours, get_next_city, tour_to_list, tour_valid, tour_to_matrix

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

    for T in T_schedule:
        print(f'T: {T}')

        # Loop through each city in current tour
        tour_list = tour_to_list(tour)
        # print(f'Tour at T={T}', tour_list)
        # print(f'Sorted tour at T={T}', sorted(tour_list))

        # Copy the original tour for this annealing step
        current_tour = deepcopy(tour)
        # print(tour_list)

        for i, cur in enumerate(tour_list):
            # print(i, '/', N)
            prev = tour_list[i - 1]
            next = tour_list[(i + 1)%N]

            # print('Prev: ', prev)
            # print('Cur: ', cur)
            # print('Next:' , next)
            
            assert(tour_valid(current_tour)), "Current tour is invalid"

            # Calculate energy of this tour
            E = classical_energy_tsp(problem, tour)

            # Generate the neighbourhood of the next city in the tour
            neighbours = generate_M_neighbours(problem, prev, next)
            # print(f'Neighbours of {next} ', neighbours)

            # Loop through each of the next city's neighbours
            for neighbour in neighbours:
                # Find the city before neighbour
                before_neighbour = tour_list[tour_list.index(neighbour) - 1]
                # print('neighbour:', neighbour)
                # print('before neighbour:', before_neighbour)

                # Make a two-opt move
                new_tour = two_opt_move(deepcopy(tour), cur, next, before_neighbour, neighbour)
                # print(tour_to_list(new_tour))

                # Calculate energy of new system
                new_E = classical_energy_tsp(problem, new_tour)

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
    problem_filepath = 'tsplib/berlin52.tsp' 
    problem = load_tsp_instance(problem_filepath)
    opt_filepath = 'tsplib/berlin52.opt.tour'
    opt = load_tsp_instance(opt_filepath)
    opt_tour = tour_to_matrix(opt.tours[0])


    # Calculate optimal solution
    opt_energy = classical_energy_tsp(problem, opt_tour)
    # print("Optimum energy:", opt_energy)

    # Run simulated annealing on problem
    T_0 = 200
    T_f = 1e-5
    annealing_steps = 1000
    T_schedule = np.linspace(T_0, T_f, annealing_steps)

    annealed_tour, E = simulated_annealing(problem, T_schedule)

    print(tour_to_list(annealed_tour))
    print('Energy:', E)