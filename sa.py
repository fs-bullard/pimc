import numpy as np
import tsplib95

from tsp import load_tsp_instance, classical_energy_tsp, generate_random_tour, two_opt_move, generate_M_neighbours, get_next_city

def metropolis(T: float, dE: float) -> bool:
    """Applies the metropolis condition

    Args:
        T (float): temperature
        dE (float): energy variation between candidate tour and current tour

    Returns:
        bool
    """
    if dE < 0:
        return True
    else:
        u = np.random.uniform()
        if np.exp(-dE/T) > u:
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
    # Generate initial tour
    N = problem.dimension
    tour = generate_random_tour(N)

    for T in T_schedule:
        # Count which city in the tour we are on, starting from city 0
        i = 0
        cur = 0
        prev = get_next_city(tour, None, cur)
        while i < N:
            # Get the next city in the tour
            next = get_next_city(tour, prev, cur)

            # Calculate energy of this tour
            E = classical_energy_tsp(problem, tour)

            # Generate the neighbourhood of the next city in the tour
            neighbours = generate_M_neighbours(problem, prev, next)

            # Loop through each of the next city's neighbours
            for neighbour in neighbours:
                # Find the city before neighbour
                before_neighbour = get_next_city(tour, None, neighbour)

                # Make a two-opt move
                new_tour = two_opt_move(tour, cur, before_neighbour, next, neighbour)
    
                # Calculate energy of new system
                new_E = classical_energy_tsp(problem, new_tour)

                # apply metropolis condition
                dE = E - new_E
                if metropolis(T, dE):
                    tour = new_tour
                    E = new_E
            prev = cur
            cur = next
            i += 1
                    
    return tour


if __name__ == "__main__":
    # Load problem and optimal tour
    problem_filepath = 'tsplib/pr1002.tsp'
    problem = load_tsp_instance(problem_filepath)
    opt_filepath = 'tsplib/pr1002.opt.tour'
    opt = load_tsp_instance(opt_filepath)

    # Run simulated annealing on pr1002
    T_0 = 100
    T_f = 0
    annealing_steps = 100
    T_schedule = np.linspace(T_0, T_f, annealing_steps)

    annealed_tour = simulated_annealing(problem, T_schedule)