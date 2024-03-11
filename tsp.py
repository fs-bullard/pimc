import tsplib95
import numpy as np
import random
import heapq
from copy import deepcopy

def load_tsp_instance(filepath: str):
    return tsplib95.load(filepath)

def get_tour_dist(problem: tsplib95.models.StandardProblem, tour: list):
    '''
    Return the total weight of a given tour
    '''
    return problem.trace_tours([tour])[0]

def get_weight(problem: tsplib95.models.StandardProblem, i: int, j: int) -> int:
    '''
    Returns the weight between nodes i and j
    '''
    return problem.get_weight(i + 1, j + 1)

def build_weight_dict(problem: tsplib95.models.StandardProblem) -> dict:
    """Returns a dictionary of the weight of each edge precomputed

    Args:
        problem (tsplib95.models.StandardProblem): TSP of interest

    Returns:
        dict: Dictionary of weights
    """
    N = problem.dimension

    weights = {}
    for i in range(N):
        for j in range(N):
            weights[(i, j)] = get_weight(problem, i, j)

    return weights

def generate_random_tour(N: int) -> np.ndarray:
    """Generates a random valid symmetric tour matrix 

    Args:
        N (int): number of nodes in problem

    Returns:
        np.ndarray: random generated tour
    """
    tour = np.zeros((N, N))
    nodes = [i for i in range(1, N)]

    cur = 0
    while nodes:
        next = nodes.pop(random.randrange(len(nodes)))
        tour[cur][next] = 1
        cur = next
    tour[cur][0] = 1

    return tour + tour.T

def tour_to_matrix(tour: tsplib95.fields.ToursField) -> np.ndarray:
    """Converts tour to symmetric matrix
    """
    N = len(tour)
    matrix = np.zeros((N, N))
    for i, cur in enumerate(tour):
        next = tour[(i + 1) % N]

        matrix[cur - 1][next - 1] = 1

    return matrix + matrix.T

def tour_to_list(tour: np.ndarray) -> list[int]:
    """Converts matrix representation of tour to a list of cities

    Args:
        tour (np.ndarray): tour as a matrix

    Returns:
        list[int]: list of cities
    """
    cur = 0
    prev = get_next_city(tour, None, cur)
    tour_list = []

    i = 0
    while i < len(tour):
        tour_list.append(cur)
        cur = get_next_city(tour, prev, cur)
        prev = tour_list[-1]

        i += 1

    return tour_list

def tour_valid(tour: np.ndarray) -> bool:
    """Returns True if tour is valid.
    Valid tour has two 1s in each row/column,
    is symmetric and has every diagonal element 0.

    Args:
        tour (np.ndarray)

    Returns:
        bool
    """
    # Check tour is symmetric
    if not np.array_equal(tour, tour.T):
        return False
    # Check diagonals are 0
    elif np.trace(tour) != 0:
        return False
    # Check each row and column has two 1s
    for i in range(len(tour)):
        if np.sum(tour[i]) != 2:
            return False
        elif np.sum(tour[:][i]) != 2:
            return False
        
    return True

def generate_M_neighbours(N:int, weights: dict, cur: int, M: int = 20) -> list:
    """Generates a list of the M nearest neighbours to city cur

    Args:
        N (int): Number of cities in the problem
        weights (dict): weights[(i, j)] gives the weight of edge (i, j)
        cur (int): current city
        M (int): number of neighbours to generate

    Returns:
        list: list of M nearest neighbours to city cur
    """
    neighbours = []

    for i in range(N):
        neighbours.append((i, weights[(cur, i)]))

     # Use a heap to efficiently get the M smallest elements
    nearest_neighbours = heapq.nsmallest(M, neighbours, key=lambda x: x[1])

    # Extract the indices from the tuples
    return [i for i, _ in nearest_neighbours]

def build_neighbours_dict(N: int, weights: dict, M: int = 20) -> dict:
    """Returns a dictionary of the M nearest neighbours to each city

    Args:
        N (int): number of cities in the problem
        weights (dict): weights[(i, j)] gives the weight of edge (i, j)

    Returns:
        dict
    """
    neighbours_dict = {}

    for i in range(N):
        neighbours_dict[i] =  generate_M_neighbours(N, weights, i, M)

    return neighbours_dict

def generate_M_random_neighbours(N:int, prev: int, next: int, M = 25) -> list:
    """Generates a list of M random cities excluding prev and next

    Args:
        N (int): number of cities in problem
        prev (int): city before the current city
        next (int): city we are looking to replace (after current city)
        M (int): number of neighbours to generate

    Returns:
        list: list of M nearest neighbours to city next (excluding prev)
    """
    candidates = [i for i in range(N) if i != prev and i != next]

    return random.sample(candidates, M)
    

def two_opt_move(tour: np.ndarray, i: int, j: int, k:int, l: int) -> np.ndarray:
    """Conducts a two opt move swapping connections (i<->j) and (k<->l) to (i<->l) and (j<->k)

    Args:
        tour (np.ndarray)
        i (int)
        j (int)
        k (int)
        l (int)

    Returns:
        np.ndarray: tour with two-opt move applied
    """
    # Delete old links
    tour[i][j] = 0
    tour[j][i] = 0
    tour[k][l] = 0
    tour[l][k] = 0

    # Construct new links
    tour[i][l] = 1
    tour[l][i] = 1
    tour[j][k] = 1
    tour[k][j] = 1
    
    return tour

def get_next_city(tour: np.ndarray, prev: int | None, cur: int) -> int:
    """Returns the next city in the tour

    Args:
        tour (np.ndarray): current tour
        prev (int / None): previous city or None if happy with any
        cur (int): current city (between 0 and N-1)

    Returns:
        int: next city in tour
    """
    # Find column of cur city
    col = tour[cur]

    # Find which cities are connected to cur
    args = np.argwhere(col)

    if prev != None:
        # Check prev is in args
        assert prev in args, "ERROR getting next city, prev incorrect"

        # Return the city that isn't prev
        for arg in args:
            if arg[0] != prev:
                return arg[0]
    else:
        # Just return the first city
        return args[0][0]

def classical_energy_tsp(weights: dict, tour: np.ndarray):
    """Returns the total weight of a given tour

    Args:
        weights (dict): weights[(i, j)] gives the weight of edge (i, j)
        tour (np.ndarray): tour represented by a matrix

    Returns:
        int: total weight of tour
    """
    N = len(tour)
    H = 0
    for i in range(N):
        for j in range(N):
            if tour[i][j]:
                H += weights[(i, j)]

    return H / 2

def quantum_energy_tsp():
    return

if __name__ == '__main__':
    # Load problem and optimal tour
    problem_filepath = 'tsplib/pr1002.tsp'
    problem = load_tsp_instance(problem_filepath)
    opt_filepath = 'tsplib/pr1002.opt.tour'
    opt = load_tsp_instance(opt_filepath)
    N = problem.dimension

    # Test building the weights dicitonary
    weights = build_weight_dict(problem)

    i = random.randint(0, N - 1)
    j = random.randint(0, N - 1)
    if weights[(i, j)] == get_weight(problem, i, j):
        print("TEST PASSED: build_weights_dict worked successfully")
    else: 
        print("TEST FAILED: build_weights_dict was unsuccessful")

    # Check classical energy gives correct weight
    opt_weight = get_tour_dist(problem, opt.tours[0])
    opt_tour = tour_to_matrix(opt.tours[0])

    print(f"Problem {problem.name} loaded with {N} cities and optimal weight {opt_weight}")

    if opt_weight == classical_energy_tsp(weights, opt_tour):
        print("TEST PASSED: Classical energy function returns correct optimal weight")
    else:
        print("TEST FAILED: Classical energy function returns incorrect optimal weight")

    rand_tour = generate_random_tour(N)
    rand_weight = classical_energy_tsp(weights, rand_tour)

    # Check that random tour is valid - each node is visited once only
    valid = True
    for i, row in enumerate(rand_tour):
        s = 0
        for el in row:
            s += el
        if s != 2: 
            valid = False

    if valid:
        print("TEST PASSED: Random generated tour is valid")
    else:
        print("TEST FAILED: Random generated tour invalid")

    print(f'Random tour has weight: {rand_weight}, which is {100*rand_weight/opt_weight}% of the optimal weight')

    # Test two_opt_move
    test_tour = np.asarray(
        [[0, 0, 0, 1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 1, 0, 0, 0]])
    test_tour += test_tour.T
    
    two_opt_tour = np.asarray(
        [[0, 0, 0, 1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0]])
    two_opt_tour += two_opt_tour.T

    if np.array_equal(two_opt_move(test_tour, 1, 4, 2, 3), two_opt_tour):
        print("TEST PASSED: two_opt_move was successful")
    else:
        print(" ----- TEST FAILED: two_opt_move was unsuccessful")
        
    # Test get_next_city with a previous city
    next_city = get_next_city(two_opt_tour, 0, 3)
    if next_city == 4:
        print("TEST PASSED: get_next_city was successful")
    else:
        print(" ----- TEST FAILED: get_next_city was unsuccessful")

    # Test get_next_city without a previous city
    next_city = get_next_city(two_opt_tour, None, 3)
    if next_city == 0:
        print("TEST PASSED: get_next_city was successful")
    else:
        print(" ----- TEST FAILED: get_next_city was unsuccessful")

    # Check get_next_city can reproduce the correct tour for opt_tour
    cur = 0
    prev = 75
    tour = []
    i = 0
    while i < N:
        tour.append(cur + 1)
        next = get_next_city(opt_tour, prev, cur)
        prev = cur
        cur = next

        i += 1
    
    if opt.tours[0] == tour:
        print("TEST PASSED: get_next_city successfully reproduced the opt_tour")
    else:
        print(" ----- TEST FAILED: get_next_city was unsuccessful in reproducing the opt_tour")

    # Test tour_valid
    if tour_valid(test_tour):
        print("TEST PASSED: tour_valid successfully identified valid tour")
    else:
        print(" ----- TEST FAILED: tour_valid failed to identify a valid tour")
    
    invalid_tour = np.asarray(
        [[0, 0, 0, 1, 0, 1, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 1],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0]])
    invalid_tour += invalid_tour.T

    if not tour_valid(invalid_tour):
        print("TEST PASSED: tour_valid successfully identified invalid tour")
    else:
        print(" ----- TEST FAILED: tour_valid")

    invalid_tour = np.asarray(
        [[1, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0]])
    invalid_tour += invalid_tour.T

    if not tour_valid(invalid_tour):
        print("TEST PASSED: tour_valid successfully identified invalid tour")
    else:
        print(" ----- TEST FAILED: tour_valid")

    
    

