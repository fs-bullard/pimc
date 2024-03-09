import tsplib95
import numpy as np
import random
import heapq

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

def generate_M_neighbours(
        problem: tsplib95.models.StandardProblem, prev: int, next: int, M = 20) -> list:
    """Generates a list of the M nearest neighbours to city i

    Args:
        problem (tsplib95.models.StandardProblem)
        prev (int): city before the current city
        next (int): city we are looking to replace (after current city)
        M (int): number of neighbours to generate

    Returns:
        list: list of M nearest neighbours to city next (excluding prev)
    """
    N = problem.dimension
    neighbours = []

    for i in range(N):
        # Skip over prev
        if i != prev:
            neighbours.append((i, get_weight(problem, next, i)))

     # Use a heap to efficiently get the M smallest elements
    nearest_neighbours = heapq.nsmallest(M, neighbours, key=lambda x: x[1])

    # Extract the indices from the tuples
    return [i for i, _ in nearest_neighbours]

def two_opt_move(tour: np.ndarray, i: int, j: int, k:int, l: int) -> np.ndarray:
    """Conducts a two opt move swapping connections (i->k) and (j->l) to (i->l) and (j->k)

    Args:
        tour (np.ndarray)
        i (int)
        j (int)
        k (int)
        l (int)

    Returns:
        np.ndarray: tour with two-opt move applied
    """
    tour[i][k] = 0
    tour[k][i] = 0
    tour[i][l] = 1
    tour[l][i] = 1
    tour[j][l] = 0
    tour[l][j] = 0
    tour[j][k] = 1
    tour[k][j] = 1
    
    return tour

def classical_energy_tsp(problem: tsplib95.models.StandardProblem, tour: np.ndarray):
    """Returns the total weight of a given tour

    Args:
        problem (tsplib95.models.StandardProblem): TSP of interest
        tour (np.ndarray): tour represented by a matrix

    Returns:
        int: total weight of tour
    """
    N = len(tour)
    H = 0
    for i in range(N):
        for j in range(N):
            if tour[i][j]:
                H += get_weight(problem, i, j)

    return H // 2

def quantum_energy_tsp():
    return

if __name__ == '__main__':
    # Load problem and optimal tour
    problem_filepath = 'tsplib/pr1002.tsp'
    problem = load_tsp_instance(problem_filepath)
    opt_filepath = 'tsplib/pr1002.opt.tour'
    opt = load_tsp_instance(opt_filepath)
    N = problem.dimension

    # Check classical energy gives correct weight
    opt_weight = get_tour_dist(problem, opt.tours[0])
    opt_tour = tour_to_matrix(opt.tours[0])

    print(f"Problem {problem.name} loaded with {N} cities and optimal weight {opt_weight}")

    if opt_weight == classical_energy_tsp(problem, opt_tour):
        print("TEST PASSED: Classical energy function returns correct optimal weight")
    else:
        print("TEST FAILED: Classical energy function returns incorrect optimal weight")

    rand_tour = generate_random_tour(N)
    rand_weight = classical_energy_tsp(problem, rand_tour)

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

    if np.array_equal(two_opt_move(test_tour, 1, 3, 4, 2), two_opt_tour):
        print("TEST PASSED: two_opt_move was successful")
    else:
        print("TEST FAILED: two_opt_move was unsuccessful")

    # Ideally I would like to test the nearest neighbours code but it will take a while - reasonably
    # sure it is ok
    