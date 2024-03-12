import matplotlib.pyplot as plt
import numpy as np
import tsplib95

from tsp import load_tsp_instance, tour_to_matrix, build_weight_dict, classical_energy_tsp
from sa import simulated_annealing

def compare_annealing_steps(problem: tsplib95.models.StandardProblem, opt_energy: int, r: int = 10):
    """Generates plot comparing the accuracy of optimisation methods for 
    different numbers of annealing steps

    Args:
        problem (tsplib95.models.StandardProblem): TSP of interest
        opt_energy (int): length of optimal tour
        r (int): number of times to repeat each calculation
    """
    annealing_steps = [10, 50, 75, 100, 150, 200, 500, 1000]
    T_0, T_f = 100, 0.001

    means = []

    for num in annealing_steps:
        T_schedule = np.linspace(T_0, T_f, num)

        lengths = []
        stds = []

        for _ in range(r):
            tour, E = simulated_annealing(problem, T_schedule)
            lengths.append(E)
        
        mean = np.mean(lengths)
        std = np.std(lengths)

        means.append(mean)
        stds.append(std)
    
    plt.errorbar(annealing_steps, 100*(np.array(means) / opt_energy - 1),
                yerr=100*(np.array(stds) / opt_energy), marker='o', color='black')

    plt.xscale('log')
    plt.xlabel('Number of Monte Carlo Steps')
    plt.ylabel('Excess Length After Annealing (%)')

    plt.savefig(f'figures/{problem.name}/annealing_steps.jpg', dpi=600)
    plt.show()

if __name__ == "__main__":
    print("Generating Plots")

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

    # Generate plot comparing performance for different number of annealing steps
    compare_annealing_steps(problem, opt_energy, 3)
