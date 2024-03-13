import matplotlib.pyplot as plt
import numpy as np
import tsplib95

from tsp import (
    load_tsp_instance, 
    tour_to_matrix, 
    build_weight_dict, 
    classical_energy_tsp
    )
from sa import simulated_annealing
from qa import quantum_annealing

def compare_annealing_steps(problem: tsplib95.models.StandardProblem, opt_energy: int, r: int = 10):
    """Generates plot comparing the accuracy of optimisation methods for 
    different numbers of annealing steps

    Args:
        problem (tsplib95.models.StandardProblem): TSP of interest
        opt_energy (int): length of optimal tour
        r (int): number of times to repeat each calculation
    """
    # Set width of image (inches)
    W = 3.2
    plt.rcParams.update({
        'figure.figsize': (W, 3*W/4),
        'font.size': 8,
        'axes.labelsize': 10,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'text.usetex': True
    })

    # annealing_steps = [50, 75, 100, 150, 200, 500, 1000]
    annealing_steps = [50, 200, 1000]
    T_0, T_f = 100, 0.001
    G_0, G_f = 300, 0.001
    T = 10/3
    P = 2

    means_sa = []
    means_qa = []

    stds_sa = []
    stds_qa = []

    for num in annealing_steps:
        T_schedule = np.linspace(T_0, T_f, num)
        G_schedule = np.linspace(G_0, G_f, num)

        lengths_sa = []
        lengths_qa = []        

        for _ in range(r):
            tour, E = simulated_annealing(problem, T_schedule)
            lengths_sa.append(E)

            tour, E = quantum_annealing(problem, G_schedule, T, P)
            lengths_qa.append(E)

        means_sa.append(np.mean(lengths_sa))
        stds_sa.append(np.std(lengths_sa))

        means_qa.append(np.mean(lengths_qa))
        stds_qa.append(np.std(lengths_qa))
    
    plt.errorbar(
        annealing_steps, 
        100*(np.array(means_sa) / opt_energy - 1),
        yerr=100*(np.array(stds_sa) / (opt_energy*np.sqrt(r))),
        fmt='o-', 
        markersize=4,
        capsize=2, 
        color='black',
        label='SA',
        linewidth=1

    )

    plt.errorbar(
        annealing_steps, 
        100*(np.array(means_qa) / opt_energy - 1),
        yerr=100*(np.array(stds_qa) / (opt_energy*np.sqrt(r))),
        fmt='o-', 
        markersize=4,
        capsize=2, 
        color='red',
        label='QA',
        linewidth=1
    )

    plt.xscale('log')
    plt.xlabel(r'Number of Monte Carlo Steps')
    plt.ylabel(r'Excess Length ($\%$)')
    plt.legend()

    # Add minor ticks on the x-axis
    plt.minorticks_on()

    # Customize the appearance of the minor ticks
    plt.tick_params(which='minor', size=1.5, width=0.7, direction='in', )
    plt.tick_params(which='both', direction='in', top=True, right=True)


    plt.savefig(
        f'figures/{problem.name}/annealing_steps.jpg', 
        dpi=1000,
        bbox_inches='tight'
    )
    # plt.show()

def compare_starting_temperature(problem: tsplib95.models.StandardProblem, opt_energy: int, r: int):
    """Generates plot comparing the accuracy of optimisation methods for 
    different starting temperature

    Args:
        problem (tsplib95.models.StandardProblem): TSP of interest
        opt_energy (int): length of optimal tour
        r (int): number of times to repeat each calculation
    """
    # Set width of image (inches)
    W = 3.2
    plt.rcParams.update({
        'figure.figsize': (W, 3*W/4),
        'font.size': 8,
        'axes.labelsize': 10,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'text.usetex': True
    })

    # temperatures = [50, 75, 100, 150, 200, 500, 1000]
    temperatures = [1, 5, 10, 20, 30, 40, 50, 75, 100, 200]
    annealing_steps = 100

    T_f = 0.001

    means = []

    for T_0 in temperatures:
        T_schedule = np.linspace(T_0, T_f, annealing_steps)

        lengths = []
        stds = []

        for _ in range(r):
            tour, E = simulated_annealing(problem, T_schedule)
            lengths.append(E)
        
        mean = np.mean(lengths)
        std = np.std(lengths)

        means.append(mean)
        stds.append(std)
    
    plt.errorbar(temperatures, 100*(np.array(means) / opt_energy - 1),
                yerr=100*(np.array(stds) / (opt_energy*np.sqrt(r))), marker='o', fmt='o-', 
                capsize=3, color='black')

    plt.xlabel('Initial Annealing Temperature (T_0)')
    plt.ylabel('Excess Length After Annealing (%)')

    # Add minor ticks on the x-axis
    plt.minorticks_on()

    # Customize the appearance of the minor ticks
    plt.tick_params(which='minor', size=3, width=1, direction='in')

    plt.savefig(
        f'figures/{problem.name}/init_temp.jpg', 
        dpi=1000,
        bbox_inches='tight'
    )
    plt.show()

if __name__ == "__main__":
    print("Generating Plots")

    # Load problem and optimal tour
    problem_filepath = 'tsplib/ulysses16.tsp' 
    problem = load_tsp_instance(problem_filepath)
    opt_filepath = 'tsplib/ulysses16.opt.tour'
    opt = load_tsp_instance(opt_filepath)
    opt_tour = tour_to_matrix(opt.tours[0])
    
    weights = build_weight_dict(problem)

    # Set optimal solution
    opt_energy = classical_energy_tsp(weights, opt_tour)
    print("Optimal energy:", opt_energy)

    # Generate plot comparing performance for different number of annealing steps
    compare_annealing_steps(problem, opt_energy, 3)

    # Generate plot comparing performance for different initial temps
    # compare_starting_temperature(problem, opt_energy, 20)
