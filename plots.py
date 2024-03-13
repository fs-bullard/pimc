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

    # Simulate for SA
    # annealing_steps_sa = [50, 75, 100, 150, 200, 350, 500, 1000, 2000, 5000, 10000]
    # annealing_steps_sa = [35] # For testing
    # T_0, T_f = 100, 0.001

    # means_sa = []
    # stds_sa = []

    # for num in annealing_steps_sa:
    #     print(f'annealing steps: {num}')
    #     T_schedule = np.linspace(T_0, T_f, num)

    #     lengths_sa = []

    #     for _ in range(r):
    #         print('repeat:', str(_))       
    #         tour, E = simulated_annealing(problem, T_schedule)
    #         lengths_sa.append(E)

    #     means_sa.append(np.mean(lengths_sa))
    #     stds_sa.append(np.std(lengths_sa))

    # print(means_sa)
    # print(stds_sa)
    # return

    # # Repeat for QA, excluding the really high numbers
    annealing_steps = [15, 25, 50, 75, 100, 150, 200, 350, 500, 1000]
    # annealing_steps = [15] # For testing

    # G_0, G_f = 300, 0.001
    # PT = 50
    # P = 10

    # means_qa = []
    # stds_qa = []

    # for num in annealing_steps:
    #     print(f'annealing steps: {num}')

    #     G_schedule = np.linspace(G_0, G_f, num)

    #     lengths_qa = []        

    #     for _ in range(r):     
    #         print('repeat:', str(_))       
    #         tour, E = quantum_annealing(problem, G_schedule, PT/P, P)
    #         lengths_qa.append(E)

    #     means_qa.append(np.mean(lengths_qa))
    #     stds_qa.append(np.std(lengths_qa))
    
    # print(means_qa)
    # print(stds_qa)

    # return

    # # Save the data
    # data = {
    #     "annealing_steps": annealing_steps_sa,
    #     "means_sa": means_sa,
    #     "stds_sa": stds_sa,
    #     "means_qa": means_qa,
    #     "stds_qa": stds_qa,
    # }
    # np.savez(f'data/{problem.name}/annealing_steps_data.npz', **data)

    # Or load data from file 
    data = np.load('data/ulysses16/annealing_steps_data_good.npz')

    annealing_steps_sa = data['annealing_steps']
    means_sa = data['means_sa']
    stds_sa = data['stds_sa']
    means_qa = data['means_qa']
    stds_qa = data['stds_qa']

    means_sa = [6951.3] + list(means_sa)
    stds_sa = [81.27859496817105] + list(stds_sa)
    annealing_steps_sa = [35] + list(annealing_steps_sa)

    means_qa = [6994.1, 6909.4] + list(means_qa)
    stds_qa = [78.91698169595692, 30.65517900779573] + list(stds_qa)
    
    plt.errorbar(
        annealing_steps_sa, 
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
    plt.xlabel(r'Number of Monte Carlo Steps $\tau$')
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

def compare_starting_temperature(problem: tsplib95.models.StandardProblem, opt_energy: int, r: int):
    """Generates plot comparing the accuracy of optimisation methods for 
    different starting temperature

    for qa compare PT - can use the same axis

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

    # temperatures = [50, 1000]
    temperatures = [1, 5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 
                    175, 200, 300, 400]
    annealing_steps = 100
    P = 10

    T_f = 0.001
    G_0, G_f = 300, 0.001

    means_sa = []
    means_qa = []

    stds_sa = []
    stds_qa = []

    # for T_0 in temperatures:
    #     print(f'T_0: {T_0}')
    #     T_schedule = np.linspace(T_0, T_f, annealing_steps)
    #     G_schedule = np.linspace(G_0, G_f, annealing_steps)

    #     lengths_sa = []
    #     lengths_qa = []  

    #     for _ in range(r):
    #         print('repeat:', str(_))       
    #         tour, E = simulated_annealing(problem, T_schedule)
    #         lengths_sa.append(E)

    #         T = T_0 / P
    #         tour, E = quantum_annealing(problem, G_schedule, T, P)
    #         lengths_qa.append(E)

        
    #     means_sa.append(np.mean(lengths_sa))
    #     stds_sa.append(np.std(lengths_sa))

    #     means_qa.append(np.mean(lengths_qa))
    #     stds_qa.append(np.std(lengths_qa))
    
    # # Save the data
    # data = {
    #     "temperatures": temperatures,
    #     "means_sa": means_sa,
    #     "stds_sa": stds_sa,
    #     "means_qa": means_qa,
    #     "stds_qa": stds_qa,
    # }
    # np.savez(f'data/{problem.name}/init_temp_data.npz', **data)

    # Or load from file
    data = np.load('data/ulysses16/init_temp_data_good.npz')

    temperatures = data['temperatures']
    means_sa = data['means_sa']
    stds_sa = data['stds_sa']
    means_qa = data['means_qa']
    stds_qa = data['stds_qa']
    
    plt.errorbar(
        temperatures, 
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
        temperatures[:-3], 
        100*(np.array(means_qa[:-3]) / opt_energy - 1),
        yerr=100*(np.array(stds_qa[:-3]) / (opt_energy*np.sqrt(r))),
        fmt='o-', 
        markersize=4,
        capsize=2, 
        color='red',
        label='QA',
        linewidth=1
    )

    plt.xlabel(r'$T_0$ and $PT$')
    plt.ylabel(r'Excess Length ($\%$)')
    plt.legend()

    # Add minor ticks on the x-axis
    plt.minorticks_on()

    # Customize the appearance of the minor ticks
    plt.tick_params(which='minor', size=1.5, width=0.7, direction='in', )
    plt.tick_params(which='both', direction='in', top=True, right=True)

    plt.savefig(
        f'figures/{problem.name}/init_temp.jpg', 
        dpi=1000,
        bbox_inches='tight'
    )

def compare_trotter_number(problem: tsplib95.models.StandardProblem, opt_energy: int, r: int = 10):
    """Generates plot comparing the accuracy of optimisation methods for 
    different Trotter numbers

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

    trotter_numbers = [2, 5, 10, 20, 30]
    # trotter_numbers = [1, 2]

    G_0, G_f = 300, 0.001
    PT = 50
    annealing_steps = 100

    means_qa = []
    stds_qa = []

    for P in trotter_numbers:
        print(f'P: {P}')
        G_schedule = np.linspace(G_0, G_f, annealing_steps)

        lengths_qa = []

        for _ in range(r):
            print('repeat:', str(_))       
            tour, E = quantum_annealing(problem, G_schedule, PT/P, P)
            lengths_qa.append(E)

        means_qa.append(np.mean(lengths_qa))
        stds_qa.append(np.std(lengths_qa))

    # Save the data
    data = {
        "trotter_numbers": trotter_numbers,
        "means_qa": means_qa,
        "stds_qa": stds_qa,
    }
    np.savez(f'data/{problem.name}/trotter_number_data.npz', **data)

    plt.errorbar(
        trotter_numbers, 
        100*(np.array(means_qa) / opt_energy - 1),
        yerr=100*(np.array(stds_qa) / (opt_energy*np.sqrt(r))),
        fmt='o-', 
        markersize=4,
        capsize=2, 
        color='red',
        label='QA',
        linewidth=1
    )

    plt.xlabel(r'Trotter Number $P$')
    plt.ylabel(r'Excess Length ($\%$)')

    # Add minor ticks on the x-axis
    plt.minorticks_on()

    # Customize the appearance of the minor ticks
    plt.tick_params(which='minor', size=1.5, width=0.7, direction='in')
    plt.tick_params(which='both', direction='in', top=True, right=True)

    plt.savefig(
        f'figures/{problem.name}/trotter_number.jpg', 
        dpi=1000,
        bbox_inches='tight'
    )

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
    compare_annealing_steps(problem, opt_energy, 20)

    # Generate plot comparing performance for different initial temps / PT
    # compare_starting_temperature(problem, opt_energy, 20)

    # Generate plot comparing performance for different Trotter numbers
    # compare_trotter_number(problem, opt_energy, 20)
