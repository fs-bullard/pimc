import tsplib95
from tsp import (
    load_tsp_instance, 
    generate_random_tour, 
    tour_to_list, 
    classical_energy_tsp, 
    tour_to_matrix, 
    build_weight_dict
)
import matplotlib.pyplot as plt


def plot_tour(problem: tsplib95.models.StandardProblem, tour: list[int], tour_length: str, subplot_position: int):
    """Plots a TSP with given tour connections

    Args:
        problem (tsplib95.models.StandardProblem): _description_
        tour (list[int]): _description_
        tour_length (str): Total tour length
        subplot_position (int): Position of the subplot
    """
    x = [problem.node_coords[city][0] for city in tour]
    y = [problem.node_coords[city][1] for city in tour]
    
    plt.subplot(1, 2, subplot_position)
    plt.plot(x, y, 'o-', markersize=8, color='black')  # Increased markersize and changed color to black
    plt.xticks([])
    plt.yticks([])
    for i, city in enumerate(tour):
        plt.text(x[i], y[i], str(city), ha='center', va='center', color='white')  # Changed color to white
    
    # Add tour length text at bottom right
    plt.text(max(x), min(y), r'$\sum_{i,j} d_{ij} = $ ' + str(tour_length), fontsize=8, ha='right', va='bottom')


if __name__ == '__main__':
    # Set matplotlib settings
    W = 3.2
    plt.rcParams.update({
        'figure.figsize': (W, 3*W/4),
        'font.size': 6,
        'axes.labelsize': 10,
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'text.usetex': True
    })

    # Load problem and optimal tour
    problem_filepath = 'tsplib/ulysses16.tsp'
    problem = load_tsp_instance(problem_filepath)
    opt_filepath = 'tsplib/ulysses16.opt.tour'
    opt = load_tsp_instance(opt_filepath)

    weights = build_weight_dict(problem)

    opt_tour = opt.tours[0]
    random_tour = [i + 1 for i in tour_to_list(generate_random_tour(problem.dimension))]
    
    # Example tour lengths
    opt_tour_length = classical_energy_tsp(weights, tour_to_matrix(opt_tour))
    random_tour_length = classical_energy_tsp(weights, tour_to_matrix(random_tour))

    # Load modified tour for nicer plotting
    problem_filepath = 'tsplib/ulysses16_mod.tsp'
    problem = load_tsp_instance(problem_filepath)

    plot_tour(problem, random_tour + [1], random_tour_length, 1)
    plot_tour(problem, opt_tour + [1], opt_tour_length, 2)

    plt.tight_layout()
    plt.savefig(
        f'figures/diagram/{problem.name}_visualised.jpg', 
        dpi=1000,
        bbox_inches='tight'
    )
    # plt.show()
