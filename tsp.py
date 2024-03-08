import tsplib95

def load_tsp_instance(filepath: str):
    return tsplib95.load(filepath)

def classical_energy_tsp():
    return

def quantum_energy_tsp():
    return

if __name__ == '__main__':
    filepath = 'tsplib/pr1002.tsp'
    problem = load_tsp_instance(filepath)

    print(problem.get_weight(1, 2))