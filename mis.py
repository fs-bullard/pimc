import numpy as np

pauli_matrices = {
    'x':np.matrix(([0, 1], [1, 0])),
    'y':np.matrix(([0, -1j], [1j, 0]), dtype=complex),
    'z':np.matrix(([1, 0], [0, -1]))
}

def generate_sigma_j(j: int, axis: str, n:int) -> np.matrix:
    """
    Inputs:
    j: the qubit being operated on
    axis: x, y or z 
    n: number of qubits in the system
    Returns:
    Matrix representation of the axis pauli operation
    on the jth qubit
    """
    # Set pauli matrix according to input axis
    sigma = pauli_matrices[axis]
    
    # Initialise sigma_j as 1
    sigma_j = 1

    # Iterate through to n
    for i in range(n):
        if i == j:
            sigma_j = np.kron(sigma_j, sigma)
        else:
            sigma_j = np.kron(sigma_j, np.identity(2))
    
    return sigma_j

def generate_strengths(M: np.matrix, kappa:float) -> tuple[np.matrix, np.ndarray]:
    """
    Inputs are M the adjacancy matrix and kappa the variable that rewards
    more independence
    Returns J and h, the coupling and field strengths respectively
    """
    # Set n to be the number of qubits in the system
    n = M.shape[0]

    # Initialise h as an array of kappas
    h = np.ones((n)) * kappa

    # Generate h - TODO method could probably be vectorised
    for k in range(n):
        for j in range(n):
            h[k] += -(M[k, j] + M[j, k])
    
    return M, h

def construct_H_ising(J: np.ndarray) -> np.matrix:
    """
    Inputs: J and h are the coupling and field strengths respectively
    Returns: The Ising Hamiltonian, a 2^n x 2^n matrix
    """
    J, h = 0, 0
    n = h.size
    H_ising = np.zeros((2**n, 2**n))

    # Add the first sum
    for k in range(n):
        for j in range(k + 1, n):
            sigma_j = generate_sigma_j(j, 'z', n)
            sigma_k = generate_sigma_j(k, 'z', n)
            H_ising += J[k, j] * sigma_k * sigma_j
    
    # Add the second sum
    for j in range(n):
        sigma_j = generate_sigma_j(j, 'z', n)
        H_ising += h[j] * sigma_j

    return H_ising