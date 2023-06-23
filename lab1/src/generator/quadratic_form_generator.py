import numpy as np


def generate_random_orthogonal_matrix(n: int):
    A = np.random.rand(n, n)
    Q, _ = np.linalg.qr(A)
    return Q


def generate_diagonal_matrix(k_min: float, k_max: float, n: int):
    eigenvalues = [k_min] + [np.random.uniform(k_min, k_max, n - 2)] + [k_max]
    return np.diag(eigenvalues)


def quadratic_function(A, x0):
    return lambda x: np.transpose(x - x0) @ A @ (x - x0)


def gradient(A, x0):
    def grad(x):
        x = np.array(x)
        return 2 * A @ (x - x0)

    return grad


def generate_quadratic(n: int, k: float):
    D = generate_diagonal_matrix(1, k, n)
    T = generate_random_orthogonal_matrix(n)
    A = np.matmul(np.matmul(T, D), np.transpose(T))

    # x0 = np.array(np.random.uniform(-5, 5, n))
    x0 = np.zeros(n)
    return quadratic_function(A, x0), gradient(A, x0)
