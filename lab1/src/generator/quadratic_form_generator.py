import numpy as np


def generate_random_orthogonal_matrix(n: int):
    A = np.random.rand(n, n)
    Q, _ = np.linalg.qr(A)
    return Q


def generate_diagonal_matrix(k_min: float, k_max: float, n: int):
    eigenvalues = [k_min] + [np.random.uniform(k_min, k_max)] * (n - 2) + [k_max]
    return np.diag(eigenvalues)


def quadratic_function(A):
    return lambda x: x.T @ A @ x


def gradient(A):
    return lambda x: 2 * A @ x


def generate_quadratic(n: int, k: int):
    k_min = np.random.uniform(1, np.sqrt(k))
    k_max = k * k_min
    D = generate_diagonal_matrix(k_min, k_max, n)
    T = generate_random_orthogonal_matrix(n)
    A = np.matmul(np.matmul(T, D), np.transpose(T))
    return quadratic_function(A), gradient(A)
