from cmath import exp

import numpy as np


def MSE(X, y):
    def foo(weights):
        weights = np.asarray(weights)
        y_pred = X @ weights.reshape(-1, 1)
        return np.sum((y_pred - y) ** 2)

    return foo


def MSE_gradient(X: np.array, y: np.array, noise_level=0.1):
    def foo(weights):
        weights = np.asarray(weights)
        y_pred = X @ weights.reshape(-1, 1)
        a = 2 * X.T @ (y_pred - y)
        noise = np.random.normal(scale=noise_level, size=a.shape)
        return a + noise

    return foo


def merge(X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    data = np.concatenate((X, Y), axis=1)
    return data


def extend(X):
    return merge(X, [1 for _ in range(len(X))])


def split(X):
    X = np.asarray(X)
    return extend(X[:, :-1]), X[:, -1:]


def exponential_decay(init_lr):
    def foo(t):
        k = 0.5
        new_lr = init_lr * np.exp(-k * t)
        return new_lr

    return foo


# additional task
def MSE_poly(X: np.array, y: np.array):
    X = np.vander(X, len(X[0]))

    def foo(weights):
        weights = np.asarray(weights)
        y_pred = X @ weights.reshape(-1, 1)
        return np.sum((y_pred - y) ** 2)

    return foo


def MSE_poly_gradient(X: np.array, y: np.array):
    X = np.vander(X, len(X[0]))

    def foo(weights):
        weights = np.asarray(weights)
        y_pred = X @ weights.reshape(-1, 1)
        return 2 * X.T @ (y_pred - y)

    return foo
