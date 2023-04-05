import numpy as np


def MSE(X, y):
    def foo(weights):
        weights = np.asarray(weights)
        y_pred = X @ weights.reshape(-1, 1)
        return np.sum((y_pred - y) ** 2)

    return foo


def MSE_gradient(X: np.array, y: np.array):
    def foo(weights):
        weights = np.asarray(weights)
        y_pred = X @ weights.reshape(-1, 1)
        return 2 * X.T @ (y_pred - y)

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


# X = np.array([[1, 2], [3, 4], [5, 6]])
# y = np.array([1, 2, 3])
#
# print(merge(X, y))
#
# weights = np.array([1, 1])
#
# grad_func = MSE_gradient(X, y)
# grad = grad_func(weights)
#
# print(grad) # [14. 20.]
