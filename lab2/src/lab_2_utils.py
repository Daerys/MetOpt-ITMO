from cmath import exp

import numpy as np

K = 16

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
        k = 0.003
        new_lr = init_lr * np.exp(-k * t)
        return new_lr

    return foo


def MSE_poly(X: np.array, y: np.array, k):
    A = np.array([polynomial(x, k) for x in X])

    def foo(weights):
        weights = np.asarray(weights)
        y_pred = A @ weights.reshape(-1, 1)
        return np.sum((y_pred - y) ** 2)

    return foo


def MSE_poly_gradient(X: np.array, y: np.array, k):
    A = np.array([polynomial(x, k) for x in X])

    def foo(weights):
        weights = np.asarray(weights)
        y_pred = A @ weights.reshape(-1, 1)
        return 2 * A.T @ (y_pred - y)

    return foo


def polynomial(x, k):
    X = []
    for i in range(k):
        X.append(x ** (i + 1))
    X.append(1)
    return np.array(X)


def poly_fun(w):
    w = np.squeeze(w)

    def foo(x):
        ans = w[-1]
        for i in range(len(w) - 1):
            ans += w[i] * (x ** (i + 1))
        return ans

    return foo
