import numpy as np


def polynomial(x, k):
    X = []
    for i in range(k):
        X.append(x ** (i + 1))
    X.append(1)
    return np.array(X)


class PolynomialRegression:
    def __init__(self, data_set=None, degree=1):
        self.degree = degree
        self.coefficients = np.zeros(degree + 1)
        self.data_set = np.asarray(data_set)
        self.X, self.Y = self.data_set.T
        self.MSE_X = np.array([polynomial(x, self.degree) for x in self.X])

    def set_coefficients(self, w):
        self.coefficients = w

    def get_coefficients(self):
        return self.coefficients

    def fun(self, x: float):
        return np.sum([self.coefficients[i] * x ** (i + 1) for i in range(self.degree)] + [self.coefficients[-1]])

    def MSE(self, w: np.array):
        y_predict = self.MSE_X @ w
        return np.sum((y_predict - self.Y) ** 2)

    def MSE_gradient(self, w: np.array, indexes):
        X, y = self.data_set[indexes].T
        X = np.array([polynomial(x, self.degree) for x in X])
        y_predict = X @ w
        return 2 * X.T @ (y_predict - y)

    def getR(self, w):
        y_predict = self.MSE_X @ w
        return self.Y - y_predict

    def Jacobi(self, _):
        return np.asarray([np.asarray([x ** (i + 1) for i in range(self.degree)] + [1]) for x in self.X])