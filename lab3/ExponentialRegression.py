import numpy as np

class ExponentialRegression:
    def __init__(self, data_set=None):
        self.coefficients = np.array([1.0, 1.0])
        self.data_set = np.asarray(data_set)
        self.X, self.Y = self.data_set.T

    def set_coefficients(self, w):
        self.coefficients = w

    def get_coefficients(self):
        return self.coefficients

    def fun(self, x: float):
        a, b = self.coefficients
        return a * np.exp(b * x)

    def draw_fun(self, w):
        self.set_coefficients(w)
        return self.fun

    def MSE(self, w: np.array):
        a, b = w
        y_predict = a * np.exp(b * self.X)
        return np.sum((y_predict - self.Y) ** 2)

    def MSE_gradient(self, w: np.array):
        X, y = self.data_set.T
        a, b = w
        y_predict = a * np.exp(b * X)
        gradient_a = 2 * np.sum((y_predict - y) * np.exp(b * X))
        gradient_b = 2 * np.sum((y_predict - y) * a * X * np.exp(b * X))
        return np.array([gradient_a, gradient_b])

    def getR(self, w):
        a, b = w
        y_predict = a * np.exp(b * self.X)
        return self.Y - y_predict

    def Jacobi(self, w):
        a, b = w
        return np.asarray([[np.exp(b * x), a * x * np.exp(b * x)] for x in self.X])

    def Heussan(self, w):
        a, b = w
        return np.asarray([[0, a * x * x * np.exp(b * x)] for x in self.X])