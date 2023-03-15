import numpy as np


class GradientDescent:
    def __init__(self, learning_rate='constant',
                 stop_criteria='max_iter',
                 function=None,
                 grad_function=None):

        self.fun = function
        self.grad_fun = grad_function

        self.eps = 1e-7
        self.x = None
        self.learning_rate = 1
        self.grad_iter = 0

        if stop_criteria == 'max_iter':
            self.max_iter = 1e1
            self.stop = self.max_iter_function
        elif stop_criteria == 'gradient':
            self.stop = self.stop_by_gradient
        else:
            raise ValueError("Invalid stop criteria algorithm")

        if learning_rate == 'constant':
            self.dlr = self.const_step
        elif learning_rate == 'dichotomy':
            self.dlr = self.dichotomy
            self.learning_rate_iter = 0
        else:
            raise ValueError("Invalid learning rate algorithm")

    def set_start_point(self, start_point: np.array):
        self.x = start_point

    def set_epsilon(self, epsilon: float):
        self.eps = epsilon

    def change_max_iter(self, max_iter: int):
        self.max_iter = max_iter

    def change_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate

    def max_iter_function(self):
        return self.max_iter <= self.grad_iter

    def const_step(self):
        # self.learning_rate = self.learning_rate / 2
        return

    def stop_by_gradient(self):
        return np.linalg.norm(self.grad_fun(self.x)) < self.eps

    def dichotomy(self):

        approximation_resolution = 5e-2
        left = 1e-10  # min learning rate
        right = 2  # max learning rate

        while right - left > approximation_resolution:
            lr1 = (left + right) / 2 - self.eps
            lr2 = (left + right) / 2 + self.eps

            c1 = self.x - lr1 * np.array(self.grad_fun(self.x))
            c2 = self.x - lr2 * np.array(self.grad_fun(self.x))

            f_c1 = self.fun(c1)
            f_c2 = self.fun(c2)

            if f_c1 <= f_c2:
                right = lr2
            else:
                left = lr1

        self.learning_rate = left

    def gradient(self):
        points = []
        while not self.stop():
            points.append(np.array(self.x))
            self.x -= self.learning_rate * np.array(self.grad_fun(self.x)).astype(np.float64)

            self.dlr()
            self.grad_iter = self.grad_iter + 1
        points = np.array(points)
        return points
