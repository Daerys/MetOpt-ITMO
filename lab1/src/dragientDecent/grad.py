import numpy as np
import src.dichotomy as dic


class GradientDescent:
    def __init__(self, learning_rate='constant',
                 stop_criteria='max_iter',
                 function=None,
                 grad_function=None,
                 eps=1e-4,
                 start_point=None):
        self.learning_rate = learning_rate
        self.stop_criteria = stop_criteria
        self.fun = function
        self.grad_fun = grad_function
        self.eps = eps
        self.x = start_point
        self.grad_iter = 0
        if stop_criteria == 'max_iter':
            self.max_iter = 1e4
            self.stop = self.max_iter_function
        else:
            self.stop = self.norm
        if learning_rate == 'constant':
            self.learning_rate = 1e-2
            self.dlr = self.const_step
        else:
            self.learning_rate_iter = 0
            self.dlr = None

    def change_max_iter(self, max_iter):
        self.max_iter = max_iter

    def change_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def max_iter_function(self):
        return self.grad_iter < self.max_iter

    def norm(self):
        return np.linalg.norm(self.x) < self.eps

    def const_step(self):
        self.learning_rate /= 2

    def dich(self):
        return None

    @staticmethod
    def gradient(x):
        return None
