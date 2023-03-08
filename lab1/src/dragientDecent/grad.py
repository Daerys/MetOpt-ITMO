import numpy as np
import src.dichotomy as dic
import src.generator.quadratic_form_generator as g

class GradientDescent:
    def __init__(self, learning_rate='constant',
                 stop_criteria='max_iter',
                 function=None,
                 grad_function=None,
                 eps=1e-4,
                 start_point=None):
        self.fun = function
        self.grad_fun = grad_function
        self.eps = eps
        self.x = start_point
        self.grad_iter = 0
        if stop_criteria == 'max_iter':
            self.max_iter = 1e1
            self.stop = self.max_iter_function
        else:
            self.stop = self.norm
        if learning_rate == 'constant':
            self.learning_rate = 10
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
        self.learning_rate = self.learning_rate / 2

    def dich(self):
        return None

    def gradient(self):
        points = [np.array(self.x)]
        while self.stop():

            self.x -= self.learning_rate * np.array(self.grad_fun(self.x))
            points.append(np.array(self.x))

            self.dlr()
            self.grad_iter = self.grad_iter + 1
        points = np.array(points)
        print(points)


if __name__ == '__main__':
    f, gr = g.generate_quadratic(3, 1)
    a = GradientDescent('constant',
                 'max_iter',
                 f,
                 gr,
                 1e-4,
                 [0,0,0])
    a.change_learning_rate(50)
    a.change_learning_rate(4)
    a.gradient()

