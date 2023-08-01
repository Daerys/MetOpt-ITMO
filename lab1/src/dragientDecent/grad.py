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
        elif stop_criteria == 'mixed':
            self.max_iter = 1e1
            self.stop = self.stop_mixed
        else:
            raise ValueError("Invalid stop criteria algorithm")

        if learning_rate == 'constant':
            self.dlr = self.const_step
        elif learning_rate == 'dichotomy':
            self.search = self.dichotomy_search
            self.dlr = self.one_dim
        elif learning_rate == 'wolfe':
            self.search = self.wolfe_search
            self.dlr = self.one_dim
        else:
            raise ValueError("Invalid learning rate algorithm")

    def set_start_point(self, start_point: np.array):
        self.x = np.array(start_point, dtype=np.float64)

    def set_epsilon(self, epsilon: float):
        self.eps = epsilon

    def change_max_iter(self, max_iter: int):
        self.max_iter = max_iter

    def change_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate

    def max_iter_function(self):
        return self.max_iter <= self.grad_iter

    def const_step(self):
        return

    def stop_mixed(self):
        return self.max_iter_function() or self.stop_by_gradient()

    def stop_by_gradient(self):
        return np.linalg.norm(self.grad_fun(self.x)) < self.eps

    def armijo_condition(self, lr, c1=1e-4):
        return self.fun(self.x - lr * self.grad_fun(self.x)) <= self.fun(self.x) - c1 * lr * np.linalg.norm(
            self.grad_fun(self.x)) ** 2

    def curvature_condition(self, lr, c2=0.9):
        return np.dot(np.array(self.grad_fun(self.x - lr * self.grad_fun(self.x))).flatten(),
                      np.array(self.grad_fun(self.x)).flatten()) >= c2 * \
            np.linalg.norm(np.array(self.grad_fun(self.x)).flatten()) ** 2

    def wolfe_search(self, left, right):
        lr = (left + right) / 2

        if not self.armijo_condition(lr):
            return left, lr, False, True
        elif not self.curvature_condition(lr):
            return lr, right, False, True
        else:
            return lr, right, True, False

    def dichotomy_search(self, left, right):
        lr1 = (left + right) / 2 - self.eps
        lr2 = (left + right) / 2 + self.eps

        c1 = self.x - lr1 * np.array(self.grad_fun(self.x))
        c2 = self.x - lr2 * np.array(self.grad_fun(self.x))

        f_c1 = self.fun(c1)
        f_c2 = self.fun(c2)

        if f_c1 <= f_c2:
            return left, lr2, False, True
        return lr1, right, False, True

    def one_dim(self):

        approximation_resolution = 5e-2
        left = 1e-10  # min learning rate
        right = self.learning_rate  # max learning rate
        stop = False
        can_make_a_step = False

        while right - left > approximation_resolution and not stop:
            left, right, stop, can_make_a_step = self.search(left, right)

        if not can_make_a_step:
            return True
        self.learning_rate = left
        return False

    def gradient(self):
        points = []
        while not self.stop():
            points.append(np.array(self.x))
            self.x -= self.learning_rate * np.array(self.grad_fun(self.x), dtype=np.float64)

            if self.dlr():
                break
            self.grad_iter = self.grad_iter + 1
        points = np.array(points)
        return points
