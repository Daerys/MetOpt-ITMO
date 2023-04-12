import math
from random import sample
import numpy as np
import lab2.src.lab_2_utils as utils


# class field
class GradientDescent:
    def __init__(self, batch_size=None,
                 learning_rate=None,
                 max_epoch=100,
                 learning_rate_scheduling=None,
                 eps=1e-4):
        """
        :NOTE: We'll make a child classes which will
        override gradient function and learning_rate_scheduling will be passed to init
        """
        self.batch_size = batch_size
        self.lr = learning_rate if learning_rate else 1.0
        self.max_epoch = max_epoch
        self.learning_rate_scheduling = learning_rate_scheduling if learning_rate_scheduling else self.constant
        self.eps = eps

    def gradient(self, X, w, _, __, ___):
        """
        :NOTE: w(weights) = len(X[0]). We'll count X[i][-1] = 1 and "expected value" is X[i][-1]
        That's how w[-1] is variable without x_i
        """
        x, y = utils.split(X)
        gradient = utils.MSE_gradient(x, y) # w - self.lr * (w / self.lr - w_new /self.lr)
        return gradient(w)

    def constant(self, _):
        return self.lr

    def run(self, data, start_weights):
        data = np.asarray(data)
        start_weights = np.asarray(start_weights)
        w = start_weights.astype(np.float64).reshape(-1, 1)
        log = [start_weights]
        for epoch in range(self.max_epoch):
            indices = sample(range(len(data)), self.batch_size if self.batch_size else len(data))
            X = data[indices]

            self.lr = self.learning_rate_scheduling(epoch)
            gr = self.gradient(X, w, epoch, self.lr, log) # gr = (w / self.lr - wp * delta)
            w -= self.lr * gr

            log.append(w.copy().ravel())
            if np.linalg.norm(gr) < self.eps or np.linalg.norm(log[-1] - log[-2]) < self.eps:
                break

        return log, w


# class field end


"""



task3:
class BruhGradient(GradientDescent):
    def __init__(...) <- if needed

    def gradient(...) <- overriding old gradient
"""

class MomentumGradientDescent(GradientDescent):
    def __init__(self, batch_size=None, learning_rate=None, max_epoch=100, learning_rate_scheduling=None, eps=1e-4,
                 gamma=0.0001):
        super().__init__(batch_size, learning_rate, max_epoch, learning_rate_scheduling, eps)
        self.gamma = gamma

    def gradient(self, X, w, epoch, lr, log):
        x, y = utils.split(X)
        old_w = w.copy()
        gradient = utils.MSE_gradient(x, y)
        gradient = gradient(w)
        for j in range(len(w)):
            previous = log[-1][j] * self.gamma
            w[j] = w[j] - (previous + self.lr * gradient[j])
        return old_w/self.lr - w/self.lr

class NesterovGradientDescent(GradientDescent):
    def __init__(self, batch_size=None, learning_rate=None, max_epoch=100, learning_rate_scheduling=None, eps=1e-4,
                 gamma=0.0001):
        super().__init__(batch_size, learning_rate, max_epoch, learning_rate_scheduling, eps)
        self.gamma = gamma

    def gradient(self, X, w, epoch, lr, log):
        new_coefficients = w.copy()
        x, y = utils.split(X)
        for j in range(len(w)):
            new_coefficients[j] -= log[-1][j] * self.gamma
        new_gradient = utils.MSE_gradient(x, y)
        new_gradient = new_gradient(new_coefficients)
        for j in range(len(w)):
            previous = log[-1][j] * self.gamma
            w[j] = w[j] - (previous + self.lr * new_gradient[j])
        return new_gradient

class AdagradGradientDescent(GradientDescent):
    def __init__(self, batch_size=None, learning_rate=None, max_epoch=100, learning_rate_scheduling=None, eps=1e-4,):
        super().__init__(batch_size, learning_rate, max_epoch, learning_rate_scheduling, eps)
        self.G = None

    def gradient(self, X, w, epoch, lr, log):
        if len(log) == 1:
            return super().gradient(X, w, epoch, lr, log)
        if self.G is None:
            self.G = [0.0 for _ in range(len(log[0]))]
        for k in range(len(log[-1])):
            self.G[k] += log[-1][k] ** 2
        result = []
        for g in self.G:
            result.append(1e-1 / math.sqrt(g + 1e-8))
        self.lr = result
        x, y = utils.split(X)
        gradient = utils.MSE_gradient(x, y)
        gradient = gradient(w)

        return gradient

'''
class RMSPropGradientDescent(GradientDescent):
    def __init__(self, batch_size, learning_rate, max_epoch, learning_rate_scheduling, eps, beta):
        super().__init__(batch_size, learning_rate, max_epoch, learning_rate_scheduling, eps)
        self.beta = beta
        self.ema_grad = None

    def make_step(self, coefficients, step, gradients, current_objects):
        gradient = utils.MSE_gradient(coefficients, current_objects)

        if self.ema_grad is None:
            self.ema_grad = [0.0 for _ in range(len(coefficients))]

        for j in range(len(coefficients)):
            self.ema_grad[j] = exponential_moving_average(self.ema_grad[j], self.beta, gradient[j] ** 2)

        for j in range(len(coefficients)):
            coefficients[j] = coefficients[j] - (step[j] * gradient[j] / (math.sqrt(self.ema_grad[j]) + 1e-8))
        return gradient


class AdamGradientDescent(GradientDescent):
    def __init__(self, limit, eps, ema_coef, batch_size, beta1, beta2):
        super().__init__(limit, eps, ema_coef, batch_size)
        self.beta1 = beta1
        self.beta2 = beta2
        self.ema_grad = None
        self.ema_grad_sqr = None

    def make_step(self, coefficients, step, gradients, current_objects):
        gradient = utils.MSE_gradient(coefficients, current_objects)

        if self.ema_grad is None:
            self.ema_grad = [0.0 for _ in range(len(coefficients))]
        if self.ema_grad_sqr is None:
            self.ema_grad_sqr = [0.0 for _ in range(len(coefficients))]

        p = len(gradients)
        for j in range(len(coefficients)):
            self.ema_grad[j] = exponential_moving_average(self.ema_grad[j], self.beta1, gradient[j])
            self.ema_grad_sqr[j] = exponential_moving_average(self.ema_grad_sqr[j], self.beta2, gradient[j] ** 2)

        for j in range(len(coefficients)):
            m = self.ema_grad[j] / (1 - math.pow(self.beta1, p))
            v = self.ema_grad_sqr[j] / (1 - math.pow(self.beta2, p))
            coefficients[j] = coefficients[j] - (step[j] * m / (math.sqrt(v) + 1e-8))
        return gradient'''
