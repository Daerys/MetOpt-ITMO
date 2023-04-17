import math
from random import sample

import numpy as np


class GradientDescent:
    def __init__(self, Regression, batch_size=None,
                 learning_rate=None,
                 max_epoch=100,
                 learning_rate_scheduling=None,
                 eps=1e-4):
        """
        :NOTE: We'll make a child classes which will
        override gradient function and learning_rate_scheduling will be passed to init
        """
        self.regression = Regression
        self.batch_size = batch_size
        self.lr = learning_rate if learning_rate else 1.0
        self.max_epoch = max_epoch
        self.learning_rate_scheduling = learning_rate_scheduling if learning_rate_scheduling else self.constant
        self.eps = eps

    def gradient(self, indices, w, _, __, ___):
        """
        :NOTE: w(weights) = len(X[0]). We'll count X[i][-1] = 1 and "expected value" is X[i][-1]
        That's how w[-1] is variable without x_i
        """
        return self.regression.MSE_gradient(w, indices)
        # x, y = utils.split(X)
        # x = np.squeeze(x[:, :-1])
        # gradient = utils.MSE_poly_gradient(x, y, utils.K)
        # gradient = utils.MSE_gradient(x, y)  # w - self.lr * (w / self.lr - w_new /self.lr)
        # return np.array(gradient(w))
        # return np.array(gradient(w) + L2 * 2 * w)
        # return np.array(gradient(w) + L1 * np.sign(w))

    #    return  np.array(gradient(w) + L1 * np.sign(w) + L2 * 2 * w )

    def constant(self, _):
        return self.lr

    def run(self):
        data_size = len(self.regression.data_set)
        w = self.regression.coefficients
        log = [w.copy]
        for epoch in range(self.max_epoch):
            indices = sample(range(data_size), self.batch_size if self.batch_size else data_size)

            self.lr = self.learning_rate_scheduling(epoch)
            gr = self.gradient(indices, w, epoch, self.lr, log)  # gr = (w / self.lr - wp * delta)
            w -= self.lr * gr

            log.append(w.copy().ravel())
            # if np.linalg.norm(gr) < self.eps or np.linalg.norm(log[-1] - log[-2]) < self.eps:
            #     break

        return log, w


class MomentumGradientDescent(GradientDescent):
    def __init__(self, Regression, batch_size=None, learning_rate=None, max_epoch=100, learning_rate_scheduling=None, eps=1e-4,
                 gamma=0.0001):
        super().__init__(Regression, batch_size, learning_rate, max_epoch, learning_rate_scheduling, eps)
        self.gamma = gamma

    def gradient(self, indexes, w, epoch, lr, log):
        old_w = w.copy()
        gradient = self.regression.MSE_gradient(w, indexes)
        for j in range(len(w)):
            previous = log[-1][j] * self.gamma
            w[j] = w[j] - (previous + self.lr * gradient[j])
        return old_w / self.lr - w / self.lr


class NesterovGradientDescent(GradientDescent):
    def __init__(self, Regression, batch_size=None, learning_rate=None, max_epoch=100, learning_rate_scheduling=None, eps=1e-4,
                 gamma=0.0001):
        super().__init__(Regression, batch_size, learning_rate, max_epoch, learning_rate_scheduling, eps)
        self.gamma = gamma

    def gradient(self, indexes, w, epoch, lr, log):
        new_coefficients = w.copy()
        for j in range(len(w)):
            new_coefficients[j] -= log[-1][j] * self.gamma
        new_gradient = self.regression.MSE_gradient(new_coefficients, indexes)
        for j in range(len(w)):
            previous = log[-1][j] * self.gamma
            w[j] = w[j] - (previous + self.lr * new_gradient[j])
        return new_gradient


class AdagradGradientDescent(GradientDescent):
    def __init__(self, Regression, batch_size=None, learning_rate=None, max_epoch=100, learning_rate_scheduling=None, eps=1e-4):
        super().__init__(Regression, batch_size, learning_rate, max_epoch, learning_rate_scheduling, eps)
        self.log = None
        self.G = None

    def gradient(self, indexes, w, epoch, lr, log):
        if len(log) == 1:
            return super().gradient(indexes, w, epoch, lr, log)
        if self.G is None:
            self.G = [0.0 for _ in range(len(log[0]))]
        for k in range(len(log[-1])):
            self.G[k] += log[-1][k] ** 2
        result = []
        for g in self.G:
            result.append(1e-1 / math.sqrt(g + 1e-8))
        self.log = result
        gradient = self.regression.MSE_gradient(w, indexes)

        return gradient


def exponential_moving_average(q_last, ema_coefficient, q_one):
    return q_one * ema_coefficient + (1 - ema_coefficient) * q_last


class RMSPropGradientDescent(GradientDescent):
    def __init__(self, Regression, batch_size=None, learning_rate=None, max_epoch=100, learning_rate_scheduling=None, eps=1e-4,
                 beta=0.1):
        super().__init__(Regression, batch_size, learning_rate, max_epoch, learning_rate_scheduling, eps)
        self.beta = beta
        self.ema_grad = None

    def gradient(self, indexes, w, epoch, lr, log):
        gradient = self.regression.MSE_gradient(w, indexes)

        if self.ema_grad is None:
            self.ema_grad = [0.0 for _ in range(len(w))]

        for j in range(len(w)):
            self.ema_grad[j] = exponential_moving_average(self.ema_grad[j], self.beta, gradient[j] ** 2)

        for j in range(len(w)):
            w[j] = w[j] - (lr * gradient[j] / (math.sqrt(self.ema_grad[j]) + 1e-8))
        return gradient
        # return


class AdamGradientDescent(GradientDescent):
    def __init__(self, Regression, batch_size=None, learning_rate=None, max_epoch=100, learning_rate_scheduling=None, eps=1e-4,
                 beta1=0.1, beta2=0.1):
        super().__init__(Regression, batch_size, learning_rate, max_epoch, learning_rate_scheduling, eps)
        self.beta1 = beta1
        self.beta2 = beta2
        self.ema_grad = None
        self.ema_grad_sqr = None

    def gradient(self, indexes, w, epoch, lr, log):
        gradient = self.regression.MSE_gradient(w, indexes)

        if self.ema_grad is None:
            self.ema_grad = [0.0 for _ in range(len(w))]
        if self.ema_grad_sqr is None:
            self.ema_grad_sqr = [0.0 for _ in range(len(w))]

        p = len(log)
        for j in range(len(w)):
            self.ema_grad[j] = exponential_moving_average(self.ema_grad[j], self.beta1, gradient[j])
            self.ema_grad_sqr[j] = exponential_moving_average(self.ema_grad_sqr[j], self.beta2, gradient[j] ** 2)

        for j in range(len(w)):
            m = self.ema_grad[j] / (1 - math.pow(self.beta1, p))
            v = self.ema_grad_sqr[j] / (1 - math.pow(self.beta2, p))
            w[j] = w[j] - (lr * m / (math.sqrt(v) + 1e-8))
        return gradient
