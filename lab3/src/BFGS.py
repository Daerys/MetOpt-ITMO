from random import sample

from lab3.src.Gradient import GradientDescent

import numpy as np
import scipy as sp


class BFGS(GradientDescent):
    function = None
    hinv = None
    lr_policy = None
    c1 = None
    c2 = None
    beta = None
    prev_g = None
    debug = False

    def __init__(
            self,
            Regression,
            batch_size=None,
            learning_rate=None,
            max_epoch=100,
            learning_rate_scheduling=None,
            eps=1e-4,
            beta=0.1,
            hinv=None,
    ):
        super().__init__(Regression, batch_size, learning_rate, max_epoch, learning_rate_scheduling, eps)
        self.hinv = hinv
        self.beta = beta

    def gradient(self, indexes, w, epoch, lr, log):
        if self.hinv is None:
            self.hinv = 0.001 * np.eye(len(w))

        #print(w)
        grad = self.regression.MSE_gradient(w)
        #print(grad)
        p = -np.matmul(self.hinv, grad)
        #print(p)

        ls = sp.optimize.line_search(self.regression.MSE, self.regression.MSE_gradient, w, p)
        k = ls[0]
        #print(ls)
        #print(k)
        new_w = w + p * k
        s = new_w - w
        w = new_w

        new_grad = self.regression.MSE_gradient(new_w)
        y = new_grad - grad
        I = np.eye(len(w))
        ro = 1.0 / (np.dot(y, s))
        A1 = I - ro * s[:, np.newaxis] * y[np.newaxis, :]
        A2 = I - ro * y[:, np.newaxis] * s[np.newaxis, :]
        self.hinv = np.dot(A1, np.dot(self.hinv, A2)) + (ro * s[:, np.newaxis] *
                                                         s[np.newaxis, :])

        return new_grad, new_w

    def run(self):
        data_size = len(self.regression.data_set)
        w = self.regression.get_coefficients().copy()
        log = [w.copy()]
        for epoch in range(self.max_epoch):
            indexes = sample(range(data_size), self.batch_size if self.batch_size else data_size)

            self.lr = self.learning_rate_scheduling(epoch)
            gr, w = self.gradient(indexes, w, epoch, self.lr, log)  # gr = (w / self.lr - wp * delta)

            log.append(w.copy().ravel())
            if np.linalg.norm(gr) < self.eps or np.linalg.norm(log[-1] - log[-2]) < self.eps:
                break

        return log, w
