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

    def gradient(self, X, w):
        """
        :NOTE: w(weights) = len(X[0]). We'll count X[i][-1] = 1 and "expected value" is X[i][-1]
        That's how w[-1] is variable without x_i
        """
        x, y = utils.split(X)
        gradient = utils.MSE_gradient(x, y)
        return gradient(w)

    @staticmethod
    def constant(lr, _):
        return lr

    def run(self, data, start_weights):
        data = np.asarray(data)
        start_weights = np.asarray(start_weights)
        w = start_weights.astype(np.float64).reshape(-1, 1)
        log = [start_weights]
        for epoch in range(self.max_epoch):
            indices = sample(range(len(data)), self.batch_size if self.batch_size else len(data))
            X = data[indices]

            gr = self.gradient(X, w)
            w -= self.learning_rate_scheduling(self.lr, epoch) * gr

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
