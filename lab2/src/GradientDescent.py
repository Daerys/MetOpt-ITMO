from random import sample
import numpy as np
import lab2.src.lab_2_utils as utils


# class field
class GradientDescent:
    def __init__(self, batch_size=None, learning_rate=None, max_epoch=100, learning_rate_scheduling=None):
        """
        :NOTE: We'll make a child classes which will
        override gradient function and learning_rate_scheduling will bw passed to init
        """
        self.batch_size = batch_size
        self.lr = learning_rate if learning_rate else 1.0
        self.max_epoch = max_epoch
        self.learning_rate_scheduling = learning_rate_scheduling if learning_rate_scheduling else self.constant

    def gradient(self, X, w):
        """
        :NOTE: w(weights) = len(X[0]). We'll count X[i][-1] = 1 and "expected value" is X[i][-1]
        That's how w[-1] is variable without x_i
        """
        x, y = utils.split(X)
        gradient = utils.MSE_gradient(x, y)
        return gradient(w)

    @staticmethod
    def constant(lr):
        return lr

    def run(self, data, start_weights):
        data = np.asarray(data)
        start_weights = np.asarray(start_weights)
        w = start_weights.astype(np.float64).reshape(-1, 1)
        log = [start_weights]
        for _ in range(self.max_epoch):
            indices = sample(range(len(data)), self.batch_size if self.batch_size else len(data))
            X = data[indices]
            w -= self.learning_rate_scheduling(self.lr) * self.gradient(X, w)
            log.append(w.copy().ravel())
        return log, w


# class field end

"""
Example of tasks 2-3:
task2:
def exp():
    def foo(old_lr)
        IMPLEMENTATION
        return new_lr
    return foo

task3:
class BruhGradient(GradientDescent):
    def __init__(...) <- if needed

    def gradient(...) <- overriding old gradient
"""
