from random import sample
import numpy as np


def me(data, y, w):
    ans = 0
    for i in range(len(data)):
        ans += w[i] * data[i]
    ans -= y
    return ans


# class field
class GradientDescent:
    def __int__(self, batch_size=None, learning_rate=None, max_epoch=100, learning_rate_scheduling=None):
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
        # mse = mse_calc()
        data = [x[:-1] + [1] for x in X]
        y = [y[-1] for y in X]
        result = np.zeros(len(w))

        for i in range(len(w)):
            for j in range(len(data)):
                result[i] += data[j][i] * me(data[j], y[j], w)
            result[i] *= 2

        return np.array(result)

    @staticmethod
    def constant(lr):
        return lr

    def run(self, data, start_weights):
        if len(data) < 1 or len(data[0]) < 2:
            raise ValueError("data has to represent at least 2d space")
        w = start_weights
        log = [start_weights]
        for _ in range(self.max_epoch):
            X = sample(data, self.batch_size if self.batch_size else len(data))
            w -= self.learning_rate_scheduling(self.lr) * self.gradient(X, w)
            log.append(w)
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
