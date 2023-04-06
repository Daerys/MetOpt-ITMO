from random import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lab2.src.GradientDescent as GD
import lab2.src.lab_2_utils as utils

data_set = []


def draw(Z, pts):
    pts = np.asarray(pts)
    x1 = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 100)
    x2 = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 100)
    x1, x2 = np.meshgrid(x1, x2)
    Z_values = np.zeros_like(x1)
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            Z_values[i, j] = Z([x1[i, j], x2[i, j]])
    levels = np.unique([Z(pt) for pt in pts])
    plt.contour(x1, x2, Z_values, levels=np.sort(levels))
    plt.plot(pts[:, 0], pts[:, 1], '-o', c='red')
    plt.show()


def plot(weights, set_of_points):
    X = np.asarray(set_of_points[:, :-1])
    Y = np.asarray(set_of_points[:, -1:])
    plt.plot(X, Y, 'o')
    xmax = max(map(lambda o: o[0], set_of_points))
    # print(xmax)
    # print(weights[0])
    # print(type(xmax))
    # print(type(weights[0]))

    plt.plot([0, xmax], [weights[1], weights[1] + xmax * weights[0]], color='orange')
    plt.show()


if __name__ == "__main__":
    N = 1  # data set from below
    M = 0  # batch size from below
    K = 0  # learning rate from below
    R = 0  # epoch from below
    data_collection = ['data1.csv', 'data2.csv']
    batch_sizes = [1, 16]
    learning_rates = [1e-3, 1e-2, 1e-1, 1]
    epoches = [1e4, 1e5]

    data = pd.read_csv(data_collection[N])
    X = np.asarray(data['x'])
    Y = np.asarray(data['y'])

    data_set = utils.merge(X, Y)
    X = utils.extend(X)
    start = np.zeros(len(X[0]))
    # start = np.array([random() * 10000 for _ in range(len(X[0]))])

    grad = GD.GradientDescent(batch_size=batch_sizes[M], learning_rate=learning_rates[K], max_epoch=int(epoches[R]))
    # grad = GD.GradientDescent(learning_rate=learning_rates[K], max_epoch=int(epoches[R]))
    points, w = grad.run(data_set, start)

    X, Y = utils.split(data_set)
    mse = utils.MSE(X, Y)
    plot(w, data_set)
    draw(mse, points)
