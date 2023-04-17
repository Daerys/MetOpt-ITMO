import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lab3.Gradient as G
import lab3.PolynomialRegression as Poly
import lab2.src.lab_2_utils as utils

data_set = []


def poly_draw(w, set_of_points=[[0, 1], [4, 5], [2, 6]]):
    set_of_points = np.asarray(set_of_points)
    Z = utils.poly_fun(w)
    plt.plot(np.asarray(set_of_points[:, :-1]),
             np.asarray(set_of_points[:, -1:]), 'o')
    X = np.linspace(min(set_of_points[:, :-1]), max(set_of_points[:, :-1]), 100)
    Y = Z(X)
    plt.plot(X, Y)
    plt.show()
    print(w)


def draw(Z, pts):
    plt.title(f"{len(pts)} итераций\nнайденные коэффициенты {pts[-1]}")
    pts = np.asarray(pts)
    x1 = np.linspace(pts[:, 0].min() - 0.5 * pts[:, 0].max(), pts[:, 0].max() + 0.5 * pts[:, 0].max(), 100)
    x2 = np.linspace(pts[:, 1].min() - 0.5 * pts[:, 1].max(), pts[:, 1].max() + 0.5 * pts[:, 1].max(), 100)
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
    plt.plot([0, xmax], [weights[1], weights[1] + xmax * weights[0]], color='red')
    plt.show()


if __name__ == "__main__":
    N = 3  # data set from below
    M = 0  # batch size from below
    K = 0  # learning rate from below
    R = 0  # epoch from below
    data_collection = []
    batch_sizes = []
    learning_rates = []
    epoches = []

    data = pd.read_csv(data_collection[N])
    X = np.asarray(data['x'])
    Y = np.asarray(data['y'])

    data_set = utils.merge(X, Y)

    poly = Poly.PolynomialRegression(data_set, 10)
    # gradient = G.GradientDescent(Regression=poly, learning_rate=learning_rates[K], max_epoch=int(epoches[R]),
    #                              learning_rate_scheduling=utils.exponential_decay(learning_rates[K]))
    gradient = G.RMSPropGradientDescent(Regression=poly, learning_rate=learning_rates[0], max_epoch=int(epoches[R]),
                          learning_rate_scheduling=utils.exponential_decay(learning_rates[0]))
    # gradient = G.AdagradGradientDescent(Regression=poly, learning_rate=learning_rates[0], max_epoch=int(epoches[R]),
    #                         learning_rate_scheduling=utils.exponential_decay(learning_rates[0]))
    # gradient = G.AdamGradientDescent(Regression=poly, learning_rate=learning_rates[0], max_epoch=int(epoches[R]),
    #                                 learning_rate_scheduling=utils.exponential_decay(learning_rates[0]))
    # gradient = G.MomentumGradientDescent(Regression=poly, batch_size=60, learning_rate=1e-4, max_epoch=10000,
    #                                   learning_rate_scheduling=utils.exponential_decay(1e-4), gamma=1e-8)
    # gradient = G.NesterovGradientDescent(Regression=poly, batch_size=60, learning_rate=1e-4, max_epoch=10000,
    #                                   learning_rate_scheduling=utils.exponential_decay(1e-4), gamma=1e-10)
    points, w = gradient.run()
    poly_draw(w, data_set)
