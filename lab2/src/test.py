import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lab2.src.GradientDescent as GD
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
    R = 1  # epoch from below
    data_collection = ['data1.csv', 'data2.csv', 'data3.csv', 'datapoly1']
    batch_sizes = [25, 50]
    learning_rates = [5.8511e-2, 1e-3, 1e-2, 1e-1, 1, 10]
    epoches = [1e4, 1e5]

    data = pd.read_csv(data_collection[N])
    X = np.asarray(data['x'])
    Y = np.asarray(data['y'])

    data_set = utils.merge(X, Y)
    X = utils.extend(X)
    # start = np.array([100000, 100000])
    start = np.zeros(utils.K + 1)
    # start = np.array([random() * 10000 for _ in range(len(X[0]))])

    # grad = GD.GradientDescent(batch_size=60, learning_rate=1e-4, max_epoch=int(epoches[R]),
    #                          learning_rate_scheduling=utils.exponential_decay(1e-4))
    # grad = GD.GradientDescent(learning_rate=learning_rates[K], max_epoch=int(epoches[R]))
    grad = GD.GradientDescent(learning_rate=learning_rates[K], max_epoch=int(epoches[R]),
                             learning_rate_scheduling=utils.exponential_decay(learning_rates[K]))
    # grad = GD.RMSPropGradientDescent(learning_rate=learning_rates[0], max_epoch=int(epoches[R]),
    #                       learning_rate_scheduling=utils.exponential_decay(learning_rates[0]))
    # grad = GD.AdagradGradientDescent(learning_rate=learning_rates[0], max_epoch=int(epoches[R]),
    #                         learning_rate_scheduling=utils.exponential_decay(learning_rates[0]))
    # grad = GD.AdamGradientDescent(learning_rate=learning_rates[0], max_epoch=int(epoches[R]),
    #                                 learning_rate_scheduling=utils.exponential_decay(learning_rates[0]))
    # grad = GD.MomentumGradientDescent(batch_size=60, learning_rate=1e-4, max_epoch=10000,
    #                                   learning_rate_scheduling=utils.exponential_decay(1e-4), gamma=1e-8)
    # grad = GD.NesterovGradientDescent(batch_size=60, learning_rate=1e-4, max_epoch=10000,
    #                                   learning_rate_scheduling=utils.exponential_decay(1e-4), gamma=1e-10)
    # grad = GD.GradientDescent(learning_rate=learning_rates[K], max_epoch=int(epoches[R]))
    # mse = utils.MSE(X, Y)
    points, w = grad.run(data_set, start)

    X, Y = utils.split(data_set)
    mse = utils.MSE_poly(np.squeeze(X[:, :-1]), Y, utils.K)
    # plot(w, data_set)
    poly_draw(w, data_set)
    # print(len(points))
    # draw(mse, points)
    # set_of_points - массив из точек на плоскости. w - коэффициенты
    # полинома в виде: [x, x^2, ... , x^n, x^0]. Для y = (x + 1)^2 w = [2, 1, 1]