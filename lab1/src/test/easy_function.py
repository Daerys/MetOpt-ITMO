import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import lab1.src.dragientDecent.grad as g


def gip_par():
    def f(x):
        return x[0] ** 2 - x[1] ** 2

    return f


def grad_gip_par():
    def f(x):
        grad = np.zeros_like(x)
        grad[0] = 2 * x[0]
        grad[1] = -2 * x[1]
        return grad

    return f


def par():
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    return f


def grad_par():
    def f(x):
        grad = np.zeros_like(x)
        grad[0] = 2 * x[0]
        grad[1] = 2 * x[1]
        return grad

    return f


C = 15


def par_var2():
    def f(x):
        return C * x[0] ** 2 + x[1] ** 2 - 2 * x[0]

    return f


def grad_par_var2():
    def f(x):
        grad = np.zeros_like(x)
        grad[0] = C * 2 * x[0]
        grad[1] = 2 * x[1] - 2
        return grad

    return f


def grad_descent(learning_rate, iterations, initial_point, grad):
    points = np.zeros((iterations, 2))
    points[0] = initial_point

    for i in range(1, iterations):
        prev_point = points[i - 1]
        points[i] = prev_point - learning_rate * np.array(grad(*prev_point))
    return points


if __name__ == '__main__':
    # f = gip_par()
    # Gradient = g.GradientDescent('dichotomy',
    #                              'max_iter',
    #                              gip_par(),
    #                              grad_gip_par()
    #                              )
    #
    # f = par_var2()
    # Gradient = g.GradientDescent('dichotomy',
    #                              'max_iter',
    #                              par_var2(),
    #                              grad_par_var2()
    #                              )

    f = par()
    Gradient = g.GradientDescent('constant',
                                 'max_iter',
                                 par(),
                                 grad_par()
                                 )
    Gradient.change_learning_rate(0.1)
    Gradient.set_start_point([-20, -10])
    Gradient.change_max_iter(30)
    points = Gradient.gradient()
    # points = grad_descent(0.5, 20, [20,20], grad_par())
    print(points)
    t = np.linspace(-20, 20, 100)
    X, Y = np.meshgrid(t, t)
    Z = par()
    # my_col = cm.jet(Z([X, Y]) / np.amax(Z([X, Y])))
    # ax = plt.subplot(111, projection='3d')
    # ax.plot_surface(X, Y, f(np.stack((X, Y))), facecolors = my_col)
    # plt.show()
    levels = []
    for i in points:
        levels.append(Z(i))
    levels = np.sort(levels)
    # print(levels)
    # print(points)
    cp = plt.contour(X, Y, Z([X, Y]), levels, linewidths=1)
    plt.plot(points[:, 0], points[:, 1], '-*', linewidth=1, color='r')
    # cp.plot(levels)
    plt.clabel(cp, inline=1, fontsize=10)
    plt.show()

    # t = np.linspace(-30, 30, 100)
    # X, Y = np.meshgrid(t, t)
    # ax = plt.subplot(111, projection='3d')
    # ax.plot_surface(X, Y, f(np.stack((X, Y))))
    # plt.show()
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.plot(f(points.T))
    # ax1.grid()
    # ax2.plot(points[:, 0], points[:, 1], 'o-')
    # ax2.contour(X, Y, f(np.stack((X, Y))), levels=np.sort(np.concatenate((f(np.sort(points.T)), np.linspace(1, 20, 10)))))
    # print(points[-1], f(points[-1]))
    # plt.show()

    # X, Y = np.meshgrid(np.arange(-1e20, 1e20), np.arange(-1e20, 1e20))
    # graddat = grad_gip_par()([X, Y])
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.plot(points[:, 0], points[:, 1], 'o-')
    #
    # plt.figure()
    # ax1.quiver(X, Y, graddat[0], graddat[1])
    # plt.show()
