import numpy as np
import matplotlib.pyplot as plt
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


C1 = 15
C2 = 0.3


def par_var2():
    def f(x):
        return C1 * x[0] ** 2 + C2 * x[1] ** 2

    return f


def grad_par_var2():
    def f(x):
        grad = np.zeros_like(x)
        grad[0] = C1 * 2 * x[0]
        grad[1] = C2 * 2 * x[1]
        return grad

    return f


def draw(Z, points: np.array):
    t = np.linspace(-20, 20, 100)
    X, Y = np.meshgrid(t, t)
    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z(np.stack((X, Y))), cmap='twilight')
    plt.show()
    levels = []
    for i in points:
        levels.append(Z(i))
    levels = np.sort(levels)
    cp = plt.contour(X, Y, Z([X, Y]), levels, linewidths=1)
    plt.plot(points[:, 0], points[:, 1], '-*', linewidth=1, color='r')
    plt.clabel(cp, inline=1, fontsize=10)
    plt.show()


def do_gradient(optimization_mod,
                stop_criteria,
                start_point,
                lr,
                fun,
                grad_fun,
                max_iter: int):
    gradient = g.GradientDescent(optimization_mod, stop_criteria, fun, grad_fun)
    gradient.set_start_point(start_point)
    gradient.change_learning_rate(lr)
    gradient.change_max_iter(max_iter)
    return gradient.gradient()


if __name__ == '__main__':
    lrs = [1, 0.5, 0.1, 0.01, 1e-3, 1e-4]
    mods = ['dichotomy', 'constant']
    criteria = ['max_iter', 'gradient']
    for i in lrs:
        Z = par()
        # points = do_gradient()
        # draw(Z, points)
