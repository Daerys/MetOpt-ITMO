import numpy as np
import matplotlib.pyplot as plt
import lab1.src.dragientDecent.grad as grad
import lab1.src.generator.quadratic_form_generator as gen
import pandas as pd


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


def draw_surface(X, Y, Z):
    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z(np.stack((X, Y))), cmap='twilight')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def draw(X, Y, Z, points: np.array):

    levels = []
    for i in points:
        levels.append(Z(i))
    levels = np.sort(levels)

    cp = plt.contour(X, Y, Z([X, Y]), levels, linewidths=1)
    plt.title("%s итераций" % len(points))
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
    gradient = grad.GradientDescent(optimization_mod, stop_criteria, fun, grad_fun)
    gradient.set_start_point(start_point)
    gradient.change_learning_rate(lr)
    gradient.change_max_iter(max_iter)
    return gradient.gradient()


if __name__ == '__main__':
    lrs = [1, 0.5, 0.1, 0.01, 1e-3, 1e-4]
    sp = [[-1, 1], [25, 25], [-200, 200], [100.9, 1e5]]
    mods = ['dichotomy', 'constant']
    criteria = ['max_iter', 'gradient']

    Z = par_var2()
    nabla_Z = grad_par_var2()
    t = np.linspace(-20, 20, 1000)
    X, Y = np.meshgrid(t, t)
    points = do_gradient(mods[1], criteria[1], [-20, -10], lrs[3], Z, nabla_Z, 200)
    draw(X, Y, Z, points)
