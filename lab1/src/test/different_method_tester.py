import numpy as np
import matplotlib.pyplot as plt
import lab1.src.dragientDecent.grad as grad
from mpl_toolkits.mplot3d import Axes3D


def draw_surface(X, Y, Z):
    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z(np.stack((X, Y))), cmap='twilight')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=10)
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


def draw_function(F, points):
    X = np.linspace(-1, 1, 1000)
    plt.title("%s итераций" % len(points))
    plt.plot(X, F([X]))
    plt.plot(points, F([points]), '-o', linewidth=0.5, color='r')
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
    mods = ['dichotomy', 'constant', 'wolfe']
    criteria = ['max_iter', 'gradient', 'quadratic']

    C1 = 15
    C2 = 0.3

    functions = {
        'hyperboloid': (lambda x: x[0] ** 2 - x[1] ** 2, lambda x: np.array([2 * x[0], - 2 * x[1]])),
        'paraboloid': (lambda x: x[0] ** 2 + x[1] ** 2, lambda x: np.array([2 * x[0], 2 * x[1]])),
        'extended_paraboloid': (
            lambda x: C1 * x[0] ** 2 + C2 * x[1] ** 2, lambda x: np.array([2 * C1 * x[0], 2 * C2 * x[1]])),
        'abs': (lambda x: abs(x[0]), lambda x: np.array([np.sign(x[0])]))
    }
    Z, nabla_Z = functions['abs']
    # t = np.linspace(-1, 1, 1000)
    # X, Y = np.meshgrid(t, t)
    points = do_gradient(mods[0], criteria[2], [-0.5], lrs[2], Z, nabla_Z, 300)
    draw_function(Z, points)
    # draw(X, Y, Z, points)
    # draw_surface(X, Y, Z)
