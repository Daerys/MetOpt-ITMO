import numpy as np
import lab1.src.test.different_method_tester as ef


if __name__ == '__main__':

    a = -10
    b = 10
    mods = ['dichotomy', 'constant', 'wolfe']
    criteria = ['max_iter', 'gradient', 'mixed']
    functions = {
        'ex1': (lambda x: b * np.sin(a * x[0]) / (a * x[0]), lambda x: b * (a * x[0] * np.cos(a * x[0]) - np.sin(a * x[0])) / (a*x[0]))
    }
    Y, nabla_Y = functions['ex1']

    points = ef.do_gradient(mods[2], criteria[2], [0.015], 0.01, Y, nabla_Y, 150)
    ef.draw_function(np.linspace(-5, 25, 1000), Y, points)
