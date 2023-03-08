import numpy as np
import lab1.src.dragientDecent.grad as g

if __name__ == '__main__':
    # y = x^2
    quadr = np.vectorize(lambda x: x**2)
    quadr_grad = np.vectorize(lambda x: 2 * x)
    a = g.GradientDescent('dichotomy',
                          'gradient',
                          quadr,
                          quadr_grad)
    a.set_start_point(10)
    a.gradient()
