import random

import numpy as np


def generate_points(f, x_range, eps):
    x_min, x_max = x_range
    while True:
        x = random.uniform(x_min, x_max)
        y = f(x) + random.uniform(-eps, eps)
        yield (x, y)


def f(x):
    return np.exp(2 * x)


# x = np.linspace(0, 10, 10)
# y = x + 5 * np.random.randn(10)
points = generate_points(f, (-10, 10), 0.5)
with open("../../lab3/src/dataexp.csv", "w") as f:
    f.write("x,y\n")
    for i in range(10):
        x, y = next(points)
        f.write("{:.2f},{:.2f}\n".format(x, y))
