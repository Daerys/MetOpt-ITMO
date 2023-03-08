import numpy as np

lambda_const = 1e5
max_iterations_const = 50


def grad():
    return lambda x: [2 * x[0], 97 * x[1]]


def gradient_descent(x, gradient, learning_rate, iterations):
    points = np.zeros((iterations, len(x)))
    points[0] = x
    for i in range(1, iterations):
        x -= int(learning_rate) * np.array(gradient(x))
        points[i] = x

    print(points)


if __name__ == '__main__':
    gradient_descent([5, 7], grad(), lambda_const, max_iterations_const)
    # return x
