import numpy as np

lambda_const = 1e5
max_iterations_const = 10
eps = 1e-4


def grad():
    return lambda x: [2 * x[0], 97 * x[1]]


def stop_crit(x, gradient, epsilon):
    norm_grad = np.linalg.norm(gradient(x))
    if norm_grad < epsilon:
        return False
    else:
        return True


def gradient_descent(x, gradient, learning_rate, iterations, stop_criteria):
    if iterations is None:
        points = np.zeros(1, len(x))
        points[0] = x
        while stop_criteria:
            x -= int(learning_rate) * np.array(gradient(x))
            points = np.append(points, x, axis=0)
    else:
        points = np.zeros((iterations, len(x)))
        points[0] = x
        for i in range(1, iterations):
        #  learning_rate = middle
            x -= int(learning_rate) * np.array(gradient(x))
            points[i] = x

    print(points)


if __name__ == '__main__':
    gradient_descent([5, 7], grad(), lambda_const, max_iterations_const, stop_crit([5, 7], grad(), eps))
    # return x
