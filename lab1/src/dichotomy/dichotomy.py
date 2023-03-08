def dichotomy_method(func, epsilon):
    iterations = 0
    left = 0
    right = 15
    delta = epsilon / 1.5

    while abs(right - left) < epsilon:

        x1 = (left + right) / 2 - delta
        x2 = (left + right) / 2 + delta

        if func(x1) >= func(x2):
            left = x1
        else:
            right = x2

        iterations += 1

        return (left + right) / 2, iterations


def f(x):
    return abs(x - 10)


print(dichotomy_method(f, 0.05))
