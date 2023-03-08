def dichotomy_method(func, left, right, epsilon):
    iterations = 0
    middle = (left + right) / 2

    while right - left > epsilon:

        x1 = (left + middle) / 2
        x2 = (right + middle) / 2

        if func(x1) > func(middle) > func(x2):
            left = middle
        elif func(x1) < func(middle) < func(x2):
            right = middle
        else:
            left = x1
            right = x2

        middle = (left + right) / 2
        iterations += 1

    return middle, iterations


def f(x):
    return abs(x - 10)


print(dichotomy_method(f, 5, 12, 0.05))
