def dichotomy_method(f, left, right, epsilon):

    iterations = 0
    middle = (left + right) / 2
    
    while right - left > epsilon:

        x1 = (left + middle) / 2
        x2 = (right + middle) /2

        if f(x1) > f(middle) and f(middle) > f(x2):
            left = middle
        elif f(x1) < f(middle) and f(middle) < f(x2):
            right = middle
        else:
            left = x1
            right = x2

        middle = (left + right) / 2
        iterations += 1

    return middle, iterations

def f(x):
    return abs(x-10)

print(dichotomy_method(f, 5, 12, 0.05))
