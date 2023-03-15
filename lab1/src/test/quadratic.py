import numpy as np
import lab1.src.test.easy_function as ef
import lab1.src.generator.quadratic_form_generator as gen
import pandas as pd


def test_quadratic():
    global f, g, points
    f, g = gen.generate_quadratic(n, k)
    points = ef.do_gradient(mods[0],
                            criteria[2],
                            np.array(np.random.uniform(-5, 5, n)),
                            lrs[0],
                            f, g,
                            10000)
    results.append({'n': n, 'k': k, 'iterations': len(points)})


if __name__ == '__main__':
    lrs = [1, 0.5, 0.1, 0.01, 1e-3, 1e-4]
    mods = ['dichotomy', 'constant']
    criteria = ['max_iter', 'gradient', 'quadratic']

    results = []

    for n in range(2, 11):
        for k in range(1, 10):
            test_quadratic()
        for k in range(10, 100, 10):
            test_quadratic()
        for k in range(100, 1000, 100):
            test_quadratic()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    results_df = pd.DataFrame(results)
    pivoted_df = results_df.pivot(index='k', columns='n')

    pivoted_df.to_csv('pivoted_results.csv')
