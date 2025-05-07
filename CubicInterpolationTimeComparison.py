"""
This script just tests the execution time of different implementations of cubic interpolation.
"""
import Dysco
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline




##########################################################################################
# Test functions. Not part of timing
##########################################################################################
def test_cubic_interpolation():
    """Using scipy's cubic spline interpolator"""
    #spline = CubicSpline([0, 1, 4, 5, 7], [1, 3, 5, 4, 6])
    spline = CubicSpline([1, 4, 5, 7], [3, 5, 4, 6])
    x_, y_ = [], []
    x = 1.0
    while x <= 7.0:
        x_.append(x)
        y_.append(spline(x))
        x += 0.2
    plt.scatter(x_, y_)
    plt.show()

def cubic_solve(x, c):
    """Helper for function below"""
    return (c[0] * (x**3)) + (c[1] * (x**2)) + (c[2] * x) + c[3]

def test_cubic_interpolation2():
    """my own cubic interpolation by solving a linear system"""
    A = np.vander([1, 4, 5, 7], increasing=False)
    b = np.array([3, 5, 4, 6])
    coefficients = np.linalg.solve(A, b)

    x_, y_ = [], []
    x = 1.0
    while x <= 7.0:
        x_.append(x)
        y_.append(cubic_solve(x, coefficients))
        x += 0.2
    plt.scatter(x_, y_)
    plt.show()



##########################################################################################
# Timing tests
##########################################################################################

def compare_cubic_interpolator_speed():
    """
    compare the speed of my interpolator vs the interpolator in scipy for a cubic function.

    After comparing, mine seems a lot faster. But most of that is in calculating the values.
    I want to also do a test for constructing them
    """
    #create test x values
    test_values = [n/10 for n in range(1, 71)]

    def scipy_cubic():
        spline = CubicSpline([1, 4, 5, 7], [3, 5, 4, 6])
        for x in test_values:
            y = spline(x)

    def my_cubic():
        A = np.vander([1, 4, 5, 7], increasing=False)
        b = np.array([3, 5, 4, 6])
        c = np.linalg.solve(A, b) #coefficients
        for x in test_values:
            y = (c[0] * (x ** 3)) + (c[1] * (x ** 2)) + (c[2] * x) + c[3]

    scipy_time = Dysco.ExecutionTime(scipy_cubic)
    my_time = Dysco.ExecutionTime(my_cubic)

    print('----Test Many Evaluations----')
    print(f'{"scipy time":12}: {scipy_time: 11.7f}')
    print(f'{"my time":12}: {my_time: 11.7f}')



    #create a bunch of random coordinate pairs
    import random
    test_this = 3.5
    x_samples, y_samples = [], []
    test_count = 40
    for _ in range(test_count):
        #x_samples.append(sorted([random.randint(10, 70)/10 for _ in range(4)]))
        x_samples.append([random.randint(10, 24)/10, random.randint(25, 39)/10, random.randint(40, 54)/10, random.randint(55, 70)/10])
        y_samples.append([random.randint(30, 60)/10 for _ in range(4)])

    def my_cubic2():
        for x_sample, y_sample in zip(x_samples, y_samples):
            x1, x2, x3, x4 = x_sample
            y1, y2, y3, y4 = y_sample
            A = np.vander([x1, x2, x3, x4], increasing=False)
            b = np.array([y1, y2, y3, y4])
            c = np.linalg.solve(A, b)  # coefficients
            y = (c[0] * (test_this ** 3)) + (c[1] * (test_this ** 2)) + (c[2] * test_this) + c[3]

    def scipy_cubic2():
        for x_sample, y_sample in zip(x_samples, y_samples):
            x1, x2, x3, x4 = x_sample
            y1, y2, y3, y4 = y_sample
            spline = CubicSpline([x1, x2, x3, x4], [y1, y2, y3, y4])
            y = spline(test_this)

    scipy_time = Dysco.ExecutionTime(scipy_cubic2)
    my_time = Dysco.ExecutionTime(my_cubic2)

    print('----Test Many Constructions----')
    print(f'{"scipy time":12}: {scipy_time: 11.7f}')
    print(f'{"my time":12}: {my_time: 11.7f}')


def main():
    compare_cubic_interpolator_speed()

if __name__ == '__main__':
    main()
