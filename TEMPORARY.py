import numpy as np
import matplotlib.pyplot as plt


######################################################################
# BASIS FUNCTION
######################################################################
def basis_function(i, k, x, T):
    if k == 0:
        return 1.0 if T[i] <= x < T[i + 1] else 0.0
    else:
        coef1 = (x - T[i]) / (T[i + k] - T[i]) if T[i + k] != T[i] else 0
        coef2 = (T[i + k + 1] - x) / (T[i + k + 1] - T[i + 1]) if T[i + k + 1] != T[i + 1] else 0
        return coef1 * basis_function(i, k - 1, x, T) + coef2 * basis_function(i + 1, k - 1, x, T)


######################################################################
# ITERATES OVER AND EVALUATES ALL THE SEGMENTS
######################################################################
def evaluate_b_spline(x, k, control_points, T):
    n = len(control_points) - 1
    point = np.zeros(2)
    for i in range(n + 1):
        b = basis_function(i, k, x, T)
        point += b * control_points[i]
    return point


######################################################################
# ITERATES OVER AND EVALUATES ALL THE SEGMENTS
######################################################################
def main():

    #PARAMETERS
    k = 3 #degree
    control_points = np.array([[0, 0], [1, 2], [3, 3], [4, 2]]) #control points (n = 3 segments)
    T = np.array([0, 0, 0, 0, 1, 1, 1, 1]) # knot vector

    #HERE IS THE BUG, Bspline evaluated at 1 should be (4,2), but instead, it just gives me (0,0)
    print("Bug Here:", evaluate_b_spline(x=1, k=k, control_points=control_points, T=T))

    #This line shows how it gets _close_ to the proper value when x is almost 1.
    print("Proper behavior:", evaluate_b_spline(x=0.999, k=k, control_points=control_points, T=T))

    """
    #This next part visualizes the B-Spline
    x_values = np.linspace(T[k], T[-k-1], 1000) # Values over which to interpolate
    curve_points = np.array([evaluate_b_spline(x, k, control_points, T) for x in x_values]) # Evaluate along curve
    plt.plot(control_points[:, 0], control_points[:, 1], 'ro--', label='Control Points')  # Control polygon
    plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='B-spline Curve')  # B-spline curve
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('B-spline Curve')
    plt.show()
    """

main()
