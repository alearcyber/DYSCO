import numpy as np
import cv2
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt



def draw_bezier_example():
    t = np.array([0,0,0,0,1,1,1,1]) # knots
    c = np.array([[2,4],[6,10],[10,9],[14, 2]]) #control points
    k = 3 #degree
    spline = BSpline(t=t, c=c, k=k) #Spline
    plt.scatter(c[:, 0], c[:, 1], color="r", s=10)

    for x in np.linspace(0, 1, 400):
        p = spline(x)
        plt.scatter(p[0], p[1], c='b', s=1)
    plt.show()



def draw_surface_example():
    """
    This draws a surface using B splines that looks like a flag waving in the wind.
    Thew knots are such a way that it really becomes a cubic bezier surface.
    """
    t = np.array([0, 0, 0, 0, 1, 1, 1, 1]) #knots (used for both directions
    flag_control_points = np.array([
        [[0, 15], [2,14], [7, 14], [11, 15]],
        [[0,10], [4, 11], [8, 11], [12, 12]],
        [[0, 5], [5,6], [8, 6], [12, 7]],
        [[0, 0], [3,1], [7, 1], [11, 3]]
    ])


    #draw control points
    for i in range(0, 4):
        for k in range(0, 4):
            control_point = flag_control_points[i, k]
            plt.scatter(control_point[0], control_point[1], c='r', s=12)


    #draw horizontal curves
    h_splines = []
    for row in range(0, 4):
        points = flag_control_points[row, :, :]
        spline = BSpline(t=t, c=points, k=3)
        h_splines.append(spline)
        for x in np.linspace(0, 1, 400):
            p = spline(x)
            plt.scatter(p[0], p[1], c='b', s=2)

    #draw vertical curves
    v_splines = []
    for col in range(0, 4):
        points = flag_control_points[:, col, :]
        spline = BSpline(t=t, c=points, k=3)
        v_splines.append(spline)
        for x in np.linspace(0, 1, 400):
            p = spline(x)
            plt.scatter(p[0], p[1], c='b', s=2)


    #draw spacers splines, the more granular ones
    for x in np.linspace(0, 1, 14):
        points_h = np.array([spline(x) for spline in v_splines])
        points_v = np.array([spline(x) for spline in h_splines])
        spacer_spline_h = BSpline(t=t, c=points_h, k=3)
        spacer_spline_v = BSpline(t=t, c=points_v, k=3)
        for x in np.linspace(0, 1, 200):
            p = spacer_spline_h(x)
            plt.scatter(p[0], p[1], c='g', s=1)
            p = spacer_spline_v(x)
            plt.scatter(p[0], p[1], c='g', s=1)

    plt.show()


def draw_sparse_spline_pointmap():
    """
    Approximates the above surface with piecewise linear points instead of splines
    """
    n_partitions = 20
    #knots and control points
    t = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # knots (used for both directions
    flag_control_points = np.array([
        [[0, 15], [2, 14], [7, 14], [11, 15]],
        [[0, 10], [4, 11], [8, 11], [12, 12]],
        [[0, 5], [5, 6], [8, 6], [12, 7]],
        [[0, 0], [3, 1], [7, 1], [11, 3]]
    ])

    # draw control points
    for i in range(0, 4):
        for k in range(0, 4):
            control_point = flag_control_points[i, k]
            plt.scatter(control_point[0], control_point[1], c='r', s=12)


    #construct horizontal splines
    h_splines = []
    for row in range(0, 4):
        points = flag_control_points[row, :, :]
        spline = BSpline(t=t, c=points, k=3)
        h_splines.append(spline)


    #draw point map
    for l1 in np.linspace(0, 1, n_partitions):
        control_points = np.array([h_spline(l1) for h_spline in h_splines])
        v_spline = BSpline(t=t, c=control_points, k=3)
        for l2 in np.linspace(0, 1, n_partitions):
            p = v_spline(l2)
            plt.scatter(p[0], p[1], c='g', s=1)

    plt.show()



def approximate_spline_surface():
    """
    Approximates the above surface with piecewise linear points instead of splines
    """
    n_partitions = 20
    #knots and control points
    t = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # knots (used for both directions
    flag_control_points = np.array([
        [[0, 15], [2, 14], [7, 14], [11, 15]],
        [[0, 10], [4, 11], [8, 11], [12, 12]],
        [[0, 5], [5, 6], [8, 6], [12, 7]],
        [[0, 0], [3, 1], [7, 1], [11, 3]]
    ])

    # draw control points
    for i in range(0, 4):
        for k in range(0, 4):
            control_point = flag_control_points[i, k]
            plt.scatter(control_point[0], control_point[1], c='r', s=12)

    #spline points
    spline_surface = np.empty((n_partitions, n_partitions, 2), dtype=np.float64)

    h_splines = [] # horizontal splines
    for row in range(0, 4):
        points = flag_control_points[row, :, :]
        spline = BSpline(t=t, c=points, k=3)
        h_splines.append(spline)


    #create displacement map
    c = 0
    for l1 in np.linspace(0, 1, n_partitions):
        control_points = np.array([h_spline(l1) for h_spline in h_splines])
        v_spline = BSpline(t=t, c=control_points, k=3)
        r = 0
        for l2 in np.linspace(0, 1, n_partitions):
            p = v_spline(l2)
            spline_surface[r, c] = p
            r += 1
        c += 1


    #draw it to ensure stored correctly
    for r in range(n_partitions):
        for c in range(n_partitions):
            p = spline_surface[r, c]
            plt.scatter(p[0], p[1], c='g', s=1)
    plt.show()













def deBoor(k: int, x: int, t, c, p: int):
    """
    From wikipedia

    Evaluates S(x).

    Arguments
    ---------
    k: Index of knot interval that contains x.
    x: Position.
    t: Array of knot positions, needs to be padded as described above.
    c: Array of control points.
    p: Degree of B-spline.
    """
    d = [c[j + k - p] for j in range(0, p + 1)]

    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p])
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

    return d[p]




def SplineSurfaceImageTransform():
    #spline parameters. Will look like a flag
    t = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # knots (used for both directions
    flag_control_points = np.array([
        [[0, 15], [2, 14], [7, 14], [11, 15]],
        [[0, 10], [4, 11], [8, 11], [12, 12]],
        [[0, 5], [5, 6], [8, 6], [12, 7]],
        [[0, 0], [3, 1], [7, 1], [11, 3]]
    ])


approximate_spline_surface()
