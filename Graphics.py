"""
This is for graphics related algorithms
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

def Tri2TriTransform(source, destination):
    """
    Given a source and destination triangle, computes the affine transformation that would transform the source triangle
    to the destination triangle.

    Triangles should be some indexable with 3 points. Each point should be indexible where indices 0 and 1 are the
    first and second values.

    Transformations T1 and T2 are formulated that map the standard triangle, (0,0), (1,0), (0,1), to the source
    and destination respectively. Then form the full transformation, T = T2*T1^-1
    """
    x = np.array([[1, 0, 2],
                  [-1, 5, 0],
                  [0, 3, -9]])
    #break up triangles into points for readability
    p1, p2, p3 = source[0], source[1], source[2]
    q1, q2, q3 = destination[0], destination[1], destination[2]

    #formulate origin to source triangle
    T1 = np.array([[p2[0]-p1[0], p3[0]-p1[0], p1[0]],
                   [p2[1]-p1[1], p3[1]-p1[1], p1[1]],
                   [0, 0, 1]])

    # formulate origin to destination triangle
    T2 = np.array([[q2[0] - q1[0], q3[0] - q1[0], q1[0]],
                   [q2[1] - q1[1], q3[1] - q1[1], q1[1]],
                   [0, 0, 1]])

    #final Transformation
    T = T2 @ np.linalg.inv(T1)
    return T


def simple_bresenham(x1, y1, x2, y2):
    m_new = 2 * (y2 - y1)
    slope_error_new = m_new - (x2 - x1)

    y = y1
    for x in range(x1, x2 + 1):

        print("(", x, ",", y, ")")

        # Add slope to increment angle formed
        slope_error_new = slope_error_new + m_new

        # Slope error reached limit, time to
        # increment y and update slope error.
        if (slope_error_new >= 0):
            y = y + 1
            slope_error_new = slope_error_new - 2 * (x2 - x1)

def simple_bresenham_walker(lo, hi):
    x1, y1 = lo
    x2, y2 = hi

    m_new = 2 * (y2 - y1)
    slope_error_new = m_new - (x2 - x1)

    y = y1
    for x in range(x1, x2 + 1):

        yield x, y

        # Add slope to increment angle formed
        slope_error_new = slope_error_new + m_new

        # Slope error reached limit, time to
        # increment y and update slope error.
        if (slope_error_new >= 0):
            y = y + 1
            slope_error_new = slope_error_new - 2 * (x2 - x1)


def bresenham_walker_steep(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    xi = 1
    if dx < 0:
        x1 = -1
        dx = dx
    D = 2*dx - dy
    x = x1

    for y in range(y1, y2+1):
        yield x, y
        if D > 0:
            x = x + xi
            D = D + (2*(dx - dy))
        else:
            D = D + 2*dx





def parallel_bresenham(lo1, hi1, lo2, hi2):
    """
    This is an generator
    """


def bresenham_walker():
    """Here is where I'm going to do the yield thing"""

    #NOTE: INSTEAD OF RETURN A LIST OF THE POINTS, JUST YIELD WHATEVER THE NEXT POINTS ARE,
    # AND THEN CALL THE FUNCTION AS AN ITERABLE IN A LOOP.




BLEND_ALPHA = 1
BLEND_DIFFERENCE = 2
BLEND_CHECKERBOARD = 3
BLEND_COLORMAP = 4
def BlendImages(image1, image2, mode=BLEND_COLORMAP, alpha=0.5, block_size=1):
    #check images are the same shape
    assert image1.shape == image2.shape, "Images must be the same shape"

    #Alpha blending
    if mode == BLEND_ALPHA:
        alpha = 0.5
        blended_image = (alpha * image1) + ((1 - alpha) * image2)
        return blended_image.astype(np.uint8)

    #Checkerboard blending
    elif mode == BLEND_CHECKERBOARD:
        #dimensions
        rows, cols = image1.shape[0], image1.shape[1]

        # Create checkerboard mask
        checkerboard = np.zeros((rows, cols), dtype=bool)
        for i in range(0, rows, block_size * 2):
            for j in range(0, cols, block_size * 2):
                checkerboard[i:i + block_size, j:j + block_size] = True
                checkerboard[i + block_size:i + block_size * 2, j + block_size:j + block_size * 2] = True
        checkerboard = np.stack([checkerboard, checkerboard, checkerboard], axis=-1)

        #apply checkerboard mask
        blended_image = np.where(checkerboard, image1, image2)
        return blended_image


    #map first image to green and second to red
    elif mode == BLEND_COLORMAP:
        if len(image1.shape) == 3:
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        blue_channel = np.zeros_like(image1, dtype=np.uint8)
        blended_image = np.stack((blue_channel, image1, image2), axis=-1)
        return blended_image


    #invalid mode
    else:
        assert False, "Invalid image blending mode."




















####################################################################################################################################
# Tests
####################################################################################################################################

def test_triangle_transformation():
    T = Tri2TriTransform([(3, 4), (1, 2), (4, 2)], [(8, 4), (6, 3), (8.5, 1.75)])
    print(T)
    print(T @ np.array([[2.66], [2.66], [1]]))





def test_simple_bham():
    simple_bresenham(3, 2, 15, 5)

def test_simple_bham_walker():
    #shallow slope
    lo = 3, 2
    hi = 47, 13
    for x, y in simple_bresenham_walker(lo, hi):
        plt.scatter(x, y, color='green')



    #steep slope
    lo = (6, 5)
    hi = (15, 41)
    for x, y in bresenham_walker_steep(*lo, *hi):
        plt.scatter(x, y, color='blue')

    #plot setup
    plt.axis([0, 50, 0, 50])
    plt.title("Bham Line Scan: Shallow vs Steep Slope")
    plt.show()




def BSplineFromGPT():
    """ B-splines interpolation funciton I got from chatgpt when asked for a simple method."""
    def _basis_function(i, k, u, T):
        if k == 0:
            return 1.0 if T[i] <= u < T[i + 1] else 0.0
        else:
            coef1 = (u - T[i]) / (T[i + k] - T[i]) if T[i + k] != T[i] else 0
            coef2 = (T[i + k + 1] - u) / (T[i + k + 1] - T[i + 1]) if T[i + k + 1] != T[i + 1] else 0
            return coef1 * _basis_function(i, k - 1, u, T) + coef2 * _basis_function(i + 1, k - 1, u, T)

    def _evaluate_b_spline(u, k, control_points, T):
        n = len(control_points) - 1
        point = np.zeros(2)
        for i in range(n + 1):
            b = _basis_function(i, k, u, T)
            point += b * control_points[i]
        return point

    def _example_execution(k, control_points, T):
        #parameter space
        u_values = np.linspace(T[k], T[len(control_points)], 1000)

        #evaluate along curve
        curve_points = np.array([_evaluate_b_spline(u, k, control_points, T) for u in u_values])

        # Plot the control points and the B-spline curve
        #plt.plot(control_points[:, 0], control_points[:, 1], 'ro--', label='Control Points')
        plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='B-spline Curve')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('B-spline Curve')
        plt.show()

    #Bezier curve
    _example_execution(k=3, control_points=np.array([[0, 0], [1, 2], [3, 3], [4, 2]]), T=np.array([0,0,0,0,1,1,1,1]))
    #_example_execution(k=2, control_points=np.array([[0, 0], [1, 2], [3, 3], [4, 0]]), T=np.array([0, 0, 0, 1, 2, 3, 3, 3]))
    #_example_execution(2, np.array([[0, 0], [1, 2], [3, 3], [4, 2], [5, 0]]), np.array([0, 0, 0, 2, 4, 5, 5, 5]))
    #_example_execution(2, np.array([[0, 0], [1, 2], [3, 3], [4, 2], [5, 0]]), np.array([0, 0, 0, 2, 4, 5, 5, 5])/5)
    #_example_execution(3, np.array([[0, 0], [1, 2], [3, 3], [4, 2], [5, 0]]), np.array([0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]))


def BSplineGPTFixed():
    def basis_function(i, k, u, T):
        if k == 0:
            return 1.0 if T[i] <= u < T[i + 1] else 0.0
        else:
            coef1 = (u - T[i]) / (T[i + k] - T[i]) if T[i + k] != T[i] else 0
            coef2 = (T[i + k + 1] - u) / (T[i + k + 1] - T[i + 1]) if T[i + k + 1] != T[i + 1] else 0
            return (coef1 * basis_function(i, k - 1, u, T)) + (coef2 * basis_function(i + 1, k - 1, u, T))

    def evaluate_b_spline(u, k, control_points, T):
        n = len(control_points) - 1
        point = np.zeros(2)
        for i in range(n + 1):
            b = basis_function(i, k, u, T)
            print(b)
            point += b * control_points[i]
        return point

    # Degree of the B-spline
    k = 3

    # Control points
    #control_points = np.array([[0, 0], [1, 2], [3, 3], [4, 2], [5, 0]])
    control_points = np.array([[0, 0], [1, 2], [3, 3], [5, 0]])

    # clamped knot vector
    #T = np.array([0, 0, 0, 0, 1, 2, 4, 5, 5, 5, 5])
    T = np.array([0, 0, 0, 0, 5, 5, 5, 5])

    # Ensure parameter values include exact end points
    u_values = np.linspace(T[k], T[len(control_points)], 10000)
    #u_values = np.append(u_values, T[len(control_points)])

    # Evaluate the B-spline curve
    curve_points = np.array([evaluate_b_spline(u, k, control_points, T) for u in u_values])
    print(curve_points)

    # Debugging: Print first and last evaluated points
    print("First point:", curve_points[0])
    print("Last point:", curve_points[-1])
    print("Expected first point:", control_points[0])
    print("Expected last point:", control_points[-1])

    # Plot the control points and the B-spline curve
    plt.plot(control_points[:, 0], control_points[:, 1], 'ro--', label='Control Points')
    plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', label='B-spline Curve')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Clamped B-spline Curve')
    plt.show()



def DeCasteljau(ControlPoints):
    """
    Does DeCasteljau's algorithm based on control points.
    The
    """



def draw_cubic_bezier_curve_simple(control_points):
    """
    This draws a cubic bezier curve by directly interpolating the blending function as opposed to
    using DeCasteljau's algorithm.

    Example control_points: [(1,2), (3,7), (6,6), (10, 1)]
    """
    assert len(control_points) == 4, "Must have 4 control points."

    #basis matrix
    B = np.array([
        [-1, 3, -3, 1],
        [3, -6, 3, 0],
        [-3, 3, 0, 0],
        [1, 0, 0, 0]
    ])

    #control points
    P = np.array(control_points)

    # iterate over and draw points.
    # Uses a simple small step size
    interpolation_domain = np.linspace(0, 1, num=1000)
    for t in interpolation_domain:
        T = np.array([[t**3, t**2, t, 1]])
        pixel = T @ B @ P
        plt.scatter(pixel[0, 0], pixel[0, 1], color='b', s=0.4)

    plt.show()



def draw_adjusted_cubic_bezier_curve_simple(control_points):
    """
    Does same as alogirhtm above, but with symmetric basis matrix
    """
    assert len(control_points) == 4, "Must have 4 control points."

    #basis matrix
    B = np.array([
        [1, -3, 3, -1],
        [1, -1, -1, 1],
        [1, 1, -1, -1],
        [1, 3, 3, 1]
    ]) * 1/8

    #control points
    P = np.array(control_points)

    # iterate over and draw points.
    # Uses a simple small step size
    interpolation_domain = np.linspace(-1, 1, num=1000)
    for t in interpolation_domain:
        T = np.array([[t**3, t**2, t, 1]])
        pixel = T @ B @ P
        plt.scatter(pixel[0, 0], pixel[0, 1], color='b', s=0.4)

    plt.show()



def reverse_cubic_bezier_test():
    from scipy.optimize import fsolve

    # Define the Bézier function for both x and y coordinates
    def bezier_2d(t, x0, x1, x2, x3, y0, y1, y2, y3, x_target, y_target):
        # Parametric equations for x and y
        x_t = ((-x0 + 3 * x1 - 3 * x2 + x3) * t ** 3 +
               (3 * x0 - 6 * x1 + 3 * x2) * t ** 2 +
               (-3 * x0 + 3 * x1) * t +
               x0)

        y_t = ((-y0 + 3 * y1 - 3 * y2 + y3) * t ** 3 +
               (3 * y0 - 6 * y1 + 3 * y2) * t ** 2 +
               (-3 * y0 + 3 * y1) * t +
               y0)

        # Return differences as a flat list
        return np.array([x_t - x_target, y_t - y_target])

    # Define the target point and control points
    x_target, y_target = 0.5, 0.5  # Example target point
    control_points = [(0, 0), (1, 2), (2, 2), (3, 0)]  # Example control points

    # Extract x and y components of control points
    x0, y0 = control_points[0]
    x1, y1 = control_points[1]
    x2, y2 = control_points[2]
    x3, y3 = control_points[3]

    print(bezier_2d(0.1, x0, x1, x2, x3, y0, y1, y2, y3, x_target, y_target))
    print(bezier_2d(0.2, x0, x1, x2, x3, y0, y1, y2, y3, x_target, y_target))
    print(bezier_2d(0.3, x0, x1, x2, x3, y0, y1, y2, y3, x_target, y_target))

    # Solve for t using fsolve
    initial_guess = [0.5]  # Initial guess should be a list or array
    solution_t, infodict, ier, mesg = fsolve(
        bezier_2d, initial_guess, args=(x0, x1, x2, x3, y0, y1, y2, y3, x_target, y_target), full_output=True
    )

    if ier == 1:
        print(f"Parameter t for point ({x_target}, {y_target}) is approximately {solution_t[0]}")
    else:
        print("Solution not found:", mesg)




def test_blending():
    fixed, moving = cv2.imread("Data/TestingDiffDiff/test5/expected.png"), cv2.imread("Data/TestingDiffDiff/test5/unobstructed-aligned.png")
    #fixed, moving = cv2.imread("/Users/aidan/Desktop/ExpoSet/expected/54.png"), cv2.imread("/Users/aidan/Desktop/ExpoSet/observed/54.png")
    import ImageSimilarity
    msd = ImageSimilarity.MeanSquaredDifference(moving, fixed)
    print(msd)
    alpha = BlendImages(fixed, moving, mode=BLEND_COLORMAP)
    cv2.imshow("fixed", fixed)
    cv2.imshow("moving", moving)
    cv2.imshow("blended", alpha)
    cv2.waitKey()




def main():
    """For running tests"""
    #draw_cubic_bezier_curve_simple(control_points=[(1,2), (3,7), (6,6), (10, 1)])

    #draw_adjusted_cubic_bezier_curve_simple(control_points=[(1, 2), (3, 7), (6, 6), (10, 1)])
    #reverse_cubic_bezier_test()
    test_blending()







if __name__ == "__main__":
    main()
