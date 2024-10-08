import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
import math





def example_transform():
    # Define the transformation matrix with example theta values
    theta = np.array([[0.9, 0.1, 0.05, 0, 0, 0],[0.1, 0.9, 0.05, 0, 0, 0]])  # original example given
    #theta = np.array([[1.00, 0.05, 0.05, 0, 0, 0],[0.05, 0.0, 0.05, 0, 0, 0]])

    # Create a 10x10 grid of points (n, m)
    n_values = np.arange(1, 10)
    m_values = np.arange(1, 10)
    #n_values = np.linspace(1, 10, 4)
    #m_values = np.linspace(1, 10, 4)
    grid_points = np.array([(n, m) for n in n_values for m in m_values])

    # Apply the transformation to each point
    def apply_transformation(theta, points):
        transformed_points = []
        for x, y in points:
            vector = np.array([x**2, x*y, y**2, x, y, 1])
            transformed_point = np.dot(theta, vector)
            transformed_points.append(transformed_point)
        return np.array(transformed_points)

    transformed_points = apply_transformation(theta, grid_points)

    # Plot the original and transformed points
    plt.figure(figsize=(10, 10))
    plt.scatter(grid_points[:, 0], grid_points[:, 1], color='blue', label='Original Points')
    plt.scatter(transformed_points[:, 0], transformed_points[:, 1], color='red', label='Transformed Points')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Original vs Transformed Points')
    plt.grid(True)
    plt.show()

def transform(theta, lo, hi, n):
    n_values = np.linspace(lo, hi, n)
    m_values = np.linspace(lo, hi, n)
    grid_points = np.array([(i, j) for i in n_values for j in m_values])

    # Apply the transformation to each point
    def apply_transformation(_theta, points):
        transformed_points = []
        for x, y in points:
            vector = np.array([x ** 2, x * y, y ** 2, x, y, 1])
            transformed_point = np.dot(_theta, vector)
            transformed_points.append(transformed_point)
        return np.array(transformed_points)

    transformed_points = apply_transformation(theta, grid_points)

    # Plot the original and transformed points
    plt.figure(figsize=(10, 10))
    plt.scatter(grid_points[:, 0], grid_points[:, 1], color='blue', label='Original Points')
    plt.scatter(transformed_points[:, 0], transformed_points[:, 1], color='red', label='Transformed Points')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Original vs Transformed Points')
    plt.grid(True)
    plt.show()

def f(x, y, theta):
    #vector = np.array([x**2, x*y, y**2, x, y, 1])
    vector = np.array([(x ** 2) * (y ** 2), x ** 2, (x ** 2) * y, x * y, x * (y ** 2), y ** 2, x, y, -1])
    transformed_point = np.dot(theta, vector)
    return transformed_point



def f_cubic(x, y, theta):
    pass




def test_ls_setup():
    inputs = np.array([
        [1, 1], [1, 2], [1, 3], [1, 4]
    ])
    inputs = []
    for i in range(1, 5):
        for j in range(1, 5):
            inputs.append((i, j))

    b = np.array(inputs).astype(np.float64)
    shift = 0.1
    b[5] += shift
    b[6] += shift
    b[9] += shift
    b[10] += shift
    #print(b)
    b = b.transpose()
    print(b)

    A = np.zeros((6, 0)).astype(np.float64)
    for x, y in inputs:
        col = np.array([[x**2], [x*y], [y**2], [x], [y], [1]])
        A = np.hstack([A, col])


    A = A.transpose()
    b = b.transpose()
    print(A.shape)
    print(b.shape)


    import scipy


    theta = scipy.linalg.lstsq(A, b)
    print("solution:", theta)


def find_theta(inputs, outputs):
    b = np.array(inputs).astype(np.float64)

    A = np.zeros((9, 0)).astype(np.float64)
    for x, y in outputs:
        #col = np.array([[x ** 2], [x * y], [y ** 2], [x], [y], [1]])
        col = np.array([[x**2 * y**2], [x ** 2], [x**2 * y], [x * y], [x * y**2], [y ** 2], [x], [y], [-1]])
        A = np.hstack([A, col])
    A = A.transpose()

    theta, _, _, _ = scipy.linalg.lstsq(A, b)
    return theta








def more_than_two():
    x = np.array([[1, 2, 3], [4, 5, 16], [17, 8, 9], [7, 8, 19]]) #4x3
    b = np.array([[1, 2, 3], [4, 5, 16]]) + 1
    C, resid, rank, sigma = np.linalg.lstsq(x, b)

    c, m = C[0:3], C[-1]
    x1 = np.array([1, 2, 4])
    print(np.dot(x1, c) + m)
    print(rank)


def verify():

    theta = np.array([[-2.50000000e-02, -2.50000000e-02],
       [-6.66133815e-16, -8.32667268e-16],
       [-2.50000000e-02, -2.50000000e-02],
       [ 1.12500000e+00,  1.25000000e-01],
       [ 1.25000000e-01,  1.12500000e+00],
       [-2.25000000e-01, -2.25000000e-01]]).transpose()
    transform(theta, 1, 50, 10)

    print(f(2, 2, theta))


    b = np.array([[2.075], [2.075]]).astype(np.float64)

    sol = np.linalg.lstsq(theta, b)

    #sol = f(1, 1.2, theta)
    print(sol)


def interpolate_pixel(source, point):
    """
    :param source: source image, i.e. the original image.
    :param point: subpixel accurate point on the source image.

    Uses bilinear interpolation to find the intensity to subpixel accuracy.
    Uses algorithm from Numerical Recipes in C: the Art of Scientific Computing
    """
    #Find neighbors
    x, y = point
    x1, x2, y1, y2 = int(x), int(x + 1), int(y), int(y + 1)

    #intensity at each neighbor
    q11, q12, q21, q22 = source[x1, y1], source[x1, y2], source[x2, y1], source[x2, y2]

    #Bilinear interpolation of intensities. From Numerical Recipes in C: the Art of Scientific Computing
    _x = 1 - x
    left_term = (1 - y) * (q11 * _x + q21 * x)
    right_term = y * (q12 * _x + q22 * x)
    solution = left_term + right_term
    return solution





def how_do_the_kps_move():
    from SelectingKeypoints import strongest_match_each_region

    expected = cv2.imread("Data/TestingDiffDiff/test1/expected.png")
    observed = cv2.imread("Data/TestingDiffDiff/test1/unobstructed-aligned.png")

    kps1, desc1, kps2, desc2 = strongest_match_each_region(observed, expected, 20, 10)


    theta = find_theta(kps1, kps2).transpose()

    # Draws the keypoints on a picture
    drawn_observed = observed.copy()
    drawn_expected = expected.copy()
    for x, y in kps1:
        point = int(x), int(y)
        cv2.circle(drawn_observed, point, radius=6, color=(240, 32, 160), thickness=2)
    for x, y in kps2:
        point = int(x), int(y)
        cv2.circle(drawn_expected, point, radius=6, color=(45, 32, 240), thickness=2)
    cv2.imshow('kps observed', drawn_observed)
    cv2.imshow('kps expected', drawn_expected)
    cv2.waitKey()


    #remap with with nearest neighbor
    out = observed.copy()
    for r in range(expected.shape[0]):
        for c in range(expected.shape[1]):
            candidate = f(r, c, theta)
            try:
                out[r, c] = observed[int(candidate[0]+0.5), int(candidate[1]+0.5)]
            except IndexError:
                pass

    cv2.imshow("DOYOUKNOWHTEWAY.png", out)
    cv2.imwrite("DOYOUKNOWHTEWAY.png", out)
    cv2.waitKey()





##########################################################################################
# PIECEWISE MESH INTERPOLATION WITH SMART THING
##########################################################################################



def test_gftt():
    expected, observed = cv2.imread("Data/TestingDiffDiff/test1/expected.png"), cv2.imread("Data/TestingDiffDiff/test1/unobstructed-aligned.png")
    observed_gray = cv2.cvtColor(observed, cv2.COLOR_BGR2GRAY)
    observed_gray = cv2.GaussianBlur(observed_gray, (5, 5), 0)
    expected_gray = cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY)

    #edge
    #observed_gray, expected_gray = cv2.Canny(observed_gray, 100, 200), cv2.Canny(expected_gray, 100, 200)

    cv2.imshow('observed edge', observed_gray); cv2.imshow('expected edge', expected_gray)

    gftt = cv2.goodFeaturesToTrack(observed_gray, 0, qualityLevel=0.01, minDistance=20,
         mask=None, blockSize=3, useHarrisDetector=False, k=0.1)

    #draw the gftt points
    drawn = observed.copy()
    for p in gftt:
        point = int(p[0][0]), int(p[0][1])
        cv2.circle(drawn, point, radius=6, color=(240, 32, 160), thickness=2)
    cv2.imshow("", drawn)


    #draw gftt on the expected
    gftt = cv2.goodFeaturesToTrack(expected_gray, 0, qualityLevel=0.01, minDistance=20, mask=None, blockSize=3, useHarrisDetector=False, k=0.1)
    drawn2 = expected.copy()
    for p in gftt:
        point = int(p[0][0]), int(p[0][1])
        cv2.circle(drawn2, point, radius=6, color=(240, 32, 160), thickness=2)
    cv2.imshow("expected", drawn2)
    cv2.waitKey()


def test_orb():
    expected, observed = cv2.imread("Data/TestingDiffDiff/test1/expected.png"), cv2.imread("Data/TestingDiffDiff/test1/unobstructed-aligned.png")
    orb = cv2.ORB_create()
    kp = orb.detect(observed)
    kp, des = orb.compute(observed, kp)
    print(len(kp))


    drawn = observed.copy()
    for p in kp:
        point = int(p[0]), int(p[1])
        cv2.circle(drawn, point, radius=6, color=(240, 32, 160), thickness=2)
    cv2.imshow("", drawn)
    cv2.waitKey()







def transformation_from_2_tri():
    # Points of the original triangle
    a, b, c = ((300, 600), (500, 1300), (900, 500))
    A, B, C = ((300, 630), (500, 1650), (900, 500))

    # Constructing the matrices
    P = np.array([
        [a[0], a[1], 1, 0, 0, 0],
        [b[0], b[1], 1, 0, 0, 0],
        [c[0], c[1], 1, 0, 0, 0],
        [0, 0, 0, a[0], a[1], 1],
        [0, 0, 0, b[0], b[1], 1],
        [0, 0, 0, c[0], c[1], 1]
    ])

    Q = np.array([A[0], B[0], C[0], A[1], B[1], C[1]])

    # Solving for the transformation parameters
    sol = np.linalg.solve(P, Q)

    # Extracting the transformation matrix
    transformation_matrix = np.array([
        [sol[0], sol[1], sol[2]],
        [sol[3], sol[4], sol[5]]
    ])

    print("Transformation matrix:")
    print(transformation_matrix)



def try_hough():
    observed = cv2.imread(f"/Users/aidan/Desktop/ExpoSet/observed/1.png")
    expected = cv2.imread(f"/Users/aidan/Desktop/ExpoSet/expected/1.png")
    observed_gray, expected_gray = cv2.cvtColor(observed, cv2.COLOR_BGR2GRAY), cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY)  # grayscale
    observed_gray = cv2.GaussianBlur(observed_gray, (5, 5), 0)  # blur
    observed_edge, expected_edge = cv2.Canny(observed_gray, 100, 200), cv2.Canny(expected_gray, 100, 200)  # Canny edge
    lines = cv2.HoughLines(observed_gray, 1, np.pi / 180, 200)

    # The below for loop runs till r and theta values
    # are in the range of the 2d array
    out = observed.copy()
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        # Stores the value of cos(theta) in a
        a = np.cos(theta)

        # Stores the value of sin(theta) in b
        b = np.sin(theta)

        # x0 stores the value rcos(theta)
        x0 = a * r

        # y0 stores the value rsin(theta)
        y0 = b * r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000 * (-b))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000 * (a))

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000 * (-b))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000 * (a))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
        # (0,0,255) denotes the colour of the line to be
        # drawn. In this case, it is red.
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('', out)
    cv2.waitKey()








#mesh1 = np.array([(300, 600), (500, 1300), (900, 500), (0, 0), (1467, 0), (0, 2200), (1467, 2200)])
#mesh2 = np.array([(300, 630), (500, 1650), (900, 500), (0, 0), (1467, 0), (0, 2200), (1467, 2200)])




#example_transform()

#more_than_two()
#test_ls_setup()
#verify()

#how_do_the_kps_move()
#test_gftt()
#test_orb()


