import numpy as np
import cv2
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

import Dysco

np.set_printoptions(suppress=True)

#Global Variables and setup stuff
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)

#{{768, 1155,  638,  172, 1127,  545},  { 307,  500, 1039,  659, 1111,  458},  {1,   1,   1 ,   1,   1,   1}}
static_keypoints = np.array([
    [768, 307, 1],
    [1155, 500, 1],
    [638, 1039, 1],
    [172, 659, 1],
    [1127, 1111, 1],
    [545, 458, 1],
])

#{{1081, 1405 , 766,  414, 1224, 827}, { 124,  411,  798,  311,  992 , 213}}
moving_keypoints = np.array([
    [1081, 124, 1],
    [1405, 411, 1],
    [766, 798, 1],
    [414, 311, 1],
    [1224, 992, 1],
    [827, 213, 1],
])

MOVING = cv2.imread("Data/AffineTeapot/moving.png")
STATIC = cv2.imread("Data/AffineTeapot/static.png")



def construct_equation():
    A = static_keypoints.transpose()
    b = moving_keypoints.transpose()[0:2, :]
    print(A)
    print(b)



def find_sift_correspondence(static, moving):
    #Ensure the images are the right size
    if moving.shape[0] != static.shape[0] or moving.shape[1] != static.shape[1]:
        interpolation = cv2.INTER_AREA
        if (moving.shape[0] > static.shape[0]) and (moving.shape[1] > static.shape[1]):  # image is bigger
            interpolation = cv2.INTER_CUBIC
        moving = cv2.resize(moving, (static.shape[1], static.shape[0]), interpolation=interpolation)

    #calculate and match the keypoints
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(moving, None)
    kp2, des2 = sift.detectAndCompute(static, None)
    bf = cv2.BFMatcher(cv2.DIST_L2, crossCheck=True)
    matches = bf.match(des1, des2)  # first is observed, second is expected
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    print(src_pts.shape)
    print(src_pts[0:10, :, :])
    print('------')
    print(dst_pts[0:10, :, :])
    print(dst_pts.shape)


def transform_image(static, moving, M):
    #setup transform
    output = np.zeros(static.shape, dtype=np.uint8)
    rows, cols = static.shape[0], static.shape[1]

    #iterate over each pixel
    for r in range(rows):
        for c in range(cols):
            #pixel in homogenous coordinates
            src = np.array([[c], [r], [1]], dtype=np.float32)

            #find where the pixel is
            dest = M @ src
            w = dest[2][0]
            dest_r, dest_c = dest[1][0]/w, dest[0][0]/w

            #make sure the pixel actually exists in the image
            if dest_r < 0 or dest_r >= moving.shape[0] or dest_c < 0 or dest_c >= moving.shape[1]:
                continue

            #nearest neighbor interpolation
            row, col = int(dest_r), int(dest_c)
            output[r, c] = moving[row, col]
    #return result
    return output



def optimization_part():
    #np.linalg.lstsq(a, b) solves for x in  a@x = b

    # construct transformation for n points
    A = np.zeros((0, 6))
    b = np.zeros((0, 1))

    for i in range(static_keypoints.shape[0]):
        x, y = static_keypoints[i][0], static_keypoints[i][1]
        A_ = np.array([
            [x, y, 1, 0, 0, 0],
            [0, 0, 0, x, y, 1]
        ])
        A = np.concatenate([A, A_], axis=0)

        b_ = np.array([[moving_keypoints[i][0]], [moving_keypoints[i][1]]])
        b = np.concatenate([b, b_], axis=0)

    print(A)
    print(b)

    #now solve as Ax = b
    x = np.linalg.lstsq(A, b, rcond=None)[0] #NOTE: it returns tuple with some other info; 0th index is solution.
    print(x)
    print(x.astype(np.int32))



def test_transform():
    transformation = np.array([  # homogenous coordinates
        [2.0, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    transformation = np.array([  # homogenous coordinates
        [1, 0, 415],
        [0, 1, -369],
        [0, 0, 1]
    ])

    transformation = np.array([[9.68074734e-01, -2.55956472e-01, 4.15899577e+02],
                               [2.57028375e-01, 9.64890426e-01, -3.69028931e+02],
                               [0, 0, 1]])
    out = transform_image(STATIC, MOVING, transformation)
    cv2.imshow('', out)
    cv2.waitKey()



def normal_equations_solution():
    # construct transformation for n points
    A = np.zeros((0, 6))
    b = np.zeros((0, 1))
    for i in range(static_keypoints.shape[0]):
        x, y = static_keypoints[i][0], static_keypoints[i][1]
        A_ = np.array([
            [x, y, 1, 0, 0, 0],
            [0, 0, 0, x, y, 1]
        ])
        A = np.concatenate([A, A_], axis=0)
        b_ = np.array([[moving_keypoints[i][0]], [moving_keypoints[i][1]]])
        b = np.concatenate([b, b_], axis=0)

    #normal equation -> x = (A^t A)^-1 A^t b
    x = np.linalg.inv(A.transpose() @ A) @ A.transpose() @ b
    print("LAST THING HERE")
    print(x)




##########################################################################################
# PERSPECTIVE/HOMOGRAPHIC TRANSFORMATION
##########################################################################################
def get_perspective_correspondence():
    """
    This routine is ONLY for testing.
    It has matching sift correspondence for the the images in Data/PerspectiveTeapot
    The list of matches is returned as a dictionary to prevent confusion.
    Match numbers used: 4, 9, 13, 1, 11, 5, 2
    """
    static = np.array([
        (181.36, 662.34),
        (1148.00, 496.26),
        (552.11, 1097.47),
        (327.68, 877.06),
        (778.42, 333.25),
        (409.19, 646.56),
        (546.71, 462.59)
    ])

    moving = np.array([
        (425.59,  355.67),
        (1364.69, 577.93),
        (729.00, 850.02),
        (547.89, 596.14),
        (1008.66,  256.67),
        (624.96, 421.25),
        (762.34, 302.59)
    ])

    return {'static': static, 'moving': moving}



def get_perspective_correspondence2():
    static = np.array([
        (181.36,  662.34),
        (337.62,  631.90),
        (482.36,  729.21),
        (500.83,  475.08),
        (814.82,  413.44),
        (639.00, 1039.00),
        (1129.00, 1110.00)
    ])

    moving = np.array([
        (236.54,  717.25),
        (458.94,  682.08),
        (622.33,  824.49),
        (673.13,  487.77),
        (980.41,  456.22),
        (756.00, 1168.00),
        (1149, 1173)
    ])
    return {'static': static, 'moving': moving}


def draw_correspondence():
    """
    Testing routine that draws the correspondence just to make sure it is correct
    """
    static = cv2.imread('Data/PerspectiveTeapot/static.png')
    moving = cv2.imread('Data/PerspectiveTeapot/moving2.png')

    static_keypoints = get_perspective_correspondence2()['static']
    moving_keypoints = get_perspective_correspondence2()['moving']


    for i in range(static_keypoints.shape[0]):
        point1, point2 = static_keypoints[i], moving_keypoints[i]

        # round points to nearest pixel, make sure they are integers
        point1 = int(point1[0] + 0.5), int(point1[1] + 0.5)
        point2 = int(point2[0] + 0.5), int(point2[1] + 0.5)

        cv2.circle(static, point1, 4, (0, 0, 255), -1)
        cv2.circle(moving, point2, 4, (0, 0, 255), -1)
        cv2.putText(static, str(i + 1), (point1[0] + 10, point1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(moving, str(i + 1), (point2[0] + 10, point2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('static', static)
    cv2.imshow('moving', moving)
    cv2.waitKey()



def construct_A(points):
    A = np.zeros((0, 9))
    for i in range(points.shape[0]):
        x, y = points[i]
        section = np.array([
            [x, y, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, x, y, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, x, y, 1]
        ])
        A = np.concatenate([A, section], axis=0)
    return A

def construct_b(points):
    b = np.zeros((0, 1))
    for i in range(points.shape[0]):
        x, y = points[i]
        b_ = np.array([[x], [y], [1]])
        b = np.concatenate([b, b_], axis=0)
    return b

def ls_solution():
    #static_keypoints = get_perspective_correspondence()['static']
    #moving_keypoints = get_perspective_correspondence()['moving']
    static_keypoints = get_perspective_correspondence2()['static']
    moving_keypoints = get_perspective_correspondence2()['moving']

    A = construct_A(static_keypoints)
    b = construct_b(moving_keypoints)

    x = np.linalg.lstsq(A, b, rcond=None)[0]
    #print(x)
    M = x.reshape(3, 3)
    print(M)

    #M[2] = np.array([2, 0, 1], dtype=np.float64)


    #test the transformation
    static = cv2.imread('Data/PerspectiveTeapot/static.png')
    moving = cv2.imread('Data/PerspectiveTeapot/moving2.png')
    out = transform_image(static, moving, M)
    cv2.imshow('ls solution', out)



def penn_state_construct_A(src, dst):
    A = np.zeros((0, 9))
    for i in range(src.shape[0]):
        x, y = src[i]
        x_, y_ = dst[i]
        section = np.array([
            [x, y, 1, 0, 0, 0, -1*x*x_, -1*y*x_, -1*x_],
            [0, 0, 0, x, y, 1, -1*x*y_, -1*y*y_, -1*y_]
        ])
        A = np.concatenate([A, section], axis=0)
    return A

def penn_construct_A_alt(src, dst):
    """
    same as above, BUT it omits solving for the bottom left part of the transformation matrix.
    The solution is constructed so that will always be 1.
    """
    A = np.zeros((0, 8))
    for i in range(src.shape[0]):
        x, y = src[i]
        x_, y_ = dst[i]
        section = np.array([
            [x, y, 1, 0, 0, 0, -1 * x * x_, -1 * y * x_],
            [0, 0, 0, x, y, 1, -1 * x * y_, -1 * y * y_]
        ])
        A = np.concatenate([A, section], axis=0)
    return A


def penn_ls_solution():
    """
    This solution works
    """
    static_keypoints = get_perspective_correspondence2()['static']
    moving_keypoints = get_perspective_correspondence2()['moving']

    A = penn_construct_A_alt(static_keypoints, moving_keypoints)
    b = moving_keypoints.reshape((-1, 1)) # flattens into column vector

    x = np.linalg.lstsq(A, b, rcond=None)[0]
    x = np.concatenate([x, [[1]]], axis=0)
    M = x.reshape(3, 3)
    print(M)


    # test the transformation
    static = cv2.imread('Data/PerspectiveTeapot/static.png')
    moving = cv2.imread('Data/PerspectiveTeapot/moving2.png')
    out = transform_image(static, moving, M)
    cv2.imshow('penn ls solution', out)
    cv2.waitKey()



#draw_correspondence()
ls_solution()
penn_ls_solution()





