import cv2
import numpy as np

#Global Variables and setup stuff
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)

static_keypoints = np.array([
    [768, 307, 1],
    [1155, 500, 1],
    [638, 1039, 1],
    [172, 659, 1],
    [1127, 1111, 1],
    [545, 458, 1],
])

moving_keypoints = np.array([
    [1081, 124, 1],
    [1405, 411, 1],
    [766, 798, 1],
    [414, 311, 1],
    [1224, 992, 1],
    [827, 213, 1],
])

MOVING = cv2.imread("/Users/aidanlear/Desktop/AffineTeapot/moving.png")
STATIC = cv2.imread("/Users/aidanlear/Desktop/AffineTeapot/static.png")



def draw_keypoints_to_verify():
    global MOVING
    global STATIC
    colors = [
        (255, 0, 0),
        (128, 128, 0),
        (0, 255, 0),
        (0, 128, 128),
        (0, 0, 255),
        (0, 0, 0)
    ]

    for i in range(6):
        MOVING = cv2.circle(MOVING, tuple(moving_keypoints[i][:2]), radius=20, color=colors[i], thickness=-1)
        STATIC = cv2.circle(STATIC, tuple(static_keypoints[i][:2]), radius=20, color=colors[i], thickness=-1)

    cv2.imshow('moving', MOVING)
    cv2.imshow('static', STATIC)
    cv2.waitKey()



def affine_transform(x1, x2, x3, x4, x5, x6):
    return np.array([
        [x1, x2, x3],
        [x4, x5, x6],
        [0, 0, 1]
    ])


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
            dest_r, dest_c = dest[1][0], dest[0][0]

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
    for i in range(STATIC.shape[0]):
        x, y = STATIC[i][0], STATIC[i][1]
        A = np.array([
            [x, y, 1, 0, 0, 0],
            [0, 0, 0, x, y, 1]
        ])
        b = np.array([[MOVING[i][0]], [MOVING[i][1]]])



transformation = np.array([  # homogenous coordinates
    [2.0, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
out = transform_image(STATIC, MOVING, transformation)
cv2.imshow('', out)
cv2.waitKey()




