import numpy as np
import cv2


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
    print(x)





#optimization_part()
#test_transform()
#normal_equations_solution()
construct_equation()



