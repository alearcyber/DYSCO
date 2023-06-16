"""
--Steps For Creating an Image Pair--
Detect and match features between image 1 and 2.
Estimate the geometric transform between that maps the two images.
Compute the transform.


--Types of Affine transformations--
Identity
Scaling
Rotation
Translation
Shear (vertical and horizontal)

--Note on different types of transformations--
Affine transformations is a subset of Geometric transformations. Affine transformation reserves lines and parallelism.






"""

import cv2
import numpy as np


img = [  # images
        cv2.imread("/Users/aidanlear/Desktop/Research2023/TESTIMAGES/fail1.jpg"),
        cv2.imread("/Users/aidanlear/Desktop/Research2023/TESTIMAGES/fail3.jpg"),
        cv2.imread("/Users/aidanlear/Desktop/Research2023/TESTIMAGES/fail4.jpg"),
        cv2.imread("/Users/aidanlear/Desktop/Research2023/TESTIMAGES/fail6.jpg"),
    ]


def test_match_features():
    i = [  # images
        cv2.imread("/Users/aidanlear/Desktop/Research2023/TESTIMAGES/fail1.jpg"),
        cv2.imread("/Users/aidanlear/Desktop/Research2023/TESTIMAGES/fail3.jpg"),
        cv2.imread("/Users/aidanlear/Desktop/Research2023/TESTIMAGES/fail4.jpg"),
        cv2.imread("/Users/aidanlear/Desktop/Research2023/TESTIMAGES/fail6.jpg"),
    ]


    """
    print(type(i[0]))
    dim = (i[0].shape[1], i[0].shape[0])
    i[1] = cv2.resize(i[1], dim, interpolation=cv2.INTER_AREA)
    #cast image to array of floats
    out = np.add(i[0], i[1]) // 2
    cv2.imshow("combined", out)
    cv2.waitKey()
    """


    #SIFT features
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(i[0], None)
    kp2, des2 = sift.detectAndCompute(i[1], None)
    #img = cv2.drawKeypoints(i[0], kp, i[0], flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow('keypoints', img)
    #cv2.waitKey()

    bf = cv2.BFMatcher(cv2.DIST_L2, crossCheck=True)
    matches = bf.match(des1, des2)  # first is query, second is train
    matches = sorted(matches, key=lambda x: x.distance)


    #organize the matches


    print(type(matches[0]))
    #match: cv2.DMatch = matches[0]
    l1, l2 = [], []
    for match in matches[:50]:
        one = kp1[match.queryIdx].pt
        two = kp2[match.trainIdx].pt
        print(one, ':', two)
    #d1: cv2.KeyPoint = kp1[match.queryIdx]
    #d2 = kp2[match.trainIdx]
    """
    print(d1.pt)
    out = cv2.drawMatches(i[0], kp1, i[1], kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('matches', out)
    cv2.waitKey()
    """




def test_matrix_algebra():
    a = np.array([[2, 2],
                  [2, 2]])

    b = np.array([[4,4], [4,4]])

    out = np.add(a, b)
    out2 = np.add(a, b) / 2
    print(out)
    print(out2)


#Enhanced Correlation Coefficient Maximization
def ecc():
    # Read the images to be aligned
    im1 = img[0]
    im2 = img[1]

    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # Show final results
    cv2.imshow("Image 1", im1)
    cv2.imshow("Image 2", im2)
    cv2.imshow("Aligned Image 2", im2_aligned)
    cv2.waitKey(0)

if __name__ == '__main__':
    dim = (img[0].shape[1], img[0].shape[0])


    img[1] = cv2.resize(img[1], dim, interpolation=cv2.INTER_AREA)
    out = np.add(img[0], img[1]) // 2
    out = np.abs(img[0] - img[1])
    cv2.imshow('DIFFERENCE', out)
    cv2.imshow('one', img[0])
    cv2.imshow('two', img[1])
    cv2.waitKey()

    #test_match_features()
    #test_matrix_algebra()
    #ecc()

