"""
Registering images.

Making sure that images align up with each other
"""

import cv2
import numpy as np




#aligns observed with expected.
#returns a copy of observed aligned with expected.
#Dimension of result are equal to that of expected.
#Note: Does not currently consider the accuracy/strength of the matches.
def align_images(observed, expected):
    #resize observed to match expected
    #inter_area to make smaller
    #bicubic to make bigger
    interpolation = cv2.INTER_AREA
    if (observed.shape[0] > expected.shape[0]) and (observed.shape[1] > expected.shape[1]): #image is bigger
        interpolation = cv2.INTER_CUBIC
    observed = cv2.resize(observed, (expected.shape[1], expected.shape[0]), interpolation=interpolation)


    # keypoints and matching
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(observed, None)
    kp2, des2 = sift.detectAndCompute(expected, None)
    bf = cv2.BFMatcher(cv2.DIST_L2, crossCheck=True)
    matches = bf.match(des1, des2)  # first is observed, second is expected
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find the transformation between points, standard RANSAC
    transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp the image
    observed = cv2.warpPerspective(observed, transformation_matrix, (observed.shape[1], observed.shape[0]))

    #Done
    return observed



#testing the alignment of images
def test1():

    expected = cv2.imread('images/rawdisplay.png', cv2.IMREAD_GRAYSCALE) #screenshot
    observed = cv2.imread('images/picture1.png', cv2.IMREAD_GRAYSCALE) #high quality picture of unobstructed display

    cv2.imshow('expected', expected)
    cv2.imshow('original observed', observed)

    observed = align_images(observed, expected)
    cv2.imshow('scaled observed', observed)
    print('expected shape:', expected.shape)
    print('observed shape:', observed.shape)

    composite = np.stack((observed, expected), axis=-1)
    cv2.imshow('composite', composite.mean(axis=2).astype(np.uint8))
    cv2.waitKey()



if __name__ == '__main__':
    """entry point"""
    test1()

