"""
Registering images.

Making sure that images align up with each other
"""

import cv2
import numpy as np

#extend image bounds with blank pixels so they match.
def _extend_image(a, b):
    #check if the image is grayscale
    if len(a.shape) == 2:
        return _extend_image_grayscale(a, b)


    #extend height
    dx = a.shape[0] - b.shape[0]
    if dx > 0: #a is bigger, extend b
        new_pixels = np.zeros((abs(dx), b.shape[1], 3), dtype='uint8')
        b = np.concatenate((b, new_pixels), axis=0)
    elif dx < 0: #b is bigger, extend a
        new_pixels = np.zeros((abs(dx), a.shape[1], 3), dtype='uint8')
        a = np.concatenate((a, new_pixels), axis=0)

    #extend width
    dy = a.shape[1] - b.shape[1]
    if dy > 0:  # a is bigger, extend b
        new_pixels = np.zeros((b.shape[0], abs(dy), 3), dtype='uint8')
        b = np.concatenate((b, new_pixels), axis=1)
    elif dy < 0:  # b is bigger, extend a
        new_pixels = np.zeros((a.shape[0], abs(dy), 3), dtype='uint8')
        a = np.concatenate((a, new_pixels), axis=1)

    #return output
    return a, b



#Aligns query image with the train image based on their features and estimating the affine transformation.
#Returns a copy of the query image and the train image. This is because both images may be given padding so they
# have the same dimensions.
#Note: Does not currently consider the accuracy/strength of the matches.
def align_images(query, train):
    # extend bounds of images
    query, train = _extend_image(query, train)

    # keypoints and matching
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(query, None)
    kp2, des2 = sift.detectAndCompute(train, None)
    bf = cv2.BFMatcher(cv2.DIST_L2, crossCheck=True)
    matches = bf.match(des1, des2)  # first is query, second is train
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find the transformation between points, standard RANSAC
    transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    #Warp the image
    query = cv2.warpPerspective(query, transformation_matrix, (query.shape[1], query.shape[0]))

    # return the aligned images
    return query, train