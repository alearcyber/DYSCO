"""
The purpose of this script is to align the pictures of the dashboards in Data/NewDashboardPictures with the
actual images being captured in NewDashboards.


Explanation on the returned value from flann.knnMatch:
    The matches variable is a list of tuples, and each tuple contains two DMatch objects.
    These tuples are the result of the knnMatch method. For each keypoint descriptor in the first image, the method
    finds the two nearest keypoints in the second image (hence two DMatch objects in each tuple).
    The reason for obtaining the two nearest matches is often to perform a ratio test (as proposed by David Lowe in
    his SIFT paper) to filter out ambiguous matches. By comparing the distance of the closest match to that of the
    second-closest match, one can discard matches where the ratio is too high, which indicates that the match is
    not reliable.
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt



def findHomography(src, dest):
    """
    Computes the homography matrix between the src and dest images.
    Uses SIFT keypoints, Lowe's ratio test with a threshold of 75%, FLANN keypoints matching, and a RANSAC method
    for computing the homography matrix
    """
    # finding SIFT keypoints
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(src, None)
    keypoints2, descriptors2 = sift.detectAndCompute(dest, None)

    # match keypoints
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe's ratio test to remove bad matches
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Computer homography matrix from keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M




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


#print("len matches:", len(matches))
#print("Type of elements in matches:", type(matches[0][0]))


def main():

    pic = cv2.imread("Data/NewDashboardPictures/darktable_exported/20240221_100532.png", cv2.IMREAD_GRAYSCALE)
    pic_color = cv2.imread("Data/NewDashboardPictures/darktable_exported/20240221_100532.png")
    scrnsht = cv2.imread("Data/NewDashboards/Boeing 737 EICAS.png", cv2.IMREAD_GRAYSCALE)


    out = align_images(pic, scrnsht)
    cv2.imshow("aligned", out)
    cv2.waitKey()
    """
    M = findHomography(pic, scrnsht)

    warped = cv2.warpPerspective(pic, M, (pic.shape[1], pic.shape[0]))
    cv2.imshow("warped", warped)
    cv2.waitKey()
    """
    




if __name__ == "__main__":
    main()


