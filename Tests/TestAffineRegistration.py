import sys
import os
import cv2
import numpy as np
import json

import Support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add the parent directory (root) to the path
import AffineRegistration
import Color


MOVING = cv2.imread("../Data/Mar14Tests/0015.jpg")
STATIC = cv2.imread("../Data/Displays/box-black.png")




def test_find_transform():

    #clockwise starting at top left
    source_points = np.array([[39,43], [15,4106], [2279,4074], [2287,100]], dtype=np.float64)
    dest_points = np.array([[0,0], [0, 1600], [900, 1600], [900, 0]], dtype=np.float64)

    #swap the columns
    source_points, dest_points = source_points[:, [1, 0]], dest_points[:, [1, 0]]


    #temp = ar.DrawPoints(MOVING, source_points)
    #cv2.imshow('', temp)
    #cv2.waitKey()

    H = AffineRegistration.FindTransform(source_points, dest_points) #equivalent to H = cv2.findHomography(source_points, dest_points)[0]
    H = cv2.findHomography(source_points, dest_points)[0]

    d = H @ np.array([[15], [4106], [1]])
    x, y = d[0][0]/d[2][0], d[1][0]/d[2][0]
    print(x, y)

    #check if the warping works
    out = cv2.warpPerspective(MOVING, H, (STATIC.shape[1], STATIC.shape[0]))
    cv2.imshow('', out)
    cv2.waitKey()

    #composite
    composite = AffineRegistration.Composite(STATIC, out, grid=(5,8))
    cv2.imshow('', composite)
    cv2.waitKey()
    cv2.destroyAllWindows()



def test_overfit_checkerboard():
    static = cv2.imread("../Data/Displays/Checkerboard-7x9.png")
    #moving = cv2.imread("/Users/aidanlear/PycharmProjects/DYSCO-2025/Data/Mar14Tests/Cropped/0044.jpg")
    moving = cv2.imread("../Data/Mar14Tests/0044.jpg")

    Support.ShowImage(moving)


    static_pts = np.array([
        [328.565217391, 226.52173913],
        [555.5, 680.488888889],
        [1009.46875, 226.46875],
        [1916.54545455, 226.454545455],
        [1916.54545455, 906.454545455],
        [1916.375, 1360.5],
        [782.4375, 1360.625],
        [102.4, 1360.46666667]
    ])
    moving_pts = np.array([
        [776.5, 665.5],
        [1149.75, 1386.25],
        [1855.5, 664.25],
        [3291.27272727, 653.090909091],
        [3285.5, 1727.0],
        [3286.375, 2433.5],
        [1516.875, 2442.875],
        [447.0, 2456.5]
    ])



    #draw points on images and show as a verification step before continuing with registration.
    static_with_pts = np.copy(static)
    moving_with_pts = np.copy(moving)
    colors = Color.GenerateColors(len(static_with_pts))
    for p1, p2, color in zip(moving_pts, static_pts, colors):
        #cv2.circle(image, centerOfCircle, radius, color, thickness)
        #moving_with_pts = cv2.circle(moving_with_pts, p1.astype(int), 10, color, -1)
        cv2.circle(static_with_pts, p2.astype(int), 10, color, -1)



    Support.ShowImage(static_with_pts)




def test_current_kps():
    """
    Good: 39, 40, 41
    """
    image_to_test = 41
    data_file = "../dataset.json"

    with open(data_file, 'r') as file:
        # Load the JSON data into a Python object
        data = json.load(file)


    #parse the json
    file, original = None, None
    source_keypoints = None
    destination_keypoints = None
    image_list = data["images"]
    for image_data in image_list:
        if image_data['id'] == image_to_test:
            file, original = image_data['file'], image_data['original']
            source_keypoints = np.array(image_data['source_keypoints'])
            destination_keypoints = np.array(image_data['destination_keypoints'])
            break

    assert len(source_keypoints) == len(destination_keypoints), "Mismatched number of keypoints."



    #read in images
    dest = cv2.imread("../Data/Displays/" + original) #aka static
    src = cv2.imread("../Data/Mar14Tests/Cropped/" + file) #aka moving

    # draw points on images and show as a verification step before continuing with registration.
    static_with_pts = np.copy(dest)
    moving_with_pts = np.copy(src)

    colors = Color.GenerateColors(len(source_keypoints))
    for p1, p2, color in zip(source_keypoints, destination_keypoints, colors):
        # cv2.circle(image, centerOfCircle, radius, color, thickness)
        cv2.circle(moving_with_pts, p1.astype(int), 10, color, -1)
        cv2.circle(static_with_pts, p2.astype(int), 10, color, -1)

    Support.ShowImages((static_with_pts, moving_with_pts), ('static', 'moving'))



def main():
    test_current_kps()

if __name__ == "__main__":
    main()
