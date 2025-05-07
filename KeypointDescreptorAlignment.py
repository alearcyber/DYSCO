import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

observed = cv2.imread("Data/TestSet1ProperlyFormatted/test2/observed.png")
expected = cv2.imread("Data/TestSet1ProperlyFormatted/test2/expected.png")



def distance_constraint(kp1, kp2, threshold):
    """A constraint for the constrained brute-force matching routine."""
    return np.linalg.norm(np.array(kp1.pt) - np.array(kp2.pt)) < threshold



def constrained_bf_matcher(descriptors1, keypoints1, descriptors2, keypoints2, constraint, threshold, norm_type=cv2.NORM_L2, lowe_ratio=0.70):
    """
    A brute force matcher with constraints.

    :param descriptors1:
    :param keypoints1:
    :param descriptors2:
    :param keypoints2:
    :param constraint:
    :param threshold:
    :param norm_type:
    :return:
    """

    matches = []
    for idx1, desc1 in enumerate(descriptors1):
        best_match = None
        second_best_match = None
        best_distance = float('inf')
        second_best_distance = float('inf')
        kp1 = keypoints1[idx1]

        for idx2, desc2 in enumerate(descriptors2):
            kp2 = keypoints2[idx2]
            if constraint(kp1, kp2, threshold):  # Check the constraint
                distance = cv2.norm(desc1, desc2, norm_type)
                if distance < best_distance:
                    second_best_distance = best_distance
                    second_best_match = best_match
                    best_distance = distance
                    best_match = cv2.DMatch(_queryIdx=idx1, _trainIdx=idx2, _distance=distance)
                elif distance < second_best_distance:
                    second_best_distance = distance
                    second_best_match = cv2.DMatch(_queryIdx=idx1, _trainIdx=idx2, _distance=distance)

        # Apply Lowe's ratio test
        if best_match and second_best_match and best_distance < lowe_ratio * second_best_distance:
            matches.append(best_match)

    #return matches
    return matches


def draw_point(image, point, label):
    #round point to the nearest pixel if necessary
    if type(point[0]) is not int:
        point = int(point[0] + 0.5), int(point[1] + 0.5)

    #draw circle
    cv2.circle(image, point, 2, (0, 0, 255), -1)  # Red point

    # Add the label next to the point
    # Parameters are: image, text, bottom-left corner of the text, font, font scale, color, thickness
    cv2.putText(image, label, (point[0] + 10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



def sift_test(n):
    """
    This function will take an unobstruced observed and expected images, grab the SIFT
    keypoints of each image, the visualize the best few matches.

    The SIFT keypoints will calculated in each image.

    Weak matches will be pruned with Lowe's ratio test first.
    Then additional matches will be pruned by taking the l2 distance between each of the keypoints.
    Finally, the closest n-matches will be visualized.
    """
    img1, img2 = cv2.cvtColor(observed, cv2.COLOR_BGR2GRAY), cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # x,y locations
    loc1, loc2 = [p.pt for p in kp1], [p.pt for p in kp2] #x,y locations



    #Match with my routine
    #1 is query, 2 is train
    matches = constrained_bf_matcher(des1, kp1, des2, kp2, distance_constraint, threshold=10, norm_type=cv2.NORM_L2,
                           lowe_ratio=0.70)
    print(len(matches))



    #draw the matches
    out1, out2 = np.copy(observed), np.copy(expected)

    #grab n random indexes
    random_indexes = []
    for i in range(n):
        random_indexes.append(random.randint(0, len(matches)-1))

    #iterate over random indexes and draw them
    for i in random_indexes:
        match = matches[i]

        point1, point2 = kp1[match.queryIdx].pt, kp2[match.trainIdx].pt


        draw_point(out1, point1, str(i)), draw_point(out2, point2, str(i))


    #show images with matches drawn on it
    cv2.imshow('one', out1)
    cv2.imshow('two', out2)
    cv2.waitKey()








sift_test(20)

