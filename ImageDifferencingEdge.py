"""
Nearly identical to ImageDifferencingTest.py and ImageDifferencingTexture.py, except this will
use edge features.

4 different edge detectors:
canny, sobel, laplacian, and fourier
    - sobel will be a 3 feature vector of a vertical, horizontal, and combined filters
"""

import numpy as np
import cv2




def bilateral_blur_example():
    test_image = cv2.imread("Data/TestSet1/test1/cam-low-exposure.png")
    for i in range(3, 10, 2):
        blurred_image = cv2.bilateralFilter(test_image, i, 75, 75)
        cv2.imshow(f'bilateral blur, d={i}', blurred_image)
        cv2.waitKey()
        cv2.destroyAllWindows()


def bilateral_canny_example():
    test_image = cv2.imread("Data/TestSet1/test1/cam-low-exposure.png")


    cv2.imshow(f'canny, no blur', cv2.Canny(test_image, 100, 200))

    for i in range(3, 10, 2):
        blurred_image = cv2.bilateralFilter(test_image, i, 75, 75)
        edges = cv2.Canny(blurred_image, 100, 200)
        cv2.imshow(f'bilateral blur then canny, d={i}', edges)
        cv2.waitKey()
        cv2.destroyAllWindows()





if __name__ == "__main__":
    bilateral_canny_example()





