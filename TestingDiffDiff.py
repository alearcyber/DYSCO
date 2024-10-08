import cv2
import numpy as np


#observed, expected, unobstructed = read_images(1) #read in images
def read_images(n):
    folder = f"Data/TestingDiffDiff/test{n}"
    o = cv2.imread(f"{folder}/observed-aligned.png")
    e = cv2.imread(f"{folder}/expected.png")
    u = cv2.imread(f"{folder}/unobstructed-aligned.png")
    return o, e, u



def diffdiff(observed, expected, unobstructed):
    """Calculates the difference of the difference image."""

    unobstructed, observed = cv2.GaussianBlur(unobstructed, (5, 5), 0), cv2.GaussianBlur(observed, (5, 5), 0) #blur
    observed, expected, unobstructed = observed.astype(np.float64), expected.astype(np.float64),  unobstructed.astype(np.float64) #convert to floats


    #calulate difference (aka delta) for the unobstructed
    """
    gt_delta = np.linalg.norm(unobstructed - expected, axis=2)
    gt_delta = gt_delta * (255.0/441.673)
    gt_delta = gt_delta.astype(np.uint8)


    #difference for the obstructed
    obstructed_delta = np.linalg.norm(observed - expected, axis=2)
    obstructed_delta = obstructed_delta * (255.0/441.673)
    obstructed_delta = obstructed_delta.astype(np.uint8)
    """

    #now diffdiff
    d1, d2 = np.linalg.norm(observed - expected, axis=2), np.linalg.norm(unobstructed - expected, axis=2)
    print(d1.shape)
    print(d2.shape)
    dd = (abs(d1 - d2) * (255.0/441.673)).astype(np.uint8)
    return dd



def test1():


    #observed, expected, unobstructed = read_images(1)
    image = diffdiff(*read_images(1))
    cv2.imshow("", image)
    cv2.waitKey()

    #thresholdimage
    image = cv2.GaussianBlur(image, (5, 5), 0)
    ret, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("", th)
    cv2.waitKey()

    #morphological operation. Closing, really just dilation followed by erosion.
    kernel = np.ones((20, 20), np.uint8)
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("", closed)
    cv2.waitKey()


test1()











