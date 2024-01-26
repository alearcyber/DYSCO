"""
each *_features() function returns a list of each difference matrix.
"""
import cv2
import numpy as np
from Texture import texture_features


#global variables
rows = 10
cols = 20
dir = "Data/TestSet1"
test_images = [] #list of tests. Each test is a dict.


#Read the test images into memory.
#Each test is a dictionary inside the list, test_images.
for k in range(1, 9):
    test_images.append({
        'observed': cv2.imread(f"{dir}/test{k}/cam-low-exposure.png"),
        'expected': cv2.imread(f"{dir}/test{k}/expected-expanded.png"),
        'ground_truth': cv2.imread(f"{dir}/test{k}/obstructionmask.png")
    })



def color_features():
    differences = [] #list of difference images, organized by index.
    for i in range(len(test_images)):
        test = test_images[i]
        # downsample observed and expected for comparison
        observed_downsampled = cv2.resize(test['observed'], (cols, rows), interpolation=cv2.INTER_AREA)
        expected_downsampled = cv2.resize(test['expected'], (cols, rows), interpolation=cv2.INTER_AREA)

        # take difference between the two images
        difference = np.absolute(observed_downsampled.astype(int) - expected_downsampled.astype(int))
        differences.append(difference)
    return differences



def texture_features():
    # TODO - finish this function
    pass


def edge_features():
    # TODO - finish this function
    pass



