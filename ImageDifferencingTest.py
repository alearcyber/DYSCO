"""
---Test Description---
The purpose of this script is to run a test for doing a simple difference of the expected image and the picture
of the display.
"""



import cv2
import numpy as np


def downsample(img, rows, cols):
    """
    Down-sample an image into rows x cols size. Maintains original image dimensions.
    """
    #get original image shape
    original_shape = img.shape

    # do the downsampling
    downsampled = cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA)

    #go back up to the original shape
    return cv2.resize(downsampled, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)


def chi_squared(A, B):
    chi = 0.5 * np.sum([((A - B) ** 2) / (A + B) for (A, B) in zip(A, B)])
    return chi

def experiment():
    """
    This function is the main routine for running the experiment itself.
    """
    #number of rows and columns working with for this experiment
    rows = 5
    cols = 10


    #Read the test images into memory.
    #Each test is a dictionary inside the list, test_images.
    dir = "Data/TestSet1"
    test_images = []
    for i in range(1, 9):
        test = {
            'observed': cv2.imread(f"{dir}/test{i}/cam-low-exposure.png"),
            'expected': cv2.imread(f"{dir}/test{i}/expected-expanded.png"),
            'ground_truth': cv2.imread(f"{dir}/test{i}/obstructionmask.png")
        }
        test_images.append(test)



    #Preprocess the observed and expected images...
    #No preprocessing, just downsample and chi squared distance
    i = 0
    for test in test_images:
        i += 1
        #original shape
        original_shape = test['observed'].shape
        print('original shape:', original_shape)
        print('original shape:', original_shape)

        #downsample observed and expected for comparison
        observed_downsampled = cv2.resize(test['observed'], (cols, rows), interpolation=cv2.INTER_AREA)
        expected_downsampled = cv2.resize(test['expected'], (cols, rows), interpolation=cv2.INTER_AREA)
        print('expected downsampled dtype:', expected_downsampled.dtype)

        cv2.imshow('observed', cv2.resize(observed_downsampled, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST_EXACT))
        cv2.imshow('expected', cv2.resize(expected_downsampled, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST_EXACT))

        #take difference between the two images
        difference = np.absolute(observed_downsampled.astype(int) - expected_downsampled.astype(int))
        print('diff dtype:', difference.dtype)

        #this next set of lines is for visualizing the difference
        bloom_difference = cv2.resize(difference, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)

        cv2.imshow(f'Difference {i}', bloom_difference.astype('uint8'))
        cv2.waitKey()
        cv2.destroyWindow("observed")
        cv2.destroyWindow("expected")
        cv2.destroyWindow("expected")
        cv2.destroyWindow(f'Difference {i}')









def test_downsampling():
    img = cv2.imread("/Users/aidan/PycharmProjects/DYSCO/Data/TestSet1/test1/cam-low-exposure.png")


    area = downsample(img, 5, 10)
    cv2.imshow("downsampled", area)
    cv2.waitKey()


if __name__ == "__main__":
    experiment()
