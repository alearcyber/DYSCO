"""
---Test Description---
The purpose of this script is to run a test for doing a simple difference of the expected image and the picture
of the display.
"""



import cv2



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



def experiment():
    """
    This function is the main routine for running the experiment itself.
    """
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





def test_downsampling():
    img = cv2.imread("/Users/aidan/PycharmProjects/DYSCO/Data/TestSet1/test1/cam-low-exposure.png")


    area = downsample(img, 5, 10)
    cv2.imshow("downsampled", area)
    cv2.waitKey()


if __name__ == "__main__":
    test_downsampling()
