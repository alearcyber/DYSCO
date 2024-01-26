"""
Nearly identical to ImageDifferencingTest.py and ImageDifferencingTexture.py, except this will
use edge features.

4 different edge detectors:
canny, sobel, laplacian, and fourier
    - sobel will be a 3 feature vector of a vertical, horizontal, and combined filters
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
from Verify import *




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
    cv2.waitKey()
    cv2.destroyAllWindows()

    for i in range(3, 10, 2):
        blurred_image = cv2.bilateralFilter(test_image, i, 75, 75)
        edges = cv2.Canny(blurred_image, 100, 200)
        cv2.imshow(f'bilateral blur then canny, d={i}', edges)
        cv2.waitKey()
        cv2.destroyAllWindows()

def sobel_example():
    #read in a test image
    test_image = cv2.imread("Data/TestSet1/test1/cam-low-exposure.png")

    #make original and blurred version of image
    x = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    xb = cv2.bilateralFilter(x, 7, 75, 75)

    # Sobel original
    sobelx = cv2.Sobel(src=x, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=x, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=x, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection

    # sobel blurred
    sobelx_b = cv2.Sobel(src=xb, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely_b = cv2.Sobel(src=xb, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    sobelxy_b = cv2.Sobel(src=xb, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection

    #show images
    cv2.imshow("original", test_image)
    cv2.imshow('sobel x', sobelx)
    cv2.imshow('sobel y', sobely)
    cv2.imshow('sobel xy', sobelxy)
    cv2.imshow('sobel x blurred',  sobelx_b)
    cv2.imshow('sobel y blurred', sobely_b)
    cv2.imshow('sobel xy blurred', sobelxy_b)
    cv2.waitKey()
    cv2.destroyAllWindows()




def test1():
    """
    This function runs the following test:
    grayscale image, 3x3 bilateral filter(blur), canny edge, 10 row x 20 col downsample,
    difference between observed and expected, k-means cluster.
    """
    #number of rows and columns working with for this experiment
    rows = 10
    cols = 20
    bilateral_filter_size = 3


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





    # Preprocess the observed and expected images...
    # No preprocessing, just downsample and chi squared distance
    i = 0
    for test in test_images:
        i += 1

        #Convert the images to grayscale
        observed = cv2.cvtColor(test['observed'], cv2.COLOR_BGR2GRAY)
        expected = cv2.cvtColor(test['expected'], cv2.COLOR_BGR2GRAY)


        #bilateral filter
        observed = cv2.bilateralFilter(observed, bilateral_filter_size, 75, 75)
        expected = cv2.bilateralFilter(expected, bilateral_filter_size, 75, 75)

        #canny
        observed = cv2.Canny(observed, 100, 200)
        expected = cv2.Canny(expected, 100, 200)

        #downsampling
        observed = cv2.resize(observed, (cols, rows), interpolation=cv2.INTER_AREA)
        expected = cv2.resize(expected, (cols, rows), interpolation=cv2.INTER_AREA)

        #Difference
        difference = np.absolute(observed.astype(int) - expected.astype(int))



        # Cluster the outcome
        flat = difference.reshape(-1, 1)  # flattened difference image
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(flat)
        print('labels:', kmeans.labels_)
        print('Cluster centers:', kmeans.cluster_centers_)
        reshaped_labels = kmeans.labels_.reshape(rows, cols)
        print('----reshaped labels----\n', reshaped_labels)

        # figure out which cluster center is the highest so I know which area is the
        # obstructed area, and which is unobstructed. The highest centroid is the obstructed area.
        if np.linalg.norm(kmeans.cluster_centers_[0, :]) > np.linalg.norm(kmeans.cluster_centers_[1, :]):
            # the 0th centroid is bigger
            prediction_matrix = (reshaped_labels == 1).tolist()
        else:
            # the first centroid is bigger
            prediction_matrix = (reshaped_labels == 0).tolist()



        #visualize the outcomes
        visualized_prediction = visualize_region_matrix(test['observed'], prediction_matrix)  # visualized results
        cv2.imshow(f'Prediction {i}', visualized_prediction)
        cv2.waitKey()
        cv2.destroyAllWindows()





def test2():
    pass





def main():
    options = "Test Options:\n" \
              "\t1) grayscale image, 3x3 bilateral filter(blur), canny edge, 10 row x 20 col downsample, difference between observed and expected, k-means cluster." \
              "\n\t2) ...." \
              "\n\t3) ...." \
              "\n\t4) ...." \
              "\n\t9) One of the example functions." \
              "\nWhich test would you like to run?:" \

    test_choice = int(input(options))



    #case for one of the example functions
    if test_choice == 9:
        options = "Example Options:\n" \
                  "\t1) Bilateral Blur" \
                  "\n\t2) Bilateral Canny" \
                  "\n\t3) Sobel" \
                  "\n\t4) ...." \
                  "\nWhich example would you like to run?:"
        example_choice = int(input(options))

        #choose case
        if example_choice == 1:
            bilateral_blur_example()
        elif example_choice == 2:
            bilateral_canny_example()
        elif example_choice == 3:
            sobel_example()

    if test_choice == 1:
        test1()



    print("YOU CHOSE:", test_choice)







if __name__ == "__main__":
    main()





