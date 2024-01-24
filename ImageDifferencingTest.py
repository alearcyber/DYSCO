"""
---Test Description---
The purpose of this script is to run a test for doing a simple difference of the expected image and the picture
of the display.
"""



import cv2
import numpy as np
from sklearn.cluster import KMeans
from Verify import *


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



def convert_bin_to_bool(array):
    """
    Converts a 2d numpy array whose entries are only either 1 or 0 to a boolean array of the same dimensions.
    Instead of simply converting 1 to True and 0 to False, If 1s appear less often than 0s, the 1s become False
    and 0s become True.


    TODO - REWRITE THIS so that the custer corresponding to the highest centroid is made false, i.e. obstructed
    """
    # Count the occurrences of 1s and 0s
    ones_count = np.count_nonzero(array)
    zeros_count = array.size - ones_count

    # Convert the array based on the frequency of 1s and 0s
    if ones_count < zeros_count:
        # Invert the values if 1s are less frequent
        return array == 0
    else:
        # Direct conversion otherwise
        return array == 1


def experiment(out_dir=None):
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


        #downsample observed and expected for comparison
        observed_downsampled = cv2.resize(test['observed'], (cols, rows), interpolation=cv2.INTER_AREA)
        expected_downsampled = cv2.resize(test['expected'], (cols, rows), interpolation=cv2.INTER_AREA)


        #take difference between the two images
        difference = np.absolute(observed_downsampled.astype(int) - expected_downsampled.astype(int))



        """
        #this next set of lines is for visualizing the difference
        bloom_difference = cv2.resize(difference, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)
        cv2.imshow('observed', cv2.resize(observed_downsampled, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST_EXACT))
        cv2.imshow('expected', cv2.resize(expected_downsampled, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST_EXACT))
        cv2.imshow(f'Difference {i}', bloom_difference.astype('uint8'))
        cv2.waitKey()
        cv2.destroyAllWindows()
        """


        #Cluster the outcome
        flat = difference.reshape(-1, difference.shape[2]) # flattened image first
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(flat)
        print('labels:', kmeans.labels_)
        print('Cluster centers:', kmeans.cluster_centers_)
        reshaped_labels = kmeans.labels_.reshape(rows, cols)
        print('----reshaped labels----\n', reshaped_labels)



        #figure out which cluster center is the highest so I know which area is the
        #obstructed area, and which is unobstructed. The highest centroid is the obstructed area.
        if np.linalg.norm(kmeans.cluster_centers_[0, :]) > np.linalg.norm(kmeans.cluster_centers_[1, :]):
            #the 0th centroid is bigger
            prediction_matrix = (reshaped_labels == 1).tolist()
        else:
            #the first centroid is bigger
            prediction_matrix = (reshaped_labels == 0).tolist()




        #prediction_matrix = convert_bin_to_bool(reshaped_labels).tolist() #2d bool list of the prediction
        visualized_prediction = visualize_region_matrix(test['observed'], prediction_matrix) #visualized results

        #Should the image be saved or displayed
        if out_dir is None: #display it
            cv2.imshow(f'Prediction {i}', visualized_prediction)
            cv2.waitKey()
            cv2.destroyAllWindows()
        else: #save it
            if not (out_dir.endswith('/')):
                out_dir.append('/')
            cv2.imwrite(f'{out_dir}Test{i}.png', visualized_prediction)









def test_downsampling():
    img = cv2.imread("/Users/aidan/PycharmProjects/DYSCO/Data/TestSet1/test1/cam-low-exposure.png")


    area = downsample(img, 5, 10)
    cv2.imshow("downsampled", area)
    cv2.waitKey()



def main():
    """
    """
    #Get save location from user
    save_folder = input("Input the local directory where you want the results to be saved, "
                        "or leave blank to show results 1-by-1 with opencv:")

    #Run experiment
    if len(save_folder) < 2: #dont save, show with opencv
        experiment()
    else: #Save to output folder
        experiment(out_dir=save_folder)

    #Let user know experiment is done
    print('Done!')




if __name__ == "__main__":
    main()
