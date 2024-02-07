"""
each *_features() function returns a list of each difference matrix.
"""
import cv2
import numpy as np
from Texture import texture_features
from sklearn.cluster import KMeans
from Verify import *
from sklearn import preprocessing






"""
This is the first version of the experiment, no scaling or anyhting liek that
"""
def experiment():
    # parameters
    rows = 10
    cols = 20
    dir = "Data/TestSet1"
    bilateral_filter_size = 3


    test_images = []  # test data as dictionaries
    color_deltas = []
    texture_deltas = []
    edge_deltas = []


    # Read the test images into memory.
    # Each test is a dictionary inside the list, test_images.
    for k in range(1, 9):
        test_images.append({
            'observed': cv2.imread(f"{dir}/test{k}/cam-low-exposure.png"),
            'expected': cv2.imread(f"{dir}/test{k}/expected-expanded.png"),
            'ground_truth': cv2.imread(f"{dir}/test{k}/obstructionmask.png")
        })


    #Color Differences
    for i in range(len(test_images)):
        test = test_images[i]
        # downsample observed and expected for comparison
        observed_downsampled = cv2.resize(test['observed'], (cols, rows), interpolation=cv2.INTER_AREA)
        expected_downsampled = cv2.resize(test['expected'], (cols, rows), interpolation=cv2.INTER_AREA)

        # take difference between the two images
        difference = np.absolute(observed_downsampled.astype(int) - expected_downsampled.astype(int))
        color_deltas.append(difference)




    # Texture differences
    for i in range(len(test_images)):
        test = test_images[i]
        # first thing to do is to convert the images to their texture images
        observed = texture_features(test['observed'])
        expected = texture_features(test['expected'])

        # downsample observed and expected for comparison
        observed_downsampled = cv2.resize(observed, (cols, rows), interpolation=cv2.INTER_AREA)
        expected_downsampled = cv2.resize(expected, (cols, rows), interpolation=cv2.INTER_AREA)

        # take difference between the two images
        difference = np.absolute(observed_downsampled.astype(int) - expected_downsampled.astype(int))
        texture_deltas.append(difference)




    #Edge differences
    for i in range(len(test_images)):
        test = test_images[i]

        # Convert the images to grayscale
        observed = cv2.cvtColor(test['observed'], cv2.COLOR_BGR2GRAY)
        expected = cv2.cvtColor(test['expected'], cv2.COLOR_BGR2GRAY)

        # bilateral filter
        observed = cv2.bilateralFilter(observed, bilateral_filter_size, 75, 75)
        expected = cv2.bilateralFilter(expected, bilateral_filter_size, 75, 75)

        # canny
        observed = cv2.Canny(observed, 100, 200)
        expected = cv2.Canny(expected, 100, 200)

        cv2.imshow("edges", observed)

        # downsampling
        observed = cv2.resize(observed, (cols, rows), interpolation=cv2.INTER_AREA)
        expected = cv2.resize(expected, (cols, rows), interpolation=cv2.INTER_AREA)

        cv2.imshow("downsampled", observed)

        # Difference
        difference = np.absolute(observed.astype(int) - expected.astype(int))

        #expand back up to 3 dimensions
        difference = np.expand_dims(difference, axis=-1)

        #append to list
        edge_deltas.append(difference)



    #concatenate all of the features together
    for i in range(8):
        #concatenate pixel vectors
        combined_image = np.concatenate((color_deltas[i], texture_deltas[i], edge_deltas[i]), -1)

        # Cluster the outcome
        flat = combined_image.reshape(-1, combined_image.shape[2]) # flatten first
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


        #Visualize the prediction
        visualized_prediction = visualize_region_matrix(test_images[i]['observed'], prediction_matrix)  # visualized results
        cv2.imshow(f'Prediction {i + 1}', visualized_prediction)
        cv2.waitKey()
        cv2.destroyAllWindows()











"""
with scaling
"""
def experiment_norm():
    # parameters
    rows = 10
    cols = 20
    dir = "Data/TestSet1"
    bilateral_filter_size = 3


    test_images = []  # test data as dictionaries
    color_deltas = []
    texture_deltas = []
    edge_deltas = []


    # Read the test images into memory.
    # Each test is a dictionary inside the list, test_images.
    for k in range(1, 9):
        test_images.append({
            'observed': cv2.imread(f"{dir}/test{k}/cam-low-exposure.png"),
            'expected': cv2.imread(f"{dir}/test{k}/expected-expanded.png"),
            'ground_truth': cv2.imread(f"{dir}/test{k}/obstructionmask.png")
        })


    #Color Differences
    for i in range(len(test_images)):
        test = test_images[i]
        # downsample observed and expected for comparison
        observed_downsampled = cv2.resize(test['observed'], (cols, rows), interpolation=cv2.INTER_AREA)
        expected_downsampled = cv2.resize(test['expected'], (cols, rows), interpolation=cv2.INTER_AREA)

        # take difference between the two images
        difference = np.absolute(observed_downsampled.astype(int) - expected_downsampled.astype(int))
        color_deltas.append(difference)




    # Texture differences
    for i in range(len(test_images)):
        test = test_images[i]
        # first thing to do is to convert the images to their texture images
        observed = texture_features(test['observed'])
        expected = texture_features(test['expected'])

        # downsample observed and expected for comparison
        observed_downsampled = cv2.resize(observed, (cols, rows), interpolation=cv2.INTER_AREA)
        expected_downsampled = cv2.resize(expected, (cols, rows), interpolation=cv2.INTER_AREA)

        # take difference between the two images
        difference = np.absolute(observed_downsampled.astype(int) - expected_downsampled.astype(int))
        texture_deltas.append(difference)




    #Edge differences
    for i in range(len(test_images)):
        test = test_images[i]

        # Convert the images to grayscale
        observed = cv2.cvtColor(test['observed'], cv2.COLOR_BGR2GRAY)
        expected = cv2.cvtColor(test['expected'], cv2.COLOR_BGR2GRAY)

        # bilateral filter
        observed = cv2.bilateralFilter(observed, bilateral_filter_size, 75, 75)
        expected = cv2.bilateralFilter(expected, bilateral_filter_size, 75, 75)

        # canny
        observed = cv2.Canny(observed, 100, 200)
        expected = cv2.Canny(expected, 100, 200)

        cv2.imshow("edges", observed)

        # downsampling
        observed = cv2.resize(observed, (cols, rows), interpolation=cv2.INTER_AREA)
        expected = cv2.resize(expected, (cols, rows), interpolation=cv2.INTER_AREA)

        cv2.imshow("downsampled", observed)

        # Difference
        difference = np.absolute(observed.astype(int) - expected.astype(int))

        #expand back up to 3 dimensions
        difference = np.expand_dims(difference, axis=-1)

        #append to list
        edge_deltas.append(difference)



    #concatenate all of the features together
    for i in range(8):
        #concatenate pixel vectors
        combined_image = np.concatenate((color_deltas[i], texture_deltas[i], edge_deltas[i]), -1)

        # Cluster the outcome
        flat = combined_image.reshape(-1, combined_image.shape[2]) # flatten first

        #SCALING DONE HERE
        min_max_scaler = preprocessing.MinMaxScaler()
        flat_scaled = min_max_scaler.fit_transform(flat)


        #now continue with clustering the SCALED data, notice the change to the line below
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(flat_scaled)
        #print('labels:', kmeans.labels_)
        #print('Cluster centers:', kmeans.cluster_centers_)
        reshaped_labels = kmeans.labels_.reshape(rows, cols)
        #print('----reshaped labels----\n', reshaped_labels)

        # figure out which cluster center is the highest so I know which area is the
        # obstructed area, and which is unobstructed. The highest centroid is the obstructed area.
        if np.linalg.norm(kmeans.cluster_centers_[0, :]) > np.linalg.norm(kmeans.cluster_centers_[1, :]):
            # the 0th centroid is bigger
            prediction_matrix = (reshaped_labels == 1).tolist()
        else:
            # the first centroid is bigger
            prediction_matrix = (reshaped_labels == 0).tolist()


        #Visualize the prediction
        visualized_prediction = visualize_region_matrix(test_images[i]['observed'], prediction_matrix)  # visualized results
        cv2.imshow(f'Prediction {i + 1}', visualized_prediction)
        cv2.waitKey()
        cv2.destroyAllWindows()





"""
elbow plot here
"""








if __name__ == "__main__":
    experiment_norm()
