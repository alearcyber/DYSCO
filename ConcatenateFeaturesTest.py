

"""
each *_features() function returns a list of each difference matrix.
"""
import cv2
import numpy as np
from Texture import texture_features
from sklearn.cluster import KMeans
from Verify import *
import ConfusionMatrix as CM
from sklearn import preprocessing
import Dysco
import matplotlib.pyplot as plt






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
def experiment_with_elbow():
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

        #create elbow plot
        Dysco.elbow_plot(flat_scaled, range(2, 10))

        #now continue with clustering the SCALED data, notice the change to the line below
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(flat_scaled)
        reshaped_labels = kmeans.labels_.reshape(rows, cols)

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

        #save images
        #cv2.imwrite(f'Results/ElbowPlots/Test{i + 1}/predictions.png', visualized_prediction)


        cv2.imshow(f'Prediction {i + 1}', visualized_prediction)
        cv2.waitKey()
        cv2.destroyAllWindows()



def experiment_elbow_no_color():
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

    # Color Differences
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

    # Edge differences
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

        # expand back up to 3 dimensions
        difference = np.expand_dims(difference, axis=-1)

        # append to list
        edge_deltas.append(difference)

    # concatenate all of the features together
    for i in range(8):
        # concatenate pixel vectors
        #combined_image = np.concatenate((color_deltas[i], texture_deltas[i], edge_deltas[i]), -1)
        combined_image = np.concatenate((texture_deltas[i], edge_deltas[i]), -1)

        # Cluster the outcome
        flat = combined_image.reshape(-1, combined_image.shape[2])  # flatten first

        # SCALING DONE HERE
        min_max_scaler = preprocessing.MinMaxScaler()
        flat_scaled = min_max_scaler.fit_transform(flat)

        # create elbow plot
        Dysco.elbow_plot(flat_scaled, range(2, 10))

        # now continue with clustering the SCALED data, notice the change to the line below
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(flat_scaled)
        reshaped_labels = kmeans.labels_.reshape(rows, cols)

        # figure out which cluster center is the highest so I know which area is the
        # obstructed area, and which is unobstructed. The highest centroid is the obstructed area.
        if np.linalg.norm(kmeans.cluster_centers_[0, :]) > np.linalg.norm(kmeans.cluster_centers_[1, :]):
            # the 0th centroid is bigger
            prediction_matrix = (reshaped_labels == 1).tolist()
        else:
            # the first centroid is bigger
            prediction_matrix = (reshaped_labels == 0).tolist()

        # Visualize the prediction
        visualized_prediction = visualize_region_matrix(test_images[i]['observed'],
                                                        prediction_matrix)  # visualized results

        # save images
        # cv2.imwrite(f'Results/ElbowPlots/Test{i + 1}/predictions.png', visualized_prediction)

        cv2.imshow(f'Prediction {i + 1}', visualized_prediction)
        cv2.waitKey()
        cv2.destroyAllWindows()


def generate_data(rows, cols, color=True, texture=True, edge=True, edge_detector='canny', canny_thresh1=50,
                  canny_thresh2=100, edge_filter='bilateral', bilateral_sigma_color=75,
                  bilateral_sigma_space=75, bilateral_filter_size = 3, debug=False):
    """
    THis function is to be used to grab the color/edge/texture data for the eight test set 1 images.
    They are scaled and flattened, then returned as a list where each element is just one of the images.
    This will be used for further visualizing the data.
    """
    # parameters
    dir = "Data/TestSet1"

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

    # determine number of rows
    if rows is None:
        rows = Dysco.row_selection(cols, test_images[0]['observed'])

    # test to make sure wierd bug with resize doesn't occur.
    # the minimum number of rows or column when downsampling is 5, 4 or less will cause an error.
    """
    if rows < 5:
        rows = 5
        print("WARNING: Number of rows less than or equal to 4. This causes weird issues with resizing in opencv")
    """

    # Color Differences
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

        #to avoid the bug, I have to split textures into separate channels, before down-sampling
        single_channels_observed = []
        single_channels_expected = []
        for channel in range(observed.shape[2]):
            #obserevd
            single_channel = cv2.resize(observed[:, :, channel:channel+1], (cols, rows), interpolation=cv2.INTER_AREA)
            single_channels_observed.append(single_channel)

            #expected
            single_channels_expected.append(cv2.resize(expected[:, :, channel:channel+1], (cols, rows), interpolation=cv2.INTER_AREA))




        observed_downsampled = np.concatenate(single_channels_observed, -1)  # concatenate
        expected_downsampled = np.concatenate(single_channels_expected, -1)  # concatenate

        print("recombo shape:", observed_downsampled.shape)




        # downsample observed and expected for comparison
        #observed_downsampled = cv2.resize(observed, (cols, rows), interpolation=cv2.INTER_AREA)
        #expected_downsampled = cv2.resize(expected, (cols, rows), interpolation=cv2.INTER_AREA)

        # take difference between the two images
        difference = np.absolute(observed_downsampled.astype(int) - expected_downsampled.astype(int))
        texture_deltas.append(difference)

    # Edge differences
    for i in range(len(test_images)):
        test = test_images[i]

        # Convert the images to grayscale
        observed = cv2.cvtColor(test['observed'], cv2.COLOR_BGR2GRAY)
        expected = cv2.cvtColor(test['expected'], cv2.COLOR_BGR2GRAY)

        # bilateral filter
        if edge_filter == 'bilateral':
            observed = cv2.bilateralFilter(observed, bilateral_filter_size, bilateral_sigma_color, bilateral_sigma_space)
            expected = cv2.bilateralFilter(expected, bilateral_filter_size, bilateral_sigma_color, bilateral_sigma_space)

        # canny
        observed = cv2.Canny(observed, canny_thresh1, canny_thresh2)
        expected = cv2.Canny(expected, canny_thresh1, canny_thresh2)

        if debug:
            cv2.imshow("edges", observed)

        # downsampling
        observed = cv2.resize(observed, (cols, rows), interpolation=cv2.INTER_AREA)
        expected = cv2.resize(expected, (cols, rows), interpolation=cv2.INTER_AREA)

        # Difference
        difference = np.absolute(observed.astype(int) - expected.astype(int))

        # expand back up to 3 dimensions
        difference = np.expand_dims(difference, axis=-1)

        # append to list
        edge_deltas.append(difference)

    # Concatenate features
    final_data = []
    for i in range(8):
        if color and texture and edge:
            features = (color_deltas[i], texture_deltas[i], edge_deltas[i])
        elif color and texture:
            features = (color_deltas[i], texture_deltas[i])
        elif color and edge:
            features = (color_deltas[i], edge_deltas[i])
        elif texture and edge:
            features = (texture_deltas[i], edge_deltas[i])
        elif color:
            features = (color_deltas[i],)
        elif texture:
            features = (texture_deltas[i],)
        else: #just edge features
            features = (edge_deltas[i],)

        #process the data
        combined_image = np.concatenate(features, -1) # concatenate
        flat = combined_image.reshape(-1, combined_image.shape[2])  # flatten
        min_max_scaler = preprocessing.MinMaxScaler() # scale the data
        observed_data = min_max_scaler.fit_transform(flat) # apply the scaling

        #now grab the ground truth
        gt = test_images[i]['ground_truth']
        #gt = gt.reshape(-1, 1 if len(gt.shape) == 2 else gt.shape[2])


        # append the data and the ground truth to a list
        final_data.append({'data': observed_data, 'ground_truth': gt})

    #return the final data for processing
    return final_data


#subroutine for testing with various downsampling sizes, used below
def varying_downsample_size_helper(r, c):

    #fail examples
    #data = generate_data(8, 10)

    #Works correctly examples
    #data = generate_data(10, 20)


    #retreive data
    data = generate_data(r, c)

    # object to accumulate results
    results = CM.ConfusionMatrix()

    for d in data:
        X = d['data']
        gt = d['ground_truth']


        # label with clustering, obstructed is higher cluster center
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
        if np.linalg.norm(kmeans.cluster_centers_[0, :]) > np.linalg.norm(kmeans.cluster_centers_[1, :]):
            prediction_matrix = (kmeans.labels_ == 1).tolist() # the first centroid is bigger
        else:
            prediction_matrix = (kmeans.labels_ == 0).tolist() # the seconds centroid is bigger



        # Grab the ground truth with the booleans for each of the down-sampled regions
        gt = create_gt_array(gt, r, c)
        gt = list(np.array(gt).flatten())


        # Accumulate data in confusion matrix
        assert len(prediction_matrix) == len(gt), "Disparate data size of prediction and ground truth"
        for i in range(r * c):
            results.register(gt[i], prediction_matrix[i])

    # metrics
    print("accuracy:", results.accuracy())
    print("specificity:", results.specificity())
    return results




#experiment with using many different down-sampling sizes
def varying_downsample_size():

    #iterate over and accumulate data
    cols = []
    accuracy = []
    specificity = []
    for c in range(3, 41):
        r = c//2 #using a ratio of 2:1 for now, maybe change later

        # Do the predictions and grab results
        # The try catch is to eliminate that weird bug
        try:
            results = varying_downsample_size_helper(r, c)
            cols.append(c)
            accuracy.append(results.accuracy())
            specificity.append(results.specificity())
        except cv2.error:
            print(f"Error with {r} rows and {c} columns.")



    #plot the data
    plt.plot(cols, accuracy, '-o', label='Accuracy', color='blue')
    plt.plot(cols, specificity, '-o', label='Specificity', color='red')
    plt.xlabel('Number of columns to down-sample')
    plt.ylabel('Accuracy & Specificity')
    plt.legend()  # Displays a legend to identify the functions
    plt.show()



def demonstrate_bug():
    data = generate_data(8, 10)










if __name__ == "__main__":
    # experiment_with_elbow()
    #varying_downsample_size()
    demonstrate_bug()
