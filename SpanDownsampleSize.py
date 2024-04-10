"""
This file is for doing experiments to see how accuracy changes with different down sampling sizes.
The idea is to produce a graph with number of rows/cols on the x-axis and accuracy on the y-axis.

So, I need a routine that just gets the accuracy.
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





def generate_data(rows, cols, color=True, texture=True, edge=True, edge_detector='canny', canny_thresh1=50,
                  canny_thresh2=100, edge_filter='bilateral', bilateral_sigma_color=75,
                  bilateral_sigma_space=75, bilateral_filter_size=3, debug=False):
    """
    This function is to be used to grab the color/edge/texture data for the eight test set 1 images.
    They are scaled and flattened, then returned as a list where each element is just one of the images.
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

    #If the option is selected, auto_rows, then determine number of rows from given number of columns and aspect ratio
    # rows = cols * height/width



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
            #dowsnsample single channel from the observed image
            single_channel = cv2.resize(observed[:, :, channel:channel+1], (cols, rows), interpolation=cv2.INTER_AREA)
            single_channels_observed.append(single_channel)

            #downsample the  single channel from the corresponding expected image.
            single_channels_expected.append(cv2.resize(expected[:, :, channel:channel+1], (cols, rows), interpolation=cv2.INTER_AREA))


        #Recombine the single texture channel images into a multi-spectral texture image.
        observed_downsampled = np.stack(single_channels_observed, -1)  # concatenate
        expected_downsampled = np.stack(single_channels_expected, -1)  # concatenate


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
        data = min_max_scaler.fit_transform(flat) # apply the scaling

        #now grab the ground truth
        gt = test_images[i]['ground_truth']
        #gt = gt.reshape(-1, 1 if len(gt.shape) == 2 else gt.shape[2])


        # append the data and the ground truth to a list
        final_data.append({'data': data, 'ground_truth': gt})

    #return the final data for processing
    return final_data




def generate_data2(rows, cols, color=True, texture=True, edge=True, edge_detector='canny', canny_thresh1=50,
                  canny_thresh2=100, edge_filter='bilateral', bilateral_sigma_color=75,
                  bilateral_sigma_space=75, bilateral_filter_size=3, debug=False, auto_rows=False):
    """
    This function is to be used to grab the color/edge/texture data for the eight test set 1 images.
    They are scaled and flattened, then returned as a list where each element is just one of the images.
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

    #If the option is selected, auto_rows, then determine number of rows from given number of columns and aspect ratio
    # rows = cols * height/width


    #iterate over all images
    for i in range(len(test_images)):
        #grab the test images
        observed = test_images[i]['observed']
        expected = test_images[i]['expected']

        #auto rows option
        if auto_rows:
            # determine number of rows from aspect ratio
            rows = int(cols * (observed.shape[0] / observed.shape[1]))  # rows = cols * height/width


        #Color
        if color:
            # downsample observed and expected for comparison
            observed_downsampled = cv2.resize(observed, (cols, rows), interpolation=cv2.INTER_AREA)
            expected_downsampled = cv2.resize(expected, (cols, rows), interpolation=cv2.INTER_AREA)

            # take difference between the two images
            difference = np.absolute(observed_downsampled.astype(int) - expected_downsampled.astype(int))
            color_deltas.append(difference)


        #texture
        if texture:
            # to avoid the bug, I have to split textures into separate channels, before down-sampling
            single_channels_observed = []
            single_channels_expected = []
            for channel in range(observed.shape[2]):
                # dowsnsample single channel from the observed image
                single_channel = cv2.resize(observed[:, :, channel:channel + 1], (cols, rows),
                                            interpolation=cv2.INTER_AREA)
                single_channels_observed.append(single_channel)

                # downsample the  single channel from the corresponding expected image.
                single_channels_expected.append(
                    cv2.resize(expected[:, :, channel:channel + 1], (cols, rows), interpolation=cv2.INTER_AREA))

            # Recombine the single texture channel images into a multi-spectral texture image.
            observed_downsampled = np.stack(single_channels_observed, -1)  # concatenate
            expected_downsampled = np.stack(single_channels_expected, -1)  # concatenate

            # take difference between the two images
            difference = np.absolute(observed_downsampled.astype(int) - expected_downsampled.astype(int))
            texture_deltas.append(difference)



        #edge
        if edge:
            # convert images to grayscale
            observed_gray = cv2.cvtColor(observed, cv2.COLOR_BGR2GRAY)
            expected_gray = cv2.cvtColor(expected, cv2.COLOR_BGR2GRAY)


            # bilateral filter
            if edge_filter == 'bilateral':
                observed_gray = cv2.bilateralFilter(observed_gray, bilateral_filter_size, bilateral_sigma_color,
                                               bilateral_sigma_space)
                expected_gray = cv2.bilateralFilter(expected_gray, bilateral_filter_size, bilateral_sigma_color,
                                               bilateral_sigma_space)

            # canny
            observed_gray = cv2.Canny(observed_gray, canny_thresh1, canny_thresh2)
            expected_gray = cv2.Canny(expected_gray, canny_thresh1, canny_thresh2)

            # downsampling
            observed_gray = cv2.resize(observed_gray, (cols, rows), interpolation=cv2.INTER_AREA)
            expected_gray = cv2.resize(expected_gray, (cols, rows), interpolation=cv2.INTER_AREA)

            # Difference
            difference = np.absolute(observed_gray.astype(int) - expected_gray.astype(int))

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
        data = min_max_scaler.fit_transform(flat) # apply the scaling

        #now grab the ground truth
        gt = test_images[i]['ground_truth']
        #gt = gt.reshape(-1, 1 if len(gt.shape) == 2 else gt.shape[2])


        # append the data and the ground truth to a list
        final_data.append({'data': data, 'ground_truth': gt, 'rows': rows, 'cols': cols})

    #return the final data for processing
    return final_data

#subroutine for testing with various downsampling sizes, used below
def varying_downsample_size_helper(data):
    # object to accumulate results
    results = CM.ConfusionMatrix()

    # Iterate over each test in data.
    for d in data:
        X = d['data']
        gt = d['ground_truth']
        r = d['rows']
        c = d['cols']


        """
        # label with clustering, obstructed is higher cluster center
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)

        if np.linalg.norm(kmeans.cluster_centers_[0, :]) > np.linalg.norm(kmeans.cluster_centers_[1, :]):
            prediction_matrix = (kmeans.labels_ == 1).tolist() # the first centroid is bigger
        else:
            prediction_matrix = (kmeans.labels_ == 0).tolist() # the seconds centroid is bigger
        """



        #new cluster with more than 1
        kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(X)
        cluster_centers = [np.linalg.norm(kmeans.cluster_centers_[i, :]) for i in range(5)]
        index_min = np.argmax(cluster_centers)
        prediction_matrix = (kmeans.labels_ != index_min).tolist()  # the seconds centroid is bigger



        # Grab the ground truth with the booleans for each of the down-sampled regions
        gt = create_gt_array(gt, r, c)
        gt = list(np.array(gt).flatten())

        # Accumulate data in confusion matrix
        assert len(prediction_matrix) == len(gt), f"Disparate data size of prediction and ground truth." \
                                                  f"\nlengths" \
                                                  f"\n\tprediction matrix:{len(prediction_matrix)}" \
                                                  f"\n\tground truth matrix:{len(gt)}"
        for i in range(r * c):
            results.register(gt[i], prediction_matrix[i])

    # metrics
    #print("accuracy:", results.accuracy())
    #print("specificity:", results.specificity())
    return results



def main():
    #grab the data from the images to work with
    # The data variable is a list of dictionaries where the key 'data' is the deltas and 'ground_truth' is the ground truth
    #data = generate_data(10, 20)

    #First, make sure everything works as expected with just one set of data
    #sample = data[0]
    #delta = sample['data']
    #gt = sample['ground_truth']



    number_of_columns = []
    accuracies = []
    sensitivities = []
    specificities = []
    for columns in range(5, 400): # iterate over different column numbers
        number_of_columns.append(columns)
        rows = columns//2


        data = generate_data2(rows, columns, auto_rows=True)
        results = varying_downsample_size_helper(data)
        accuracies.append(results.accuracy() * 100)
        sensitivities.append(results.sensitivity() * 100)
        specificities.append(results.specificity() * 100)

    #make plot
    plt.plot(number_of_columns, accuracies, label='Accuracy', color='red')
    plt.plot(number_of_columns, specificities, label='Specificity', color='blue')
    plt.plot(number_of_columns, sensitivities, label='Sensitivity', color='green')

    #set the scale and ticks on the y-axis
    plt.ylim(0, 100) # Set the y-axis limits
    plt.yticks(range(0, 101, 5)) # Set the y-axis tick marks at every 5 units


    plt.xlabel('Number of Columns')
    plt.ylabel('Accuracy')
    plt.title('No texture Features')
    plt.legend()
    plt.show()





if __name__ == "__main__":
    main()
