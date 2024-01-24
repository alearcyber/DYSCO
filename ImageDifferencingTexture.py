"""
The goal of this test is to accomplish the same thing as ImageDifferencingTest.py, except to use texture instead
of color.
"""
import numpy as np
import cv2
from sklearn.cluster import KMeans
from Verify import *

####################################################################################################
#
####################################################################################################

# filter vectors
level = np.array([[1, 4, 6, 4, 1]])
edge = np.array([[-1, -2, 0, 2, 1]])
spot = np.array([[-1, 0, 2, 0, -1]])
wave = np.array([[-1, 2, 0, -2, 1]])
ripple = np.array([[1, -4, 6, -4, 1]])

# edge kernels
el = np.dot(edge.reshape(-1, 1), level)
ee = np.dot(edge.reshape(-1, 1), edge)
es = np.dot(edge.reshape(-1, 1), spot)
ew = np.dot(edge.reshape(-1, 1), wave)
er = np.dot(edge.reshape(-1, 1), ripple)

# level kernels
ll = np.dot(level.reshape(-1, 1), level)
le = np.dot(level.reshape(-1, 1), edge)
ls = np.dot(level.reshape(-1, 1), spot)
lw = np.dot(level.reshape(-1, 1), wave)
lr = np.dot(level.reshape(-1, 1), ripple)

# spot kernels
sl = np.dot(spot.reshape(-1, 1), level)
se = np.dot(spot.reshape(-1, 1), edge)
ss = np.dot(spot.reshape(-1, 1), spot)
sw = np.dot(spot.reshape(-1, 1), wave)
sr = np.dot(spot.reshape(-1, 1), ripple)

# wave kernels
wl = np.dot(wave.reshape(-1, 1), level)
we = np.dot(wave.reshape(-1, 1), edge)
ws = np.dot(wave.reshape(-1, 1), spot)
ww = np.dot(wave.reshape(-1, 1), wave)
wr = np.dot(wave.reshape(-1, 1), ripple)

# ripple kernels
rl = np.dot(ripple.reshape(-1, 1), level)
re = np.dot(ripple.reshape(-1, 1), edge)
rs = np.dot(ripple.reshape(-1, 1), spot)
rw = np.dot(ripple.reshape(-1, 1), wave)
rr = np.dot(ripple.reshape(-1, 1), ripple)

epsilon = 1e-8
energy_maps = {
    'L5E5/E5L5': np.round(le / (el + epsilon), decimals=2),
    'L5R5/R5L5': lr // rl,
    'E5S5/S5E5': np.round(es / (se + epsilon), decimals=2),
    'S5S5': ss,
    'R5R5': rr,
    'L5S5/S5L5': np.round(ls / (sl + epsilon), decimals=2),
    'E5E5': ee,
    'E5R5/R5E5': np.round(er / (re + epsilon), decimals=2),
    'S5R5/R5S5': np.round(sr / (rs + epsilon), decimals=2)
}


def texture_features(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = np.empty(shape=(img.shape[0], img.shape[1], 0), dtype='uint8')
    for name in energy_maps:
        energy_map = energy_maps[name]
        layer = cv2.filter2D(img, ddepth=-1, kernel=energy_map)
        if len(img.shape) == 2: #for grayscale images
            layer = np.expand_dims(layer, axis=2)
        out = np.concatenate((out, layer), axis=2)
    return out




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
        #first thing to do is to convert the images to their texture images
        observed = texture_features(test['observed'])
        expected = texture_features(test['expected'])

        #original shape
        original_shape = observed.shape


        #downsample observed and expected for comparison
        observed_downsampled = cv2.resize(observed, (cols, rows), interpolation=cv2.INTER_AREA)
        expected_downsampled = cv2.resize(expected, (cols, rows), interpolation=cv2.INTER_AREA)


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




def main():
    """
    """
    # Get save location from user
    save_folder = input("Input the local directory where you want the results to be saved, "
                        "or leave blank to show results 1-by-1 with opencv:")

    # Run experiment
    if len(save_folder) < 2:  # dont save, show with opencv
        experiment()
    else:  # Save to output folder
        experiment(out_dir=save_folder)

    # Let user know experiment is done
    print('Done!')




if __name__ == "__main__":
    for key in energy_maps:
        print(energy_maps[key])


    main()
