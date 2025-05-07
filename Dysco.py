"""
The plan for this function is to have utility routines that are helpful throughout the code

Tests Spreadsheet: https://docs.google.com/spreadsheets/d/1mhoO8N5Z-TylaHwxd7LD56jFhuGWrWUZAdQT9jR2OG0/edit?usp=sharing

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing
import sklearn.metrics as SKmetrics
from itertools import product as CartesianProduct
import os
import re
import skimage
import math
from scipy.ndimage import generic_filter
from PIL import Image
from skimage.exposure import match_histograms
np.set_printoptions(suppress=True)



#a global constant for what random state to use for anything that requires rng
RandomSeed = 25


#Options for features in generate data
COLOR = 1
TEXTURE = 2
EDGE = 4

#options for image shape
FLAT = 1 # image is flattened to 1 dimension
RECT = 2 # image maintains rectangular shape


#Options for edge detector




def row_selection(cols, image):
    """
    Automatically selects the appropriate number of rows for down-sampling given the number of columns and the
    image to be down-sampled.
    :param cols:
    :param image:
    :return:
    """
    h, w, d = image.shape
    rows = round(cols * (h / w))
    if rows < 1:
        rows = 1
    return rows



def aabb(image):
    """
    Removes extra transparent space around object within image to create a new image of the axially aligned
    bounding box
    :param image:
    :return:
    """
    if image.shape[2] < 4:
        raise ValueError("The image array must have an alpha channel.")

        # Find the alpha channel
    alpha_channel = image[:, :, 3]

    # Identify rows and columns that contain non-transparent pixels
    rows_with_content = np.any(alpha_channel != 0, axis=1)
    cols_with_content = np.any(alpha_channel != 0, axis=0)

    # Find the bounding box of the non-transparent areas
    y_min, y_max = np.where(rows_with_content)[0][[0, -1]]
    x_min, x_max = np.where(cols_with_content)[0][[0, -1]]

    # Crop the image using numpy slicing
    cropped_image_array = image[y_min:y_max + 1, x_min:x_max + 1, :]

    return cropped_image_array







def scale_image(image, scaling_factor, interpolation=None):
    """
    Apply the following transformation, where s is the scaling_factor:
        s, 0, 0
        0, s, 0
        0, 0, 1
    :param image:
    :param scaling_factor:
    :param interpolation:
    :return:
    """
    #grab dimensions of the image
    h, w = image.shape[0], image.shape[1]

    #automatically choose an interpolation method if one is not given
    if interpolation is None:
        if scaling_factor< 1.0:  # scaling down
            interpolation = cv2.INTER_AREA
        else:  # scaling up
            interpolation = cv2.INTER_CUBIC

    #resize/scale the image
    return cv2.resize(image, (w, h), interpolation=interpolation)






def elbow_plot(X, k_range, title=None):
    """
    creates an elbow plot
    """
    # perform the pre-clustering
    cost = []
    for i in k_range:
        km = KMeans(n_clusters=i, max_iter=500)
        km.fit(X)
        cost.append(km.inertia_)

    # make the elbow plot
    plt.plot(k_range, cost, color='b')
    plt.xlabel('Value of K')
    plt.ylabel('Squared Error (Cost)')
    if title is not None:
        plt.title(title)
    plt.show()




def read_test_set(directory):
    """
    TODO - Include the screenshot of the obstructed image when reading in the test set. Currently unneeded.
    Reads a test set into memory.
    See Diagrams/TestSetFileStructure.png for details and requirements for properly formatting a test set.
    Returns a list of dictionaries, where each dictionary represents a single test. The keys 'observed',
    'expected', and 'obstructionmask' correspond to images of a test.
    The key, 'id', uniquely identifies the test.
    """
    #Output list
    test_images = []

    #Iterate over items in directory.
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path): # Check if the item is a directory
            match = re.search(r'\d+$', item_path) # grab the id with a regex
            assert match, 'Directories must end with an integer. Found one that does not.' #Verify it has an id

            #create dictionary with results and append to output list
            test_images.append({
                'observed': cv2.imread(f"{item_path}/observed.png"),
                'expected': cv2.imread(f"{item_path}/expected.png"),
                'obstructionmask': cv2.imread(f"{item_path}/obstructionmask.png"),
                'id': int(match.group())
            })

    #status and return
    print(f"Read {len(test_images)} tests into memory...")
    return test_images



def MatchHistograms(image, target):
    """assumes images are in BGR"""
    # Convert images from BGR to HSV
    hsv_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image2 = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    # Extract the Value channel
    v_channel1 = hsv_image1[:, :, 2]
    v_channel2 = hsv_image2[:, :, 2]

    # Match histograms of the value channels
    matched_v_channel = skimage.exposure.match_histograms(v_channel1.astype('float32'), v_channel2.astype('float32'))

    # Replace the Value channel in the second image with the matched one
    hsv_image1[:, :, 2] = matched_v_channel.astype('uint8')

    # Convert back from HSV to BGR
    result = cv2.cvtColor(hsv_image1, cv2.COLOR_HSV2BGR)

    # return matched image
    return result




def adaptive_cluster(X, generate_plot=False):
    """
    Uses various cluster metrics to identify what the appropriate number of clusters should be.

    For silhouette, best is 1, worst is -1, and scores around 0 indicate overlapping clusters.
    For Davies-Bouldin, lower better, best is 0
    """
    #storing results
    sil = []
    db = []
    ch = []

    #for now, just test 2 thru 8 clusters.
    for i in range(2, 9):
        kmeans = KMeans(n_clusters=i, random_state=RandomSeed, n_init="auto").fit(X)
        sil.append(SKmetrics.silhouette_score(X, kmeans.labels_))
        db.append(SKmetrics.davies_bouldin_score(X, kmeans.labels_))
        ch.append(SKmetrics.calinski_harabasz_score(X, kmeans.labels_))


    #make plot
    if generate_plot:
        plt.plot([i for i in range(2, 9)], sil, label='Sil (high good)', color='red')
        plt.plot([i for i in range(2, 9)], db, label='DB (low good)', color='blue')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.title('Cluster Metrics')
        plt.legend()
        plt.show()


    """
    kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(X)
    cluster_centers = [np.linalg.norm(kmeans.cluster_centers_[i, :]) for i in range(5)]
    index_min = np.argmax(cluster_centers)
    prediction_matrix = (kmeans.labels_ != index_min).tolist()  # the seconds centroid is bigger
    """




def generate_data(rows, cols, directory, features=COLOR, edge_detector='canny', canny_thresh1=50,
                  canny_thresh2=100, edge_filter='bilateral', bilateral_sigma_color=75,
                  bilateral_sigma_space=75, bilateral_filter_size=3, debug=False, auto_rows=False,
                  include_location=False, shape=FLAT, preprocess=None):
    """
    Reads in and processes the image data from a directory. Returns the image deltas along with the ground truth
    This function is to be used to grab the color/edge/texture data for the eight test set 1 images.
    They are scaled and flattened, then returned as a list where each element is just one of the images.

    :preprocess: A function used to preprocess the images. It will be the first thing done to all the images. It
        is expected to be a function that operates on a SINGLE image, both color and grayscale.
    """
    #This string acts as an accumulator for a description of what is done to the image
    description = f"columns:{cols}"

    #Add what features were used to the description
    description += " | Features used:"
    if features & COLOR > 0: # color used
        description += 'color'
    if features & EDGE > 0: # edge used
        description += ', edge'
    if features & TEXTURE > 0: # texture used
        description += ', texture'


    test_images = read_test_set(directory)  # test data, list of dictionaries

    #lists to hold the various features
    color_deltas = []
    texture_deltas = []
    edge_deltas = []

    #iterate over all images
    for i in range(len(test_images)):
        #grab the test images
        observed = test_images[i]['observed']
        expected = test_images[i]['expected']

        #apply preprocessing to the image
        if preprocess is not None:
            observed = preprocess(observed)
            expected = preprocess(expected)

        #auto rows option
        if auto_rows or cols < 1:
            # determine number of rows from aspect ratio
            rows = int(cols * (observed.shape[0] / observed.shape[1]))  # rows = cols * height/width


        #Color
        if features & COLOR > 0:
            # downsample observed and expected for comparison
            observed_downsampled = cv2.resize(observed, (cols, rows), interpolation=cv2.INTER_AREA)
            expected_downsampled = cv2.resize(expected, (cols, rows), interpolation=cv2.INTER_AREA)

            # take difference between the two images
            difference = np.absolute(observed_downsampled.astype(int) - expected_downsampled.astype(int))
            color_deltas.append(difference)


        #texture
        if features & TEXTURE > 0:
            # to avoid the bug, I have to split textures into separate channels, before down-sampling
            single_channels_observed = []
            single_channels_expected = []
            for channel in range(observed.shape[2]):
                # downsample single channel from the observed image
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
        if features & EDGE > 0:
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

    # Create matrix of pixel locations if options is selected
    if include_location:
        location_features = np.zeros((rows, cols, 2), dtype=np.uint8)  # empty matrix of appropriate dimensions
        for r in range(rows):
            for c in range(cols):
                location_features[r, c] = [r, c]

    # Concatenate features
    final_data = []
    for i in range(len(test_images)):
        accumulate_features = []
        if features & COLOR > 0:
            accumulate_features.append(color_deltas[i])
        if features & TEXTURE > 0:
            accumulate_features.append(texture_deltas[i])
        if features & EDGE > 0:
            accumulate_features.append(edge_deltas[i])

        # add information about each pixels location if option is selected.
        if include_location:
            accumulate_features.append(np.copy(location_features))



        #process the data
        combined_image = np.concatenate(tuple(accumulate_features), -1) # concatenate
        combined_image = combined_image.reshape(-1, combined_image.shape[2])  # flatten
        min_max_scaler = preprocessing.MinMaxScaler() # scale the data
        data = min_max_scaler.fit_transform(combined_image) # apply the scaling

        #now grab the ground truth
        gt = test_images[i]['obstructionmask']
        #gt = gt.reshape(-1, 1 if len(gt.shape) == 2 else gt.shape[2])

        #reshape to rectangular based on option selected.
        # Note: Scaling must be done on flattened image, so reshaping comes after.
        if shape == RECT:
            data = data.reshape(rows, cols, -1)
        elif shape == FLAT:
            pass  # flattened image, already done
        else:
            assert False, "Passed an invalid shape argument"

        # append the data and the ground truth to a list
        final_data.append({'X': data, 'obstructionmask': gt, 'rows': rows, 'cols': cols})

    #return the final data for processing
    return final_data






def show_images(images, descriptions=None, title=None):
    """
    Displays a bunch of images at once along with their descriptions
    :param images: list of images
    :param descriptions: corresponding list if image descriptions
    """
    plt.rcParams['figure.dpi'] = 400

    if descriptions is not None:
        assert len(images) == len(descriptions), "The lists of images and descriptions must have the same length."

    num_items = len(images)

    if descriptions is None:
        fig, axs = plt.subplots(num_items, 1, figsize=(5, num_items * 2))
        if num_items == 1:
            axs = [axs]  # Make it iterable
        for idx, (image,) in enumerate(zip(images)):
            # Load and display the image
            axs[idx].imshow(image, cmap='gray', vmin=image.min(), vmax=image.max())
            axs[idx].axis('off')  # Turn off axis for image


    else:
        fig, axs = plt.subplots(num_items, 2, figsize=(10, num_items * 2))
        if num_items == 1:
            axs = [axs]  # Make it iterable
        for idx, (image, description) in enumerate(zip(images, descriptions)):
            # Load and display the image
            axs[idx][0].imshow(image, cmap='gray', vmin=image.min(), vmax=image.max())
            axs[idx][0].axis('off')  # Turn off axis for image

            # Display the description
            axs[idx][1].text(0.15, 0.5, description, fontsize=14, transform=axs[idx][1].transAxes, ha='left',
                             va='center', wrap=True)
            axs[idx][1].axis('off')  # Turn off axis for text

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    #plt.subplots_adjust(hspace=0.0)
    if title is not None:
        plt.suptitle(title, fontsize=16)
    #plt.savefig(f'{title}.png')
    plt.show()



def MinNeighborFilter(image1, image2, d):
    """
    This function takes the difference between two images, but each the difference is taken is the minimum in a
    window of the image being compared to. The motivation is to be able to account for poor registration of the images.
    :param image1: Candidate image.
    :param image2: Image being compared to. Window moves through this image.
    :param d: diameter of the moving window.
    :return: a 2D image of the same dimensions as the input with the differences.
    """

    #validation
    assert image1.shape == image2.shape, "Input images must have same shape"
    assert d % 2 == 1, "Filter diameter, d, must be an odd number"
    assert d >= 3, "Filter diameter must be greater than or equal to 3"

    #grab image dimensions
    rows, cols = image1.shape[0], image1.shape[1]

    image1, image2 = image1.astype(np.float64), image2.astype(np.float64)

    #instantiate new image/matrix
    out = np.zeros((rows, cols), dtype=np.float64)

    #iterate over each pixel
    for i in range(rows):
        for j in range(cols):
            m = None # min
            offset = (d-1)//2
            #handle the window
            for r, c in CartesianProduct(list(range(i-offset, i+offset+1)), list(range(j-offset, j+offset+1))):
                #skip pixels out of range
                if r < 0 or c < 0 or r >= rows or c >= cols:
                    continue

                #calculate the euclidean distance
                distance = np.linalg.norm(image1[i, j] - image2[r, c])

                #set minimum to current if it has not been set yet
                if m is None:
                    m = distance

                #set new minimum distance if found
                elif distance < m:
                    m = distance

            #set the new value
            out[i, j] = m

    #return the distance image
    return out



def filter_image(image, func, d, scale=False):
    """
    :param image: image to filter. Must be grayscale
    :param func: Handler that calculates the candidate pixel
    :param d: diameter of the moving window.
    :param scale: bool, if scale, then the image is scaled to uint8, otherwise left as float64.
    """
    #check proper arguments passed
    assert len(image.shape) == 2, "Image must be grayscale, i.e. two dimensions, nxm."
    assert d > 2 and d % 2 > 0, "Invalid window diameter. Must be odd and at least 3."

    # setup
    rows, cols = image.shape[0], image.shape[1] # image dimensions
    out = np.zeros((rows, cols), dtype=np.float64)  # instantiate output image

    #extend to handle image boundaries
    offset = (d - 1) // 2

    #mirror top and bottom
    top = image[0:offset, 0:cols]
    bottom = image[rows-offset:rows, 0:cols]
    top = np.flip(top, 0)
    bottom = np.flip(bottom, 0)

    #append top and bottom
    extended_image = np.concatenate((top, image, bottom), axis=0)

    #mirror left and right
    left = extended_image[0:rows+(offset*2), 0:offset]
    right = extended_image[0:rows+(offset*2), cols-offset:cols]
    left = np.flip(left, 1)
    right = np.flip(right, 1)

    #append left and right
    extended_image = np.concatenate((left, extended_image, right), axis=1)

    #move the window over the extended image
    for r in range(rows):
        for c in range(cols):
            window = extended_image[r:(r+d), c:(c+d)]

            #evaluate window and assign resulting value to output image.
            value = func(window)
            out[r, c] = value

    #return final output image
    return out



def ClusterMatch(image, target, n_clusters):
    #cluster on the target
    image_hsv, target_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV), cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    image_hue, target_hue = image_hsv[:, :, 0].reshape(-1, 1), target_hsv[:, :, 0].reshape(-1, 1)
    image_shape, target_shape = image.shape[:-1], target.shape[:-1]   # original shapes

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(target_hue)
    #centers = kmeans.cluster_centers_
    predicted_labels = kmeans.predict(image_hue)


def constrained_bf_matcher(descriptors1, keypoints1, descriptors2, keypoints2, constraint, threshold, norm_type=cv2.NORM_L2, lowe_ratio=0.70):
    """
    A brute force matcher with constraints.

    :param descriptors1:
    :param keypoints1:
    :param descriptors2:
    :param keypoints2:
    :param constraint:
    :param threshold:
    :param norm_type:
    :return:
    """

    matches = []
    for idx1, desc1 in enumerate(descriptors1):
        best_match = None
        second_best_match = None
        best_distance = float('inf')
        second_best_distance = float('inf')
        kp1 = keypoints1[idx1]

        for idx2, desc2 in enumerate(descriptors2):
            kp2 = keypoints2[idx2]
            if constraint(kp1, kp2, threshold):  # Check the constraint
                distance = cv2.norm(desc1, desc2, norm_type)
                if distance < best_distance:
                    second_best_distance = best_distance
                    second_best_match = best_match
                    best_distance = distance
                    best_match = cv2.DMatch(_queryIdx=idx1, _trainIdx=idx2, _distance=distance)
                elif distance < second_best_distance:
                    second_best_distance = distance
                    second_best_match = cv2.DMatch(_queryIdx=idx1, _trainIdx=idx2, _distance=distance)

        # Apply Lowe's ratio test
        if best_match and second_best_match and best_distance < lowe_ratio * second_best_distance:
            matches.append(best_match)

    #return matches
    return matches






def warp_image():
    image = cv2.imread("/Users/aidan/Desktop/obs2.png", cv2.IMREAD_UNCHANGED)
    print(image.shape)
    H = np.array([
        [1.0221, -0.1326, 16.0],
        [0.0546, 1.0876, -150.6667],
        [0.0, -0.0001, 1.0]
    ])
    # Get the image dimensions
    height, width = image.shape[:2]

    # Apply the homography transformation
    transformed_image = cv2.warpPerspective(image, H, (width, height))


    # Save the transformed image
    cv2.imwrite("/Users/aidan/Desktop/obs-aligned.png", transformed_image)



def entropy(image):
    """
    Helper routine for finding the entropy of a grayscale image.
    Uses the definition of entropy defined by Gonzales and Woods.
    """
    pixel_counts = np.bincount(image.flatten(), minlength=256) #flatten
    probabilities = pixel_counts / np.sum(pixel_counts) #intensity probability
    probabilities = probabilities[probabilities > 0] #remove the zeros
    entropy = -np.sum(probabilities * np.log2(probabilities)) #entropy calculation
    return entropy




def SplitImage(image):
    """given a bgr or rgb image, split it into the three separate channels"""
    assert len(image.shape) == 3, "Must be a multi-channel image"
    assert image.shape[2] == 3, "Must have 3 color channels"
    return image[:, :, 0], image[:, :, 1], image[:, :, 2]




def GammaCorrection(image, gamma):
    return PowerLawTransform(image, gamma)

def PowerLawTransform(image, gamma):
    gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype='uint8')
    return gamma_corrected


def AffineCorrespondenceRegistration(fixed, moving):
    """
    Performs a correspondence based affine registration
    """
    #grab sift features


def GetGaussianKernel(n, sigma, normalize=True):
    """
    Returns a 2D gaussian kernel of shape n x n.
    If normalized, the elements of the gaussian kernel will be between 0 and 1, and
    the sum of all elements will add up to 1.
    """
    #Check that n is odd and gte to 3.
    assert n >= 3, "n must be greater than or equal to 3."
    assert n % 2 == 1, "n must be and odd number."

    #grab gaussian kernel
    kernel = cv2.getGaussianKernel(n, sigma)

    #un-normalize it.
    kernel /= kernel[0, 0]

    #multiply by transpose to make it 2D
    kernel = kernel @ kernel.T

    #make normal
    if normalize:
        kernel /= kernel.sum()

    #return the resulting gaussian kernel
    return kernel

def chatgpt_get_gaussian_kernel(n, sigma):
    """This appears to work properly"""
    # Compute the range of x values (centered)
    x = np.arange(n) - (n - 1) / 2

    # Apply the Gaussian formula
    kernel = np.exp(-0.5 * (x / sigma) ** 2)

    # Normalize the kernel so the sum is 1
    kernel /= kernel.sum()

    # Convert to a column vector
    return kernel.reshape(-1, 1)


def gaussian_derivative(x, sigma):
    """
    Returns the derivative of a 1d gaussian function at x, where the mean is 0, and the standard deviation is sigma.
    Verify results with wolfram...

    Calculating the derivative of the gaussian function:
        https://www.wolframalpha.com/input?i2d=true&i=Divide%5Bd%2Cdx%5D+Divide%5B1%2C%CF%83Sqrt%5B2%CF%80%5D%5D+*+exp%5C%2840%29%5C%2840%29-Divide%5B1%2C2%5D+%5C%2841%29*+Power%5B%5C%2840%29Divide%5Bx%2C%CF%83%5D%5C%2841%29%2C2%5D%5C%2841%29

    Evaluate derivative of gaussian at some point:
        https://www.wolframalpha.com/input?i=-%28e%5E%28-x%5E2%2F%282+%CF%83%5E2%29%29+x%29%2F%28sqrt%282+%CF%80%29+%CF%83%5E3%29%3B+x%3D3%2C+%CF%83%3D2
    """
    result = ((-1 * x) / (sigma**3 * np.sqrt(2*np.pi))) * np.exp(-0.5 * (x/sigma)**2)
    return result

def GaussianKernelDerivative(n, sigma, normalize=True, dimensions=1):
    #range of x values
    x = np.arange(n) - (n - 1) / 2

    #evaluate in
    kernel = gaussian_derivative(x, sigma)

    # Normalize the kernel so the sum is 1
    if normalize:
        kernel /= kernel.sum()

    #convert to column vector
    kernel = kernel.reshape(-1, 1)

    #make into 2d kernel
    if dimensions == 2:
        kernel = kernel @ kernel.T

    return kernel


def colorize_2d_matrix(m):
    """
    NOTE - THIS IS NOT DONE. DO NOT USE.
    """
    max_val = np.amax(m)
    min_val = np.amin(m)
    r = max_val - min_val#range the data spans
    tol = 0.001 * r # tolerance is 0.1% the range of the data
    out = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8) # output image
    red_hi = np.array([0, 0, 255])
    red_lo = np.array([0, 0, 255])

    for row in range(m.shape[0]):
        for col in range(m.shape[1]):
            value = m[row][col]
            if math.isclose(value, 0, rel_tol=tol): # close to zero -> make white
                color = (255, 255, 255)
            elif value > 0.0: # green
                p = value



def ExecutionTime(func):
    import time
    start = time.time()
    func()
    end = time.time()
    total = end - start
    return total


def draw_sift_matches(image1, image2, n, lowe_ratio=0.70, print_point_locations=False):
    #convert to grayscale. OpenCV SIFT feature detection only works with grayscale images.
    image1_gray, image2_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    #Find sift features
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1_gray, None)
    kp2, des2 = sift.detectAndCompute(image2_gray, None)

    #match keypoints
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    #apply lowe's ratio test
    good_matches = []
    for m1, m2 in matches:
        if m1.distance < (lowe_ratio * m2.distance):
            good_matches.append(m1)

    #sort matches by distance
    good_matches = sorted(good_matches, key=lambda match: match.distance)

    #error checking where not enough good matches were found
    if len(good_matches) < n:
        print(f"WARNING: Not enough sift matches found. Reducing n to match number of matches, {len(good_matches)}")
        print("\tYou can also try reducing the lowe ratio to allow for less strict matching.")
        n = len(good_matches)

    #draw each match on the image
    if print_point_locations:
        print(f"MATCH N:   IMAGE1\t\t\t\t\tIMAGE2")
    for i in range(n):
        #parse out matching point locations
        match = good_matches[i]
        point1, point2 = kp1[match.queryIdx].pt, kp2[match.trainIdx].pt

        # round points to nearest pixel, make sure they are integers
        point1_i = int(point1[0] + 0.5), int(point1[1] + 0.5)
        point2_i = int(point2[0] + 0.5), int(point2[1] + 0.5)

        #draw points and labels
        cv2.circle(image1, point1_i, 4, (0, 0, 255), -1)
        cv2.circle(image2, point2_i, 4, (0, 0, 255), -1)
        cv2.putText(image1, str(i+1), (point1_i[0] + 10, point1_i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image2, str(i+1), (point2_i[0] + 10, point2_i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if print_point_locations:
            #print(f"{i}: {point1}\t{point2}")
            # tup[0]:10.2f
            print(f"{i+1:6}: ({point1[0]:8.2f},{point1[1]:8.2f})    ({point2[0]:8.2f},{point2[1]:8.2f})")

    cv2.imshow('image1', image1)
    cv2.imshow('image2', image2)
    cv2.waitKey()





def sliding_window_stride_view(img, window_size):
    """
    --Helper for SlidingWindow--
    Reshapes the image using numpy's stride feature to create a set of windows.
    """
    # Assumes img is a NumPy array
    if len(img.shape) == 2:
        stride_h, stride_w = img.strides
        h, w = img.shape
        window_h, window_w = window_size
        out_shape = (h - window_h + 1, w - window_w + 1, window_h, window_w)
        strides = (stride_h, stride_w, stride_h, stride_w)
    elif len(img.shape) == 3:
        stride_h, stride_w, stride_c = img.strides
        h, w, c = img.shape
        window_h, window_w = window_size
        out_shape = (h - window_h + 1, w - window_w + 1, window_h, window_w, c)
        strides = (stride_h, stride_w, stride_h, stride_w, stride_c)
    else:
        raise ValueError(f"image has to have 2 or 3 dimensions, this one has {len(img.shape)}.")

    # Compute start indices of window centers
    center_row_start = window_h // 2
    center_col_start = window_w // 2
    return (
        np.lib.stride_tricks.as_strided(img, shape=out_shape, strides=strides),
        center_row_start,
        center_col_start
    )

def sliding_window_generator(image, window_size):
    """
    --Helper for SlidingWindow--
    A generator for SlidingWindow for iterating over the windows
    """
    windows, row_start, col_start = sliding_window_stride_view(image, window_size)
    for row_idx, window_row in enumerate(windows):
        for col_idx, window in enumerate(window_row):
            row = row_idx + row_start
            col = col_idx + col_start
            yield window, row, col

def SlidingWindow(image, shape, window_handler, padding_mode='reflect'):
    """
    - Shape must be 2 dimensions. It is (rows, cols).
    - Default padding mode is 'reflect'.
    """
    is_color = len(image.shape) == 3

    #check that the window shape has odd dimensions
    h, w = shape
    pad_y = h//2
    pad_x = w//2
    is_odd = (w%2 == 1) and (h%2 == 1)
    is_big_enough = (w >= 3) and (h >= 3)
    if not is_odd:
        raise ValueError("Window width and height must be odd.")
    elif not is_big_enough:
        raise ValueError("Window width and height must be >=3")

    #check for valid padding mode
    possible_padding_modes = ['linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect', 'edge', 'symmetric', 'wrap']
    if padding_mode not in possible_padding_modes:
        raise ValueError(f"Invalid padding mode, {padding_mode}. Options are {possible_padding_modes}.")


    #apply padding to input image
    if len(image.shape) == 2:
        pad_width = ((h // 2, h // 2), (w // 2, w // 2))
    elif len(image.shape) == 3:
        pad_width = ((h // 2, h // 2), (w // 2, w // 2), (0, 0))
    else:
        raise ValueError('Image shape must be 2 or 3 dimensions')

    #pad_width = ((h//2, h//2), (w//2, w//2))  # ((top, bottom), (left, right))
    image = np.pad(image, pad_width=pad_width, mode=padding_mode)
    print(pad_width)
    #Allocate output image
    out = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

    #Iterate over window locations and invoke window handler
    for window, row, col in sliding_window_generator(image, shape):
        #print(window_handler(window))
        try:
            out[row, col] = window_handler(window)
        except TypeError:
            print(window_handler(window))
        except ValueError:
            print(window_handler(window))

    #remove the extraneous padding
    out = out[pad_y: -1 * pad_y, pad_x:-1 * pad_x]
    if is_color:
        #out = out[pad_y: -1*pad_y, pad_x:-1*pad_x, :]
        pass
    else:
        pass
        #out = out[pad_y: -1*pad_y, pad_x:-1*pad_x]
    return out




def UnsharpMask(image, k, sigma=2.0, alpha=2.0, beta=-1.0, gamma=0):
    """
    k is the size of the window for the gaussian blur
    The strength of the effect can be altered by adjusting the alpha and beta weightings, for example: 1.5 and -0.5.
    """
    gblur = cv2.GaussianBlur(image, (k, k), 2.0)
    unsharp_image = cv2.addWeighted(image, 2.0, gblur, -1.0, 0)
    return unsharp_image



############################################################################################################
# TESTS
############################################################################################################



def test_filter_and_entropy():

    def mean_blur(array):
        return np.sum(array)/np.size(array)

    image = cv2.imread("Data/TestingDiffDiff/test1/observed.png", cv2.IMREAD_GRAYSCALE)

    cv2.imshow("original", image)

    smoothed = filter_image(image, entropy, 55).astype(np.uint8)
    smoothed = smoothed * 5

    cv2.imshow("out", smoothed)
    cv2.waitKey()



def test_entropy():
    # data = generate_data(10, 200, "Data/TestSet1ProperlyFormatted", auto_rows=True, features=COLOR|EDGE|TEXTURE)
    # X1 = data[0]['X']
    A = np.zeros((6, 6), dtype=np.uint8)
    i = 1
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            A[r, c] = i
            i = i + 1

    # image_filter(A, None, d=5)
    test_filter_and_entropy()


def test_histogram_matching():
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    np.set_printoptions(suppress=True)

    observed = cv2.imread("/Users/aidan/PycharmProjects/DYSCO/Data/TestingDiffDiff/test1/unobstructed-aligned.png")
    observed = cv2.medianBlur(observed, 3)
    expected = cv2.imread("/Users/aidan/PycharmProjects/DYSCO/Data/TestingDiffDiff/test1/expected.png")
    cv2.imshow("observed", observed)
    cv2.imshow("expected", expected)

    #hsv histogram
    hsv = cv2.cvtColor(expected, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    shape = hue.shape
    hue = hue.reshape(-1, 1)


    #just one test
    kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(hue)
    centers = kmeans.cluster_centers_
    predicted_labels = kmeans.predict(hue)
    quantized = np.choose(predicted_labels, centers)
    quantized = quantized.reshape(shape)
    hsv[:, :, 0] = quantized
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.uint8)
    cv2.imshow("expected quantized", result)

    #Now fit the other image
    hsv_o = cv2.cvtColor(observed, cv2.COLOR_BGR2HSV)
    hue_o = hsv_o[:, :, 0]
    shape_o = hue_o.shape
    hue_o = hue_o.reshape(-1, 1)

    predicted_labels_o = kmeans.predict(hue_o)
    quantized_o = np.choose(predicted_labels_o, centers)
    quantized_o = quantized_o.reshape(shape_o)
    hsv_o[:, :, 0] = quantized_o
    result_o = cv2.cvtColor(hsv_o, cv2.COLOR_HSV2BGR).astype(np.uint8)
    cv2.imshow("matched", result_o)


    cv2.waitKey()

    """
    for n in range(2, 10):
        kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto").fit(hue)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        print(centers)
        x = np.array([0, 20, 50, 70, 90, 120, 150, 200]).reshape(-1, 1)


        out = kmeans.predict(x)
        print("out:", out)
        quantized = np.choose(out, centers)
        print("quantized:", quantized)
        score = davies_bouldin_score(hue, labels)
        #score = silhouette_score(hue, labels)
        print(f"Cluster {n}: {score}")





    hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    plt.plot(hist, color='g')
    plt.title('Hue channel histogram')
    plt.yscale("log")
    plt.show()


    #observed = cv2.GaussianBlur(observed, (3, 3), 0)
    observed = cv2.medianBlur(observed, 3)
    cv2.imshow("observed", observed)
    cv2.imshow("expected", expected)
    matched = MatchHistograms(observed, expected)
    cv2.imshow("matched", matched)
    #cv2.waitKey()
    """
def directly_on_rgb():
    n_clusters = 5
    observed = cv2.imread("/Users/aidan/PycharmProjects/DYSCO/Data/TestingDiffDiff/test2/unobstructed-aligned.png")
    #observed = cv2.medianBlur(observed, 3)
    expected = cv2.imread("/Users/aidan/PycharmProjects/DYSCO/Data/TestingDiffDiff/test2/expected.png")
    cv2.imshow("observed", observed)
    cv2.imshow("expected", expected)
    original_shape = observed.shape

    expected = expected.reshape(-1, 3)
    observed = observed.reshape(-1, 3)

    # just one test
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(expected)
    centers = kmeans.cluster_centers_
    centers = [centers[i] for i in range(0, n_clusters)]
    predicted_labels = kmeans.predict(observed).reshape(-1, 1)
    print(centers)
    print(predicted_labels.shape)
    quantized = np.choose(predicted_labels, centers)
    result = quantized.reshape(original_shape).astype(np.uint8)
    cv2.imshow("expected quantized", result)


    cv2.waitKey()


def test_execution_time_of_e():
    import random

    #generate samples to test
    n_samples = 50
    samples = [random.uniform(-1.0, 2.0) for _ in range(n_samples)]

    #functions to test
    def e_with_math():
        for x in samples:
            math.exp(x)

    def e_with_numpy():
        for x in samples:
            np.exp(x)

    #test the execution time

    t1 = ExecutionTime(e_with_numpy)
    t2 = ExecutionTime(e_with_math)
    print('numpy time:', t1)
    print('math time:', t2)
    if t1 < t2: #numpy faster
        print(f"Numpy execution time was {t2/t1} times faster.")
    else: #math faster
        print(f"Math execution time was {t1/t2} times faster.")




def test_gaussian_filtering_1d():
    """
    test one dimensional kernels against plot in this link...
       https://hannibunny.github.io/orbook/preprocessing/04gaussianDerivatives.html
    """
    #Disable scientific notation for numpy
    np.set_printoptions(suppress=True)

    n = 15
    sigma = 1.5
    x = np.arange(n) - (n - 1) / 2
    kernel = chatgpt_get_gaussian_kernel(n, sigma)
    print(kernel)
    print(kernel.flatten())
    derivative_kernel = GaussianKernelDerivative(n, sigma, normalize=False)

    #plot the kernel value
    plt.plot(x, kernel.flatten(), color='red')
    plt.show()

    plt.plot(x, derivative_kernel.flatten(), color='blue')
    plt.show()



def test_gaussian_filtering_2d():
    # now check to make sure the
    # 2d gaussian filter works
    np.set_printoptions(suppress=True)
    n = 15
    sigma = 1.5
    x = np.arange(n) - (n - 1) / 2


    derivative_kernel = GaussianKernelDerivative(n, sigma, normalize=False, dimensions=2)
    print(derivative_kernel)

    #normalize the range of the derivative kernel for visualizing
    image = cv2.normalize(derivative_kernel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('', image)
    cv2.waitKey()




def fill_with_blue():
    """
    Goal is top fix the other image. has some problems
    """
    background_color = np.array([154, 106, 64], dtype=np.uint8)
    image = cv2.imread("Data/PerspectiveTeapot/moving.png", cv2.IMREAD_UNCHANGED)
    out = image[:, :, :3]
    alpha = image[:, :, 3]

    alpha_mask = alpha == 0
    out[alpha_mask] = background_color
    cv2.imshow('filled in', out)
    #cv2.imshow('alpha channel', alpha)
    cv2.waitKey()

def test_drawing_matching_sift_features():
    static = cv2.imread('Data/PerspectiveTeapot/static.png')
    moving = cv2.imread('Data/PerspectiveTeapot/moving2.png')
    draw_sift_matches(static, moving, 30, lowe_ratio=0.50, print_point_locations=True)




def test_sliding_window():
    #setup test
    image = cv2.imread("Data/PerspectiveTeapot/moving2.png")
    gray = cv2.imread("Data/PerspectiveTeapot/moving2.png", cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Original Image', image)
    window_size = (13, 13)

    #call with both a color and grayscale image
    blurred = SlidingWindow(image, window_size, lambda window: window.mean(axis=(0, 1)))
    gray_blurred = SlidingWindow(gray, window_size, lambda window: window.mean())

    print("Check these are all the same...")
    print('original shape:', image.shape)
    print('blurred shape:', blurred.shape)
    print('gray blurred shape:', gray_blurred.shape)

    #display the results
    cv2.imshow('Blurred Image', blurred)
    cv2.imshow('Gray Blurred Image', gray_blurred)
    cv2.waitKey()






def main():
    """for testing things in this script"""
    #test_gaussian_filtering_1d()
    #test_gaussian_filtering_2d()
    #fill_with_blue()
    #test_drawing_matching_sift_features()
    test_sliding_window()





if __name__ == "__main__":
    main()





