"""
This script will be focused on color clustering.

"""
import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt



def ColorCluster(image, template, n_clusters=None):
    """
    Quantizes the color of an image relative to a template image
    """
    assert image.shape == template.shape, "images have to have the same dimensions in the ColorCluster routine."

    #store original shape
    original_shape = image.shape

    #Flatten the images
    observed = image.reshape(-1, 3)
    expected = template.reshape(-1, 3)

    #cluster the template image
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(expected)
    centers = kmeans.cluster_centers_
    centers = [centers[i] for i in range(0, n_clusters)]
    predicted_labels = kmeans.predict(observed).reshape(-1, 1)
    quantized = np.choose(predicted_labels, centers) #apply predicted color center
    result = quantized.reshape(original_shape).astype(np.uint8) #reshape to original image dimensions.
    return result








###############################################################################################
# Tests
###############################################################################################
def TestColorCluster():
    expected, observed = cv2.imread("Data/TestingDiffDiff/test1/expected.png"), cv2.imread("Data/TestingDiffDiff/test1/unobstructed-aligned.png")
    #observed = cv2.GaussianBlur(observed, (5,5), 0)
    observed = cv2.medianBlur(observed, 3)
    adjusted = ColorCluster(observed, expected, n_clusters=6)
    adjusted_expected = ColorCluster(expected, expected, n_clusters=6)
    cv2.imshow("EXPECTEDADJUSTED-TEMP.png", adjusted_expected)
    cv2.imshow("COLORADJUSTED-TEMP.png", adjusted)
    cv2.waitKey()


def draw_expected_hsv_histogram():
    """
    Converts a screenshot of the display into hsv, then shows the histogram of the hue channel.
    In theory, there should be distinct groupings of hues representing the colors used
    to create the dashboard.
    """
    expected, observed = cv2.imread("Data/TestingDiffDiff/test1/expected.png"), cv2.imread("Data/TestingDiffDiff/test1/unobstructed-aligned.png")

    #no preprocessing
    expected_hsv = cv2.cvtColor(expected, cv2.COLOR_BGR2HSV) #convert to hsv
    hue_channel = expected_hsv[:, :, 0]
    b, bins, patches = plt.hist(hue_channel.flatten(), 255)
    print(np.unique(hue_channel))
    plt.title("Hue channel histogram, no preprocessing")
    plt.show()


    #with median blur after converting to hsv
    expected_hsv = cv2.cvtColor(expected, cv2.COLOR_BGR2HSV)  # convert to hsv
    hue_channel = expected_hsv[:, :, 0]
    hue_channel = cv2.medianBlur(hue_channel, 5)
    b, bins, patches = plt.hist(hue_channel.flatten(), 255)
    print(np.unique(hue_channel))
    unique_values = dict(zip(*np.unique(hue_channel, return_counts=True)))



    #print(np.unique(hue_channel))
    print(unique_values)
    plt.title("Hue channel histogram, median blur")
    plt.show()





def visualize_lab_intensities():
    """
    I want to see the peaks for the color intensities of a screenshot in lab color space
    """
    expected = cv2.imread("Data/TestingDiffDiff/test1/expected.png")
    expected_lab = cv2.cvtColor(expected, cv2.COLOR_BGR2Lab)
    expected_a, expected_b = expected_lab[:, :, 1], expected_lab[:, :, 2]


    print(np.amax(expected_a))
    print(np.amin(expected_a))
    print(np.amax(expected_b))
    print(np.amin(expected_b))
    #unique_values = dict(zip(*np.unique(expected_a, return_counts=True)))

    #log_scale = np.log10(expected)

    # create histogram
    histogram = np.zeros(shape=(256, 256), dtype=int)
    for r in range(expected_lab.shape[0]):
        for c in range(expected_lab.shape[1]):
            a, b = expected_lab[r, c, 1], expected_lab[r, c, 2]
            histogram[a, b] = histogram[a, b] + 1


    log_scale_hist = np.log2(histogram + 1)
    max_marker_size = 100
    min_marker_size = 2
    use_log_scale = True
    for a in range(256):
        for b in range(256):
            if histogram[a, b] > 0:
                if use_log_scale:
                    l = log_scale_hist[a, b]/np.amax(log_scale_hist)
                else:
                    l = histogram[a, b]/np.amax(histogram)
                marker_size = (max_marker_size * l) + (min_marker_size *(1-l))
                plt.scatter(a, b, sizes=[marker_size])

    plt.title("L*a*b* Histogram")
    plt.xlabel("a*")
    plt.ylabel("b*")
    plt.show()










def main():
    #draw_expected_hsv_histogram()
    visualize_lab_intensities()


if __name__ == "__main__":
    main()
