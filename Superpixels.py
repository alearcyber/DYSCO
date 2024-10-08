from skimage.segmentation import slic
import cv2
import numpy as np
from ImageSimilarity import *
import matplotlib.pyplot as plt


####################################################################################################################################
# HELPER ROUTINES
####################################################################################################################################

def TestImages():
    images = []
    n = [1, 3, 4, 5]
    for i in n:
        o, e = cv2.imread(f"Data/TestingDiffDiff/test{i}/unobstructed-aligned.png"), cv2.imread(f"Data/TestingDiffDiff/test{i}/expected.png")
        images.append((o, e))
    return images

def ObstructedTestImages():
    images = []
    n = [1, 3, 4, 5]
    for i in n:
        o, e = cv2.imread(f"Data/TestingDiffDiff/test{i}/observed-aligned.png"), cv2.imread(f"Data/TestingDiffDiff/test{i}/expected.png")
        images.append((o, e))
    return images



def ApplySegments(image, segments):
    """
    Gives an image and segment labels of the same shape, creates a new image with the segments applied,
    and the intensity is the mean within a segment.
    """
    # Eels and Escalators
    average_colors = np.zeros((segments.max() + 1, image.shape[2]))  # Initialize an array to hold the average intensity for each segment
    for segment_id in np.unique(segments):  # Loop through each segment
        segment_pixels = image[segments == segment_id]  # Get the pixels belonging to the current segment
        average_colors[segment_id] = segment_pixels.mean(axis=0)  # Calculate the mean color (for RGB)

    # Create a new image and assign the mean color to each segment
    new_image = np.zeros_like(image)
    for segment_id in np.unique(segments):
        new_image[segments == segment_id] = average_colors[segment_id]
    return new_image


def MySlic(image):
    segments = slic(image, slic_zero=True)
    result = ApplySegments(image, segments)
    return result



####################################################################################################################################
# TESTS
####################################################################################################################################
def test_slic():
    """Just see what SLIC looks like on different pictures with various parameters"""
    observed, expected = TestImages()[0]
    segments = slic(observed, n_segments=300, sigma=5)
    result = ApplySegments(observed, segments)
    cv2.imshow("SLIC, 300 segments, sigma=5", result)


    #SLIC0, SLIC without any parameters
    slic0_segments = slic(observed, slic_zero=True)
    slic0_results = ApplySegments(observed, slic0_segments)
    cv2.imshow("Young LaFlame, he in SLICO mode.", slic0_results)



    #see what the expected would look like
    slic_expected = MySlic(expected)
    cv2.imshow("expected", slic_expected)

    cv2.waitKey()


def slic_image_similarity():
    """See the difference between images when SLIC is used"""
    print('----UNOBSTRUCTED COMPARISON----')
    test_images = TestImages()
    for observed, expected in test_images:
        observed, expected = MySlic(observed), MySlic(expected)
        cv2.imshow("observed", observed); cv2.imshow("expected", expected)
        cv2.waitKey()
        msd = MeanSquaredDifference(observed, expected)
        print(msd)

    print('----OBSTRUCTED COMPARISON----')
    obstructed_test_images = ObstructedTestImages()
    for observed, expected in obstructed_test_images:
        observed, expected = MySlic(observed), MySlic(expected)
        cv2.imshow("observed", observed); cv2.imshow("expected", expected)
        cv2.waitKey()
        msd = MeanSquaredDifference(observed, expected)
        print(msd)


def slic_similarity_test_with_different_params():
    """Instead of SLIC0, try messing with the params, and see what that does"""

    #range of values to calculate
    #n_segments = [50, 100, 150, 200, 300, 400, 600, 700, 900, 1000]
    n_segments = [50, 100, 150, 200, 300, 400, 600]
    sigma = [0.8, 1.2, 1.5, 2, 3, 4, 5, 8, 10]

    #store the intensities at each n_segments, sigma pair
    data = np.zeros(shape=(len(n_segments), len(sigma)))

    #iterate over and calculate all the intensities at each n_segments, sigma pair
    observed, expected = TestImages()[0]
    for n in range(len(n_segments)):
        for s in range(len(sigma)):
            print("n_segments:", n_segments[n], "   sigma:", sigma[s])
            observed_segmented = ApplySegments(observed, slic(observed, n_segments=n_segments[n], sigma=sigma[s]))
            expected_segmented = ApplySegments(expected, slic(expected, n_segments=n_segments[n], sigma=sigma[s]))
            msd = MeanSquaredDifference(observed_segmented, expected_segmented)
            data[n, s] = msd

    #plot the points
    m = np.amax(data)
    for n in range(len(n_segments)):
        for s in range(len(sigma)):
            l = data[n, s]/m #lambda
            color = (l, 0, 1-l)
            plt.plot(n, s, color=color)
    plt.show()

    print(data)

def precomputed_values():
     data = np.array([[455.63571553, 405.96785571, 349.74948619, 459.42332038, 396.14182101, 431.85758109, 478.30763618, 443.39234343, 455.91738309],
    [570.45381134, 507.38210164, 429.8208966, 510.54445389, 536.80214052, 678.738234,   486.3349668,  526.03570587, 588.13078666],
    [559.33214349,614.83859791,557.48640057,539.4501895,611.21227465,639.90037834,513.56796735,463.16757974,450.05356675],
    [575.82099658, 626.929502,   595.74217631, 648.55647878, 601.63699278, 618.1576326,  582.88965056, 542.22773531, 409.25330931],
    [580.66204171, 694.1043274, 689.00177516, 674.71761139, 587.9488573,639.66222313,539.75049328,518.17994099,435.14746299],
    [638.25610283, 679.70205935, 673.7480507,  650.82626513, 614.39055436,603.99842281, 560.50249622, 435.30799638, 389.71999439],
    [741.82437939,771.14418348,760.98221328,741.64761396,624.79458433,526.5354238,482.77143305,431.62543402,397.52158336]])

     n_segments = [50, 100, 150, 200, 300, 400, 600]
     sigma = [0.8, 1.2, 1.5, 2, 3, 4, 5, 8, 10]

     # plot the points
     m = np.amax(data)
     for n in range(data.shape[0]):
         for s in range(data.shape[1]):
             print(n, s)
             l = data[n, s] / m  # lambda
             color = (l, 0, 1 - l)
             plt.scatter(n_segments[n], sigma[s], color=color)
     plt.show()


def trying_more_slic_params():








def main():
    """entry point"""
    #slic_similarity_test_with_different_params()
    precomputed_values()


if __name__ == '__main__':
    main()




