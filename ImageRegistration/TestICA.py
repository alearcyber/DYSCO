import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import cv2
import DataProcessing


#testing basic ica
def basic_ica():
    """
    # Assuming you have the composite image 'composite_image' and the corresponding background image 'background_image'

    # Define the region of interest (ROI) for the overlaid portion
    roi = (100, 200, 300, 400)  # Example: (top-left y, top-left x, bottom-right y, bottom-right x)

    # Extract the overlaid portion and the background from the composite image
    overlaid_portion = composite_image[roi[0]:roi[2], roi[1]:roi[3], :]
    background = background_image[roi[0]:roi[2], roi[1]:roi[3], :]

    # Reshape the overlaid portion and background images into a 2D array
    n_pixels = (roi[2] - roi[0]) * (roi[3] - roi[1])
    overlaid_portion_2d = overlaid_portion.reshape(n_pixels, -1)
    background_2d = background.reshape(n_pixels, -1)

    # Concatenate the overlaid portion and background to create the observed mixed signals
    mixed_signals = np.concatenate((overlaid_portion_2d, background_2d), axis=1)
    """
    image = cv2.imread('images/fail3.jpg')
    shape = image.shape
    print('shape:', shape)
    mixed_signals = image.reshape(shape[0] * shape[1], -1)
    print('mixed_signals shape:', mixed_signals.shape)

    # Apply ICA to separate the independent components
    ica = FastICA(n_components=2)
    independent_components = ica.fit_transform(mixed_signals)

    # Retrieve the separated overlaid portion component
    #separated_overlaid_portion = independent_components[:, 0].reshape(overlaid_portion.shape)
    separated_overlaid_portion = independent_components[:, 0].reshape(image.shape[0], image.shape[1], -1)
    other = independent_components[:, 1].reshape(image.shape[0], image.shape[1], -1)
    # Display the separated overlaid portion component
    plt.imshow(separated_overlaid_portion)
    plt.imshow(other)
    plt.axis('off')
    plt.show()


#mix the expected output into the signal
def combine_expected():

    #import and align the images
    query = cv2.imread('images/fail3.jpg')
    train = cv2.imread("images/dash4.jpg")
    train, query = DataProcessing.align_images(train, query)
    print(f'train shape:{train.shape}, query shape:{query.shape}')

    #combine and reshape
    shape = query.shape
    mixed_signals = np.concatenate((train, query), axis=2) #mix
    mixed_signals = mixed_signals.reshape(shape[0] * shape[1], -1) #reshape to 2d
    print('flattened mixed signals shape:', mixed_signals.shape)


    #ICA
    ica = FastICA(n_components=2, algorithm='deflation', whiten='arbitrary-variance')
    independent_components = ica.fit_transform(mixed_signals)

    #reshape to 3d
    component_one = independent_components[:, 0].reshape(shape[0], shape[1], -1)
    component_two = independent_components[:, 1].reshape(shape[0], shape[1], -1)
    print('component one reshape:', component_one.shape)
    print('component two reshape:', component_two.shape)


    #rescale and statistics
    component_one = DataProcessing.rescale(component_one)
    component_two = DataProcessing.rescale(component_two)
    print('average:', np.average(component_two))

    #visualize
    cv2.imshow('component one', component_one)
    cv2.imshow('component two', component_two)
    cv2.waitKey()




if __name__ == '__main__':
    """entry point"""
    combine_expected()


