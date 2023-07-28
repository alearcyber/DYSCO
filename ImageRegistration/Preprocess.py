"""
deals with preprocessing images
"""
import cv2
import numpy as np
import DataProcessing
from matplotlib import pyplot as plt

from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms



#convert BGR numpy array to grayscale
def convert_to_grayscale(image):
    if len(image.shape) == 2: # already grayscale
        return  image
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



#Local contrast normalization.
#Done by subtracting the local mean from each pixel and dividing by the local standard deviation.
#Will do opencv's implementation of CLAHE, this is a form of histogram equalization.
def local_contrast_norm(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    cl1 = clahe.apply(image)
    return cl1


#Global Centering
def global_centering(image):

    image = image.astype('float32')
    mean = np.mean(image)
    image = image - mean
    print('image shape:', image.shape)
    #DataProcessing.show(image)

    #show image in pyplot without scaling
    extent = [0, image.shape[1], image.shape[0], 0]
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray', extent=extent)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    plt.show()




def match_histogram(image):
    return


