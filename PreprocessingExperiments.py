"""
The purpose of this file is to visually experiment with different preprocessing methods.

For example, one thing to test would be read in some of the difference images with no preprocessing, display all of
them, then display the difference images with a gaussian blur done beforehand, then compare the results.
"""
import cv2
import Dysco
import numpy as np

K_WIDTH = 1

def gauss_blur3(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

def gauss_blur5(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def gauss_blur7(image):
    return cv2.GaussianBlur(image, (7, 7), 0)

def gauss_blur9(image):
    return cv2.GaussianBlur(image, (9, 9), 0)

def gauss_blur11(image):
    return cv2.GaussianBlur(image, (11, 11), 0)

def gauss_blur15(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

def gauss_blur27(image):
    return cv2.GaussianBlur(image, (27, 27), 0)


def median(k_width):
    global K_WIDTH
    K_WIDTH = k_width
    return median_helper

def median_helper(image):
    return cv2.medianBlur(image, K_WIDTH)

def gauss():
    functions = [None, gauss_blur3, gauss_blur5, gauss_blur7, gauss_blur9, gauss_blur11, gauss_blur15, gauss_blur27]
    titles = ["No Blur", "3x3", "5x5", "7x7", '9x9', '11x11', '15x15', '27x27']
    for func, title in zip(functions, titles):
        data = Dysco.generate_data(0, 400, "Data/TestSet1ProperlyFormatted", auto_rows=True,
                                   features=Dysco.COLOR | Dysco.EDGE, shape=Dysco.RECT, preprocess=func)
        images = [d['X'].mean(2) for d in data]
        Dysco.show_images(images, 'Different Blur')


def gauss_same_sample():
    """
    For a single sample image, does a gaussian blur beforehand at different kernel sizes
    """
    functions = [None, gauss_blur3, gauss_blur5, gauss_blur7, gauss_blur9, gauss_blur11, gauss_blur15, gauss_blur27]
    descriptions = ["No Blur", "3x3", "5x5", "7x7", '9x9', '11x11', '15x15', '27x27']
    images = []
    for func, title in zip(functions, descriptions):
        data = Dysco.generate_data(0, 400, "Data/TestSet1ProperlyFormatted", auto_rows=True,
                                   features=Dysco.COLOR | Dysco.EDGE, shape=Dysco.RECT, preprocess=func)
        test = data[0]
        delta_image = test['X']
        images.append(delta_image.mean(2))
    Dysco.show_images(images, 'Gaussian Blur', descriptions)


def median_same_sample():
    k_sizes = [1, 3, 5, 7, 9, 11]
    descriptions = ["No Blur", "3x3", "5x5", "7x7", '9x9', '11x11']
    images = []
    for k_size in k_sizes:
        data = Dysco.generate_data(0, 200, "Data/TestSet1ProperlyFormatted", auto_rows=True,
                                   features=Dysco.COLOR | Dysco.EDGE, shape=Dysco.RECT, preprocess=median(k_size))
        test = data[0]
        delta_image = test['X']
        images.append(delta_image.mean(2))
    Dysco.show_images(images, 'Median Blur 200 cols', descriptions)



def median_after_gauss():
    """
    7x7 gauss blur pre-processing, median various levels post-processing, otsu threshold
    """
    k_sizes = [1, 3, 5, 7, 9, 11]
    descriptions = [f"{k}x{k}" for k in k_sizes]
    data = Dysco.generate_data(0, 350, "Data/TestSet1ProperlyFormatted", auto_rows=True,
                               features=Dysco.COLOR | Dysco.EDGE, shape=Dysco.RECT, preprocess=gauss_blur7)
    image = data[0]['X'].mean(2) # grab image to work with

    #rescale back to unsigned 8-bit int
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)

    images = [cv2.medianBlur(image, k) for k in k_sizes] # median blur each image with different kernel sizes.
    Dysco.show_images(images, '7x7 gauss blur preprocessing, Median Blur after, 350 cols', descriptions)


def gauss_after(cols):
    """
    just gaussian blur on the final results for a single image sample
    """
    #params to test
    k_sizes = [1, 3, 5, 7, 9, 11]
    descriptions = [f"{k}x{k}" for k in k_sizes]

    #retreive data
    data = Dysco.generate_data(0, cols, "Data/TestSet1ProperlyFormatted", auto_rows=True,
                               features=Dysco.COLOR | Dysco.EDGE, shape=Dysco.RECT)
    image = data[0]['X'].mean(2)  # grab image to work with

    # rescale back to unsigned 8-bit int
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)

    #Do tests
    images = [cv2.GaussianBlur(image, (k, k), 0) for k in k_sizes]

    #show results
    Dysco.show_images(images, 'No Preprocess, gaussian after, 200 cols', descriptions)

    #otsus
    ths = [cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] for image in images]
    Dysco.show_images(ths, f'No Preprocess, gaussian after, otsu\'s last, {cols} cols', descriptions)

    #binary thresh
    ths = [cv2.threshold(image, 125, 255, cv2.THRESH_BINARY)[1] for image in images]
    Dysco.show_images(ths, f'No Preprocess, gaussian after, bin thresh last, {cols} cols', descriptions)

    #adaptive gaussian thresh
    ths = [cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) for image in images]
    Dysco.show_images(ths, f'No Preprocess, gaussian after, bin thresh last, {cols} cols', descriptions)





#gauss_same_sample()
#median_same_sample()
#median_after_gauss()
gauss_after(400)



