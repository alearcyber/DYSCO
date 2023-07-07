from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2
import numpy as np
import DataProcessing


#visualing comparing the fourier space of screenshot and picture
def comparing_ft():
    #read images
    screenshot = cv2.imread('images/dash.png', cv2.IMREAD_GRAYSCALE)
    picture = cv2.imread('images/picture-of-display.png', cv2.IMREAD_GRAYSCALE)

    #dft of screenshot
    f = np.fft.fft2(screenshot)
    fshift = np.fft.fftshift(f)
    ft_screenshot = 20 * np.log(np.abs(fshift))
    #ft_screenshot = 20 * np.log(np.abs(f))
    cv2.imshow('fft screenshot', DataProcessing.rescale(ft_screenshot))
    """
    #dft of picture
    f = np.fft.fft2(picture)
    fshift = np.fft.fftshift(f)
    ft_picture = 20 * np.log(np.abs(fshift))
    """

    # inverse dft
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    cv2.imshow('returned from fft',  DataProcessing.rescale(img_back))

    #do it without inversing the shift
    img_back = np.fft.ifft2(fshift)
    img_back = np.abs(img_back)
    cv2.imshow('returned from fft, no shift', DataProcessing.rescale(img_back))


    cv2.waitKey()


def now_with_the_picture():
    # read images
    screenshot = cv2.imread('images/dash.png', cv2.IMREAD_GRAYSCALE)
    picture = cv2.imread('images/picture-of-display.png', cv2.IMREAD_GRAYSCALE)

    # dft of screenshot
    f = np.fft.fft2(picture)
    fshift = np.fft.fftshift(f)
    ft_screenshot = 20 * np.log(np.abs(fshift))
    # ft_screenshot = 20 * np.log(np.abs(f))
    cv2.imshow('fft screenshot', DataProcessing.rescale(ft_screenshot))
    """
    #dft of picture
    f = np.fft.fft2(picture)
    fshift = np.fft.fftshift(f)
    ft_picture = 20 * np.log(np.abs(fshift))
    """

    # inverse dft
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    cv2.imshow('returned from fft', DataProcessing.rescale(img_back))

    # do it without inversing the shift
    img_back = np.fft.ifft2(fshift)
    img_back = np.abs(img_back)
    cv2.imshow('returned from fft, no shift', DataProcessing.rescale(img_back))

    cv2.waitKey()



if __name__ == '__main__':
    """entry point"""
    #comparing_ft()
    now_with_the_picture()
