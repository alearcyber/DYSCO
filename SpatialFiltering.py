import cv2
import numpy as np

import Color
import Support



def BackgroundSubtraction(image, background, args=None):
    """
    :param image:
    :param background:
    :return:
    """
    #temporary parameters. Not sure if should be function args
    d = 11 #gaussian blur kernel size
    use_otsu = True #should otsu's method for thresholding be used.

    #blur
    bg_blurred = cv2.GaussianBlur(background, (d, d), 0.0)

    #difference the images
    difference = np.absolute(image.astype(np.float32) - bg_blurred.astype(np.float32))

    #grayscale before threshold
    difference_gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    #threshold
    cutoff, threshed = cv2.threshold(difference_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # erode
    kernel = np.ones((7,7), np.uint8)
    eroded = cv2.erode(threshed, kernel)
    dilated = cv2.dilate(eroded, kernel)

    cv2.imshow('',threshed)
    cv2.waitKey()

    cv2.imshow('', eroded)
    cv2.waitKey()

    cv2.imshow('', dilated)
    cv2.waitKey()
    cv2.destroyAllWindows()





def test1():
    #background = cv2.imread("Data/Mar14Tests/0015.jpg")
    background = cv2.imread("TEMPORARY.png")
    image = cv2.imread("Data/Displays/box-black.png")

    print(background.shape)
    print(image.shape)

    #reshape to match
    image, background = Support.MatchShape(image, background)

    #This is new. match the colors first
    image = Color.ColorPaletteMatch(background, image, debug=True)


    BackgroundSubtraction(image, background)


def bilateral_test():
    image = cv2.imread("Data/Mar14Tests/Cropped/0039.jpg")
    out = cv2.bilateralFilter(image, 5, 1.0, 1.0)
    cv2.imwrite("TEMPBILATERAL.png", out)

bilateral_test()

