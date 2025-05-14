"""
This is a bunch of routines that provide _support_ all across the board, so to speak.
"""
import cv2
import numpy as np

GROW = 1
SHRINK = 2
AVERAGE = 3
def MatchShape(image1, image2, method=SHRINK):
    """
    Resizes two images so their width and height match.
    Does so by shrinking the bigger to match the smaller. Comparison determined by area, i.e. number of pixels.
    :return:
    """
    if (image1.shape[0] != image2.shape[0]) or (image1.shape[1] != image2.shape[1]): # if there is some difference in width or height
        if image1.shape[0] * image1.shape[1] > image2.shape[0] * image2.shape[1]: # if image1 is bigger
            image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]), interpolation=cv2.INTER_AREA)
        else: #image2 is bigger
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)

    return image1, image2




def ImageInfo(image):
    """
    TODO - Instead of printing, make a better formatted output string.
    Returns a string with information about the image.
    :param image:
    :return:
    """
    print(np.amax(image))
    print(np.amin(image))
    print(image.dtype)

def FlattenToVoxels(image):
    """
    Flatten an image to be a list of pixels.
    The resulting image is n x v, where n is the number of pixels, and v is the vector length of the pixel intensity.
    """
    #just flattens an image so that it is a list of its voxels.
    return image.reshape(-1, image.shape[-1])


def ShowImage(image, title=''):
    cv2.imshow(title, image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def ShowImages(images, titles=None):
    """
    If no titles are passed, images are shown 1 at a time.
    Otherwise, all windows are shown at the same time
    """
    if titles is None:
        for image in images:
            cv2.imshow('', image)
            cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        for image, title in zip(images, titles):
            cv2.imshow(title, image)
        cv2.waitKey()

    cv2.destroyAllWindows()





#need to make something to parse the JSON better.


#EOF
