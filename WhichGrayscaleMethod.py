"""
This script is to determine what method ITK uses to convert color images to grayscale when
Simpleitk.ReadImage() is used. I will compare the grayscale image made by itk to techniques
that I implement, then compare the grayscale images to see which one matches.

Grayscale algorithms tested:
    - mean
    - weighted mean (Y←0.299⋅R+0.587⋅G+0.114⋅B)
    - single channel extraction

One of the techniques listed above is most likely. I will test more if none of them match



Got this on first run of optimizing weighted mean:
    0.06781111 B
    0.72044793 G
    0.21174096 R


THE CORRECT ANSWER IS REC709 with the following weights:
    0.0722 B
    0.7152 G
    0.2126 R


"""
import SimpleITK as sitk
import numpy as np
import cv2
import base64
from Graphics import Images
from scipy.optimize import minimize

def PrintBase64(path):
    """ print out the image in base64 so it can be pasted directly into the python code."""
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    print(encoded_string)

def TestRetrievingEncodedImage():
    """Test that images are returned properly from Graphics.Images"""
    teapot = Images.ShapesTeapot()
    cv2.imshow("", teapot)
    cv2.waitKey()

def SitkGray(path="Data/Teapot/ShapesTeapot.png"):
    """
    Give the image file. Will create a grayscale version of that image using SimpleITK.
    Returns the image as a uint8 numpy array.
    """
    sitk_image = sitk.ReadImage(path, sitk.sitkFloat32)
    numpy_image_gray = sitk.GetArrayViewFromImage(sitk_image)
    out = cv2.normalize(numpy_image_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) #normalize back to numpy array
    return out


def MeanGrayscale(image):
    """ Converts to grayscale using an unweighted mean."""
    gray_image = np.mean(image.astype(np.float64), axis=-1)
    return gray_image.astype(np.uint8)



def WeightedMeanGrayscale(image=None, weights=(0.299, 0.587, 0.114)):
    """
    Converts to grayscale using a weighted mean.
    weights are in (R, G, B), but image should be in bgr format.
    """
    #read in image; instantiate image
    if image is None:
        image = Images.ShapesTeapot()
    elif type(image) is str:
        image = cv2.imread(image)
    else:
        image = Images.ShapesTeapot()

    #convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image




def SingleChannel(image, channel):
    """
    Converts to grayscale by getting a single channel.
    The image is expected to be bgr format
    """
    #handle string input
    if type(channel) is str:
        if channel == 'b' or channel == 'blue':
            channel = 0
        elif channel == 'g' or channel == 'green':
            channel = 1
        elif channel == 'r' or channel == 'red':
            channel = 2
        else:
            assert False, "Invalid channel selected"

    return image[:, :, channel]


def l2_norm_grayscale(image1, image2):
    """ Calculates the l2 norm (euclidean distance) for 2 grayscale images"""
    assert len(image1.shape) == 2 and len(image2.shape) == 2, "Images must be grayscale."
    assert image1.shape == image2.shape, "Images must be the same shape."
    delta = image1.astype(np.float64) - image2.astype(np.float64)
    norm = np.linalg.norm(delta)
    return norm


def test_main_three_algos():
    #read in sitk grayscale image
    sitk_gray = SitkGray()

    #create grayscale images
    weighted_gray = WeightedMeanGrayscale()
    unweighted_gray = MeanGrayscale(Images.ShapesTeapot())
    blue = SingleChannel(Images.ShapesTeapot(), 'b')
    green = SingleChannel(Images.ShapesTeapot(), 'g')
    red = SingleChannel(Images.ShapesTeapot(), 'r')

    #compare
    print("Weighted Mean:", l2_norm_grayscale(sitk_gray, weighted_gray))
    print("Unweighted Mean:", l2_norm_grayscale(sitk_gray, unweighted_gray))
    print("blue:", l2_norm_grayscale(sitk_gray, blue))
    print("green:", l2_norm_grayscale(sitk_gray, green))
    print("red:", l2_norm_grayscale(sitk_gray, red))


    #visualize
    cv2.imshow('sitk', sitk_gray)
    cv2.imshow("weighted mean", weighted_gray)
    cv2.imshow('unweighted mean', unweighted_gray)
    cv2.imshow('blue', blue)
    cv2.imshow('green', green)
    cv2.imshow('red', red)
    cv2.waitKey()






def optimize_weights():
    """
    This script will attempt to optimize the weights of rgb

    THIS ROUTINE USES BGR. X SHOULD BE BGR
    """
    #read in and setup images
    sitk_image = SitkGray().astype(np.float64)
    color_image = Images.ShapesTeapot()
    b = color_image[:, :, 0].astype(np.float64)
    g = color_image[:, :, 1].astype(np.float64)
    r = color_image[:, :, 2].astype(np.float64)

    #initial guess
    x0 = np.array([0.114, 0.587, 0.299])

    def objective(x):
        #apply weights to make grayscale image
        gray = b*x[0] + g*x[1] + r*x[2]

        #compare to sitk image
        return np.linalg.norm(sitk_image - gray)


    def constraint1(x):
        """eq, weights sum to 1"""
        return x[0] + x[1] + x[2] - 1.0

    def constraint2(x):
        return x[0] - 1.0


    #store constraints in dictionary
    con1 = {'type': 'eq', 'fun': constraint1}



    #execute optimization
    result = minimize(objective, x0, method='SLSQP', constraints=[con1])
    print("Optimal solution:", result.x)
    print("Objective value:", result.fun)


    #visualize results
    image_star = (b * result.x[0] + g * result.x[1] + r * result.x[2]).astype(np.uint8)
    sitk_image = SitkGray()
    cv2.imshow('optimum', image_star)
    cv2.imshow('sitk', sitk_image)
    cv2.waitKey()





def main():
    """entry point"""
    #optimize_weights()
    x = (0.0722, 0.7152, 0.2126) # rec 709 standard
    sitk_image = SitkGray()
    color_image = Images.ShapesTeapot().astype(np.float64)
    b = color_image[:, :, 0]
    g = color_image[:, :, 1]
    r = color_image[:, :, 2]
    gray = b * x[0] + g * x[1] + r * x[2]
    gray = gray.astype(np.uint8)
    print(np.linalg.norm(sitk_image - gray))


if __name__ == '__main__':
    main()

