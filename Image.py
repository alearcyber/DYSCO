import cv2
import numpy as np

#Colorspace enum
GRAY = 1
COLOR = 2
LAB = 3
HSV = 4
BGR = 5

#Access Mode enum
XY = 1 # access by Image[x, y]
RC = 2  # access by Image[row, column]


class Image:
    """
    --Attributes--
    colorspace - what color models is being used
    width - number of pixels wide
    height - number of pixels tall

    n_rows - Number of rows. Equivalent to height.
    n_cols - Number of cols.
    """
    def __init__(self, source, colorspace=None):
        #check valid colorspace selected
        assert colorspace in [None, GRAY, COLOR, LAB, HSV, BGR], "Invalid color space chosen."


        #handle class attribute for which color space the image is in
        if colorspace is None:
            self.colorspace = BGR
        elif colorspace == COLOR:
            self.colorspace = BGR
        else:
            self.colorspace = colorspace

        #set option for reading in the image
        color_option = cv2.IMREAD_COLOR
        if colorspace == GRAY:
            color_option = cv2.IMREAD_GRAYSCALE


        #handle path input
        if source is str:
            self.image = cv2.imread(source, color_option)

        #unpack dimension variables
        self.height, self.width, self.depth = self.image.shape

        #set default access mode
        self.access_mode = RC





    def set_access_mode(self, option):
        """
        Sets the access mode
        :param option: What to set the access mode to.
        """
        #check that a valid access mode was passed.
        assert option == XY or option == RC, "Error: Setting invalid access mode."

        #set the access mode
        self.access_mode = option


    def scale_to_uint8(self):
        """
        Changes the image dtype to uint8, and scales the values between 255-0, where the max is 255, and the min is 0.
        """
        # rescale back to unsigned 8-bit int
        _min = self.image.min()
        _image = (self.image - _min) / (self.image.max() - _min)
        self.image = (_image * 255).astype(np.uint8)

    def __add__(self, o):
        """
        TODO - To be implemented...
        Component-wise addition of two images
        """
        return self.a + o.a