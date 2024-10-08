import cv2

GRAY = 1
COLOR = 2
LAB = 3
HSV = 4
BGR = 5

class Image:
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
