"""sub-class of Filter for the purpose of using various blurs"""
import cv2

from Filter import Filter




class BlurFilter(Filter):
    #blur method enums
    MEAN = 1
    MEDIAN = 2
    GAUSSIAN = 3
    BILATERAL = 4


    def __init__(self, expected, observed, method, k_size, title=None, sigma_x=None, sigma_y=None, border_type=cv2.BORDER_DEFAULT):
        self.method = method
        self.k_size = k_size
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.border_type = border_type

        #setup blurring function





        super().__init__(expected, observed, self.mean_blur, title=title)


    #used in the higher order function apply_filter
    def mean_blur(self, img):
        return cv2.blur(src=img, ksize=(self.k_size, self.k_size), borderType=self.border_type)

    def median_blur(self, img):
        return cv2.medianBlur(src=img, ksize=(self.k_size, self.k_size))

    def gaussian_blur(self, img):
        return cv2.GaussianBlur(src=img, ksize=(self.k_size, self.k_size), sigmaX=self.sigma_x, sigmaY=self.sigma_y, borderType=self.border_type)

    def bilateral_filter(self, img):
        #from the docs:
        #Filter size: Large filters (d > 5) are very slow, so it is recommended to use d=5 for real-time applications,
        # and perhaps d=9 for offline applications that need heavy noise filtering.
        #Sigma values: For simplicity, you can set the 2 sigma values to be the same. If they are small (< 10),
        # the filter will not have much effect, whereas if they are large (> 150), they will have a very strong effect, making the image look "cartoonish".
        return cv2.bilateralFilter(img, 5, 75, 75)














