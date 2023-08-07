"""
Filter object does the filtering

filter function requirements -> I think I can just take a numpy array as input and spits out a different
"""
import cv2

class Filter:
    def __init__(self, expected, observed, func, title=None):
        #class fields
        self.e = expected  #expected image
        self.o = observed   #observed image
        self.e_filtered = None  #expected image after filtering
        self.o_filtered = None  #observed image after filtering
        self.filter_function = func  #function used to apply the filter
        self.title = title  #string describing what was done


        #setup the object
        self._apply_filter()



    #applies a given filtering function to the image
    def _apply_filter(self):
        self.e_filtered = self.filter_function(self.e)
        self.o_filtered = self.filter_function(self.o)


    #set the title
    def title(self, title):
        self.title = title
        






################################################
# Functions Used for Filtering
################################################

#blur with a 15 x 15 average filter
def average_blur_15(img):
    return cv2.blur(img, (15, 15))






if __name__ == '__main__':

    og = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail3.jpg")

    f = Filter(og, og, average_blur_15, 12)
    out = f.o_filtered

    cv2.imshow('og', og)
    cv2.imshow('blurred', out)
    cv2.waitKey()





