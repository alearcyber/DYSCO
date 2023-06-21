#Handles morphology stuff

import cv2


################################
# Enums
################################

#Basic Kernels
NEIGHBORHOOD_8 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
NEIGHBORHOOD_4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


BINARY = 1
OTSU = 2



#does the morphology thing, expects grayscale image
def morph(image):
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = NEIGHBORHOOD_4
    m = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return m


