"""
This file has to do with preparing and creating test cases.
The test cases will involved various visualizations in combination with various obstructions applied to those
 visualizations in various locations.

Just off the top of my head, I need to have a scheme to save and create
"""



import cv2
import numpy as np


def convert_binary(image_matrix, thresh_val):
    white = 255
    black = 0
    initial_conv = np.where((image_matrix <= thresh_val), image_matrix, white)
    final_conv = np.where((initial_conv > thresh_val), initial_conv, black)
    return final_conv

image = cv2.cvtColor(cv2.imread("images/failure-masks/line.png"), cv2.COLOR_BGR2GRAY)



ret, thresh = cv2.threshold(image, 2, 255, cv2.THRESH_BINARY)

#cv2.imwrite('/Users/aidanlear/Desktop/linethresh.png', thresh)



#mask = cv2.imread("/Users/aidanlear/Desktop/linethresh processes.png", cv2.IMREAD_GRAYSCALE)
mask = cv2.imread("/Users/aidanlear/Desktop/linemask.png", cv2.IMREAD_GRAYSCALE)

mask = (mask / 255).astype('uint8')
print('new thresh dtype:', thresh.dtype)


cv2.imshow('mask', thresh)
test = cv2.resize(cv2.cvtColor(cv2.imread("images/dash.png"), cv2.COLOR_BGR2GRAY), dsize=(thresh.shape[1], thresh.shape[0]))
cv2.imshow('out', thresh * test)

#binarize
out = np.ma.make_mask(mask, shrink=False)
print('dtype of binary mask:', out.dtype)




def contours():
    color = cv2.imread('/Users/aidanlear/Desktop/linethresh.png')
    image = cv2.imread('/Users/aidanlear/Desktop/linethresh.png', cv2.IMREAD_GRAYSCALE)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color, contours[0:50], -1, (255, 255, 0), 3)
    cv2.imshow('contours', color)
    cv2.waitKey()




contours()




