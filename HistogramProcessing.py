"""
The idea here is to just take images and to make their histograms match up a little better
"""
import cv2
import numpy as np
from skimage.exposure import match_histograms
import Dysco


#x = cv2.imread("data/dash5325.png")
#x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
#cv2.imwrite("data/newdash5325.png", x)

def read_test_images():
    template = cv2.imread("Data/NewDashboards/fDeck2.png")
    moving = cv2.imread("data/dash5325.png")
    return moving, template



def line_em_up():
    # read in the images
    moving, template = read_test_images()

    #gauss blur
    moving, template = cv2.GaussianBlur(moving, (5, 5), 0), cv2.GaussianBlur(template, (5, 5), 0)

    #store originals for visualizing
    moving_original = cv2.cvtColor(moving, cv2.COLOR_BGR2RGB)
    template_original = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

    #convert to HSV color space
    moving, template = cv2.cvtColor(moving, cv2.COLOR_BGR2HSV), cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

    #extact value channel
    v_moving, v_template = moving[:, :, 2].astype('float32'), template[:, :, 2].astype('float32')

    #match the moving value channel with the template value channel
    v_moving = match_histograms(v_moving, v_template)

    #replace the old value channel with the new one
    moving[:, :, 2] = v_moving.astype('uint8')

    #convert back to bgr
    moving, template = cv2.cvtColor(moving, cv2.COLOR_HSV2RGB), cv2.cvtColor(template, cv2.COLOR_HSV2RGB)

    #displaythe result
    Dysco.show_images([template_original, moving_original, moving])

    #save the result
    cv2.imwrite("data/dash5325.png", moving)







def match_histograms_hsv(image, reference):
    image1, image2 = image, reference

    # Convert images from BGR to HSV
    hsv_image1, hsv_image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV), cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # Extract the Value channel
    v_channel1, v_channel2 = hsv_image1[:, :, 2], hsv_image2[:, :, 2]

    # Match histograms of the value channels
    #matched_v_channel1 = match_histograms(v_channel1.astype('float32'), v_channel2.astype('float32'), multichannel=False)
    matched_v_channel1 = match_histograms(v_channel1.astype('float32'), v_channel2.astype('float32'))

    # Replace the Value channel in the second image with the matched one
    hsv_image1[:, :, 2] = matched_v_channel1.astype('uint8')

    result = cv2.cvtColor(hsv_image1, cv2.COLOR_HSV2BGR) # Convert back from HSV to BGR
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    return result




def function_here():
    observed, expected = cv2.imread("Data/TestSet1ProperlyFormatted/test2/observed.png"), cv2.imread("Data/TestSet1ProperlyFormatted/test2/observed.png")
    observed = cv2.GaussianBlur(observed, (5, 5), 0)

    #just do the regular grayscale
    gray_unmodified = cv2.cvtColor(observed, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray_unmodified", gray_unmodified)

    #match the hsv part of the histogram
    observed = match_histograms_hsv(observed, expected)

    cv2.imshow('matched up', observed)
    cv2.waitKey()




function_here()



#line_em_up()










