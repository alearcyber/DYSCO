"""
The idea here is to just take images and to make their histograms match up a little better
"""
import cv2
import numpy as np
from skimage.exposure import match_histograms
import Dysco


x = cv2.imread("data/dash5325.png")
x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
cv2.imwrite("data/newdash5325.png", x)

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







def match_histograms_hsv(image_path1, image_path2):
    # Read the images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Convert images from BGR to HSV
    hsv_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    # Extract the Value channel
    v_channel1 = hsv_image1[:, :, 2]
    v_channel2 = hsv_image2[:, :, 2]

    # Match histograms of the value channels
    matched_v_channel2 = match_histograms(v_channel2.astype('float32'), v_channel1.astype('float32'), multichannel=False)

    # Replace the Value channel in the second image with the matched one
    hsv_image2[:, :, 2] = matched_v_channel2.astype('uint8')

    # Convert back from HSV to BGR
    result_image1 = cv2.cvtColor(hsv_image1, cv2.COLOR_HSV2BGR)
    result_image2 = cv2.cvtColor(hsv_image2, cv2.COLOR_HSV2BGR)

    # Save or display the results
    cv2.imwrite('result_image1.jpg', result_image1)
    cv2.imwrite('result_image2.jpg', result_image2)
    print("Images processed and saved.")


#line_em_up()










