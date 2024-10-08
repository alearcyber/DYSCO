import cv2
import Dysco
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider




def ReadImages():
    images = [
        cv2.imread("/Users/aidan/Desktop/SamTvDefault/20240724_152612.jpg"),
        cv2.imread("/Users/aidan/Desktop/SamTvDefault/20240724_152606.jpg"),
        cv2.imread("/Users/aidan/Desktop/SamTvCustom/20240724_154255.jpg"),
        cv2.imread("/Users/aidan/Desktop/SamTvCustom/20240724_154316.jpg"),
        cv2.imread("/Users/aidan/Desktop/SamTvCustom/20240724_154350.jpg"),
    ]
    return images



def test_hist_equalization():
    images = ReadImages()
    for image in images:
        b, g, r = Dysco.SplitImage(image)
        b, g, r = cv2.equalizeHist(b), cv2.equalizeHist(g), cv2.equalizeHist(r)
        equalized = np.stack((b, g, r), axis=-1)
        cv2.imshow("original", image)
        cv2.imshow("equalized", equalized)
        cv2.waitKey()

def show_in_hsv():
    images = ReadImages()
    for image in images:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = Dysco.SplitImage(hsv)
        #cv2.imshow("h", h)
        #cv2.imshow("s", s)
        cv2.imshow("v", v)
        print(np.amax(v))
        print(np.amin(v))
        print(type(v))
        adjusted = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
        cv2.imshow("adjusted", adjusted)
        cv2.waitKey()





def inpainting():
    image = cv2.imread("/Users/aidan/Desktop/SamTvDefault/20240724_152612.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Inpaint the image to remove the bright spots
    inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    cv2.imshow("original", image)
    cv2.imshow("mask", mask)
    cv2.imshow("inpainted", inpainted)
    cv2.waitKey()


def clamp_lightness():
    image = cv2.imread("/Users/aidan/Desktop/SamTvDefault/20240724_152612.jpg")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = Dysco.SplitImage(hsv)
    cv2.imshow("lightness", v)



#gamma correction
def gamma_correction():
    image = cv2.imread("/Users/aidan/Desktop/SamTvDefault/20240724_152612.jpg")
    gamma = Dysco.GammaCorrection(image, 2.5)
    cv2.imshow("original", image)
    cv2.imshow("gamma 2.5", gamma)
    cv2.imshow("gamma 2.75", Dysco.GammaCorrection(image, 2.75))
    cv2.imshow("gamma 3", Dysco.GammaCorrection(image, 3))
    cv2.waitKey()





def main():
    """entry point"""
    #show_in_hsv()
    gamma_correction()


if __name__ == '__main__':
    main()
