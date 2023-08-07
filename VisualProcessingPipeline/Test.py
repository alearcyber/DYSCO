from BlurFilter import BlurFilter
import cv2
import TextureFilter
import numpy as np


def test1():
    observed = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail3.jpg")
    expected = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/rawdisplay.png")
    f = BlurFilter(expected, observed, BlurFilter.GAUSSIAN, 15)
    f2 = BlurFilter(expected, observed, BlurFilter.MEDIAN, 31)
    cv2.imshow('original', f.o)
    cv2.imshow('gaussian blurred', f.o_filtered)
    cv2.imshow('median blur 5 x 5', f2.o_filtered)
    cv2.waitKey()

def test2():
    observed = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail3.jpg")
    out = TextureFilter.texture_features(observed)

    cv2.imshow('composite', np.mean(out, axis=2).astype(np.uint8))
    cv2.waitKey()
    for i in range(9):
        o = out[:, :, i]
        cv2.imshow('texture' + str(i), o)
    print(out.shape)
    cv2.waitKey()

if __name__ == '__main__':
    """run tests"""
    test2()
