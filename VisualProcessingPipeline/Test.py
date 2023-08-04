from BlurFilter import BlurFilter
import cv2


def test1():
    observed = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/fail3.jpg")
    expected = cv2.imread("/Users/aidanlear/PycharmProjects/CVResearch/ImageRegistration/images/rawdisplay.png")
    f = BlurFilter(expected, observed, BlurFilter.GAUSSIAN, 15)
    f2 = BlurFilter(expected, observed, BlurFilter.MEDIAN, 31)
    cv2.imshow('original', f.o)
    cv2.imshow('gaussian blurred', f.o_filtered)
    cv2.imshow('median blur 5 x 5', f2.o_filtered)
    cv2.waitKey()


if __name__ == '__main__':
    """run tests"""
    test1()
