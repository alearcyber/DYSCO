import cv2


def f1():
    capture = cv2.imread("Data/Mar14Tests/0003.jpg")
    original = cv2.imread("Data/Displays/box-black.png")
    cv2.imshow('capture', capture)
    cv2.imshow('original', original)
    cv2.waitKey()


f1()

