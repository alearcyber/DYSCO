import cv2
import numpy as np
from skimage.exposure import match_histograms
import Dysco


def some_experiment():
    expected = cv2.imread("Data/TestSet1ProperlyFormatted/test7/expected.png")
    observed = cv2.imread("Data/TestSet1ProperlyFormatted/test7/observed.png")
    expected = cv2.GaussianBlur(expected, (7, 7), 0)
    observed = cv2.GaussianBlur(observed, (7, 7), 0)

    expected = cv2.cvtColor(expected, cv2.COLOR_BGR2HSV)
    observed = cv2.cvtColor(observed, cv2.COLOR_BGR2HSV)



    hue_diff = abs(expected[:, :, 0] - observed[:, :, 0])
    sat_diff = abs(expected[:, :, 1] - observed[:, :, 1])
    val_diff = abs(expected[:, :, 2] - observed[:, :, 2])

    #cv2.imshow('h', hue_diff)
    #cv2.imshow('s', sat_diff)
    #cv2.imshow('v', val_diff)



    X_mean = abs(observed.astype(np.float64) - expected.astype(np.float64)).mean(2).astype(np.uint8)
    X_norm = np.linalg.norm(observed.astype(np.float64) - expected.astype(np.float64), axis=-1).astype(np.uint8)

    #back to rgb


    cv2.imshow("mean", X_mean)
    cv2.imshow("norm", X_norm)



    cv2.waitKey()
    cv2.destroyAllWindows()



def match_histo():
    n = 1  # 7 is the two dots
    expected = cv2.imread(f"Data/TestSet1ProperlyFormatted/test{n}/expected.png")
    observed = cv2.imread(f"Data/TestSet1ProperlyFormatted/test{n}/observed.png")
    expected = cv2.GaussianBlur(expected, (5, 5), 0)
    observed = cv2.GaussianBlur(observed, (5, 5), 0)



    matched = match_histograms(observed, expected, channel_axis=-1)

    descriptions = []
    images = []

    images.append(observed)
    descriptions.append('observed')

    images.append(expected)
    descriptions.append('expected')


    images.append(matched)
    descriptions.append('matched')

    images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
    Dysco.show_images(images, descriptions)




match_histo()



