import cv2
import numpy as np
import Dysco

test_number = 1
OBSERVED = cv2.imread(f"/Users/aidan/Desktop/ExpoSet/observed/1.png")
EXPECTED = cv2.imread(f"/Users/aidan/Desktop/ExpoSet/expected/1.png")

def canny_demonstration():
    observed_gray, expected_gray = cv2.cvtColor(OBSERVED, cv2.COLOR_BGR2GRAY), cv2.cvtColor(EXPECTED, cv2.COLOR_BGR2GRAY) # grayscale
    observed_gray = cv2.GaussianBlur(observed_gray, (5, 5), 0) # blur
    observed_edge, expected_edge = cv2.Canny(observed_gray, 100, 200), cv2.Canny(expected_gray, 100, 200) # Canny edge
    cv2.imshow("observed_edge", observed_edge)
    cv2.imshow("expected_edge", expected_edge)


    diff = abs(observed_edge - expected_edge)
    cv2.imshow("diff", diff)
    fuzzy_diff = Dysco.MinNeighborFilter(observed_edge, expected_edge, 3)
    cv2.imshow('fuzzydiff', fuzzy_diff)
    cv2.waitKey()



def more_involved_tests():
    gray = cv2.cvtColor(OBSERVED, cv2.COLOR_BGR2GRAY)
    expected_gray = cv2.cvtColor(EXPECTED, cv2.COLOR_BGR2GRAY)
    observed_edge = cv2.Canny(gray, 100, 200)
    expected_edge = cv2.Canny(expected_gray, 100, 200)
    #saliency = Dysco.filter_image(edge, Dysco.entropy, 25)
    #scaled = (saliency * (255/np.amax(saliency))).astype(np.uint8)

    cv2.imshow("observed_edge", observed_edge)
    cv2.imshow("expected_edge", expected_edge)
    #cv2.imshow("saliency", scaled)

    #downsample, downsampling by a factor of 5
    down_observed = cv2.resize(observed_edge, (int(OBSERVED.shape[1]/5), int(OBSERVED.shape[0]/5)), interpolation=cv2.INTER_AREA)
    down_expected = cv2.resize(expected_edge, (int(OBSERVED.shape[1]/5), int(OBSERVED.shape[0]/5)), interpolation=cv2.INTER_AREA)
    cv2.imshow("down_observed", down_observed)
    cv2.imshow("down_expected", down_expected)

    #difference
    down_observed, down_expected = cv2.GaussianBlur(down_observed, (11, 11), 0), cv2.GaussianBlur(down_expected, (11, 11), 0)
    diff = abs(down_observed.astype(np.float32) - down_expected.astype(np.float32)).astype(np.uint8)
    cv2.imshow("difference", diff)



    #now compare difference with obstructed
    obstructed = cv2.imread(f"Data/TestingDiffDiff/test{test_number}/observed-aligned.png", cv2.IMREAD_GRAYSCALE)
    obstructed = cv2.Canny(obstructed, 100, 200)
    obstructed = cv2.resize(obstructed, (int(OBSERVED.shape[1] / 5), int(OBSERVED.shape[0] / 5)), interpolation=cv2.INTER_AREA)
    obstructed = cv2.GaussianBlur(obstructed, (11, 11), 0)
    obstructed_diff = abs(obstructed.astype(np.float32) - down_expected.astype(np.float32)).astype(np.uint8)
    cv2.imshow("obstructed_diff", obstructed_diff)


    #otsus method
    _, otsu = cv2.threshold(obstructed_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("otsu", otsu)




    cv2.waitKey()




canny_demonstration()

