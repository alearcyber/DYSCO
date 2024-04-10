"""
The purpose of this script is to test of the functionality in Dysco.py
"""
from Dysco import *



def TestMinNeighborFilter():
    #read images
    observed = cv2.imread("Data/TestSet1ProperlyFormatted/test1/observed.png").astype(np.float64)
    expected = cv2.imread("Data/TestSet1ProperlyFormatted/test1/expected.png").astype(np.float64)

    #images and descriptions to be visualized
    out = []

    #first, just take the pixelwise distance without blurring
    out.append((abs(observed - expected).mean(-1), 'Raw average voxel difference'))

    #with blurring
    observed = cv2.GaussianBlur(observed, (7, 7), 0)
    expected = cv2.GaussianBlur(expected, (7, 7), 0)
    out.append((abs(observed - expected).mean(-1), 'blurred average voxel difference'))

    #try with the min neighbor filter
    out.append((MinNeighborFilter(observed, expected, 5), 'min neighbor d=5'))
    out.append((MinNeighborFilter(observed, expected, 7), 'min neighbor d=7'))
    out.append((MinNeighborFilter(observed, expected, 11), 'min neighbor d=11'))
    out.append((MinNeighborFilter(observed, expected, 15), 'min neighbor d=15'))

    #visualize the results
    show_images([e[0] for e in out], [e[1] for e in out])



TestMinNeighborFilter()
