"""
The purpose of this script is to have the code for grabing features of an image with svd
"""
import cv2
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt



def normalize_pixels(image):
    # scale the compact image
    min_val = image.min()
    max_val = image.max()
    image = (image - min_val) / (max_val - min_val) * 255
    image = np.uint8(image)
    return image


def main():
    #A = cv2.imread("Data/NewDashboardPictures/darktable_exported/20240221_100532.png", cv2.IMREAD_GRAYSCALE)
    A = cv2.imread("Data/TestSet1ProperlyFormatted/test7/observed.png", cv2.IMREAD_GRAYSCALE)



    u, s, v = la.svd(A, full_matrices=False)
    values = list(s)
    x_values = np.arange(len(values))
    plt.scatter(x_values, values, alpha=0.6)
    plt.show()

    resolutions = [10, 25, 50, 100, 200, 500, 1000]


    cv2.imshow("Original Image", A)
    for res in resolutions:
        leading_eigenvalues = [values[i] if i < res else 0 for i in range(len(values))]
        s = np.diag(np.array(leading_eigenvalues))
        compact_image = np.matmul(np.matmul(u, s), v)
        print(compact_image.shape)
        print(compact_image.dtype)
        print(np.amax(compact_image))
        print(np.amin(compact_image))

        # scale the compact image
        min_val = compact_image.min()
        max_val = compact_image.max()
        compact_scaled = (compact_image - min_val) / (max_val - min_val) * 255
        compact_scaled = np.uint8(compact_scaled)

        # show it
        cv2.imshow("Compact Image", compact_scaled)
        cv2.waitKey()


def svd_compaction(image, n_values):
    #convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    #SVD
    u, s, v = la.svd(image, full_matrices=False)
    values = list(s)

    #Reconstruct with only n_values singular values
    leading_eigenvalues = [values[i] if i < n_values else 0 for i in range(len(values))]
    s = np.diag(np.array(leading_eigenvalues))
    compact_image = np.matmul(np.matmul(u, s), v)

    # scale the compact image
    min_val = compact_image.min()
    max_val = compact_image.max()
    compact_scaled = (compact_image - min_val) / (max_val - min_val) * 255
    compact_scaled = np.uint8(compact_scaled)
    return compact_scaled




def test():
    resolutions = [10, 25, 50, 100, 200, 500, 1000]
    image = cv2.imread("Data/TestSet1ProperlyFormatted/test7/observed.png", cv2.IMREAD_GRAYSCALE)
    expected = cv2.imread("Data/TestSet1ProperlyFormatted/test7/expected.png", cv2.IMREAD_GRAYSCALE)
    for res in resolutions:
        out = svd_compaction(image, res)
        out2 = svd_compaction(expected, res)

        difference = normalize_pixels(np.absolute(out.astype(np.float64) - out2.astype(np.float64)))
        difference = cv2.medianBlur(difference, 7)
        difference = cv2.GaussianBlur(difference, (7, 7), 0)

        #difference = cv2.threshold(difference, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)






        cv2.imshow(f"observed (obstructed)", out)
        cv2.imshow(f"expected", out2)
        cv2.imshow("difference", difference)
        cv2.waitKey()



if __name__ == "__main__":
    #main()
    test()


