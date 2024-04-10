"""
The purpose of this is to see what information I can get with svd and eigenvalues from images.
"""
import cv2
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt




def main():
    A = cv2.imread("Data/NewDashboardPictures/darktable_exported/20240221_100532.png", cv2.IMREAD_GRAYSCALE)

    """
    #form A^tA
    covariance = np.matmul(np.transpose(A), A)
    file = open("CovarianceMatrix.npy", 'wb')
    np.save(file, covariance)
    file.close()
    
    #load covariance, A^TA
    file = open("CovarianceMatrix.npy", 'rb')
    covariance = np.load(file)
    file.close()
    """


    """
    determine eigenvalues of A^TA
    y = list(eigenvalues)
    x = [i for i in range(1, len(y) + 1)]
    print(len(x))
    print(len(y))
    plt.bar(x, y)
    plt.show()
    """

    #eigenvalues, eigenvectors = la.eig(covariance)
    print("performing svd on image A")
    u, s, v = la.svd(A, full_matrices=False)
    values = list(s)
    x_values = np.arange(len(values))
    plt.scatter(x_values, values, alpha=0.6)
    plt.show()

    resolutions = [10, 25, 50, 100, 200, 500, 1000]

    "Data/TestSet1/test1/cam-low-exposure.png"

    cv2.imshow("Original Image", A)
    for res in resolutions:
        leading_eigenvalues = [values[i] if i < res else 0 for i in range(len(values))]
        s = np.diag(np.array(leading_eigenvalues))
        compact_image = np.matmul(np.matmul(u, s), v)
        print(compact_image.shape)
        print(compact_image.dtype)
        print(np.amax(compact_image))
        print(np.amin(compact_image))

        #scale the compact image
        min_val = compact_image.min()
        max_val = compact_image.max()
        compact_scaled = (compact_image - min_val) / (max_val - min_val) * 255
        compact_scaled = np.uint8(compact_scaled)


        #show it
        cv2.imshow("Compact Image", compact_scaled)
        cv2.waitKey()











main()