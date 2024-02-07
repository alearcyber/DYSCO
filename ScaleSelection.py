"""
THe purpose of this scirpt is to test and visualize the scale selection described in the paper:
    Blobworld: Image segmentation using Expectation-Maximization and its application to image querying
    link: https://www.cse.psu.edu/~rtc12/CSE586/papers/emCarson99blobworld.pdf

"""
import cv2



def gradient_image():
    image = cv2.imread("Data/TestSet1/test1/cam-low-exposure.png", cv2.IMREAD_GRAYSCALE)

    image = cv2.Laplacian(image, cv2.CV_64F)
    print("shape:", image.shape)
    cv2.imshow("Laplacian", image)
    cv2.waitKey()


def sobel_gradient_image():
    img = cv2.imread("Data/TestSet1/test1/cam-low-exposure.png", cv2.IMREAD_GRAYSCALE)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    cv2.imshow("sobelx", sobelx)
    cv2.imshow("sobley", sobely)
    cv2.waitKey()


def gradI_x_gradIT(v1):
    """gradient of I times(outer product) I transpose"""




if __name__ == "__main__":
    sobel_gradient_image()

