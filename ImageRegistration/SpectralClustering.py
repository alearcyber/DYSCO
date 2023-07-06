import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import segmentation, color
from skimage import graph
from sklearn.cluster import spectral_clustering

def first_spectral_clustering():
    image = cv2.imread('images/fail3.jpg', cv2.IMREAD_GRAYSCALE)

    #padding
    desired_size = max(image.shape)
    vertical_pad = (desired_size - image.shape[0]) // 2
    horizontal_pad = (desired_size - image.shape[1]) // 2
    padded_image = np.pad(image, ((vertical_pad, vertical_pad), (horizontal_pad, horizontal_pad)), mode='constant')
    image = padded_image
    print('image shape:', image.shape)



    #spectral clustering
    mask = image.astype(bool)


    labels = spectral_clustering(image, n_clusters=2, eigen_solver="arpack")
    label_im = np.full(mask.shape, -1.0)
    label_im[mask] = labels

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    axs[0].matshow(image)
    axs[1].matshow(label_im)

    plt.show()



def normalized_cut():
    img = cv2.imread("images/SlicTest.png")
    #stuff
    labels1 = segmentation.slic(img, compactness=30, n_segments=3, start_label=1)
    out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)
    plt.imshow(out1)
    plt.show()

if __name__ == '__main__':
    """entry point"""
    normalized_cut()
