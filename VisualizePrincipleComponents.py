"""
This file will take the difference data and visualize the principal components.
This script will plot the PCA variance, that is the variance relative to however many principal components.
This will also produce visualizations of the difference data in Principal component space for
a few different combinations of features.


"""

import cv2
import numpy as np
import Dysco
import Verify
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt




def plot_pca_variance(data, n_components=None, title=None):
    """
    Helper Function For Plotting The Variances FOR PCA
    :param data:
    :param n_components:
    :return:
    """
    # calculate principle components
    pc = PCA(n_components=n_components)
    pc.fit(data)  # try with SCALED data instead

    # plot explained variance
    plt.bar(range(1, pc.n_components_ + 1), pc.explained_variance_ratio_, align='center', label='Explained Variance')

    # also plot cumulative variance
    cumulative_variance = []
    total = 0
    for i in range(pc.n_components_):
        total += pc.explained_variance_ratio_[i]
        cumulative_variance.append(total)
    plt.step(range(1, pc.n_components_ + 1), cumulative_variance, where='mid', label='Cumulative Explained Variance',
             color='red')

    # clean up and display the plot
    plt.xticks(range(1, pc.n_components_ + 1), range(1, pc.n_components_ + 1))
    for i in range(pc.n_components_):
        text_label = str(round(100 * pc.explained_variance_ratio_[i], 2)) + '%'
        plt.text(i + 1, pc.explained_variance_ratio_[i], text_label, ha='center')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('\'K\'-th Principle Component')
    plt.legend(loc='center right')
    if title is not None:
        plt.title(title)
    plt.show()



def visualize_pca_variance():

    #retreive/calculate data
    data = Dysco.generate_data(10, 200, "Data/TestSet1ProperlyFormatted", auto_rows=True,
                               features=Dysco.COLOR | Dysco.EDGE | Dysco.TEXTURE, shape=Dysco.FLAT)

    #retreive sample to work with
    X = data[0]['X']
    plot_pca_variance(X, X.shape[-1], "Color, edge, and texture, 200 columns")


    # Test 2
    data = Dysco.generate_data(10, 200, "Data/TestSet1ProperlyFormatted", auto_rows=True,
                               features=Dysco.COLOR | Dysco.EDGE, shape=Dysco.FLAT)
    X = data[0]['X']
    plot_pca_variance(X, X.shape[-1], "Color and edge (no texture), 200 columns")




    # Test 3
    data = Dysco.generate_data(0, 300, "Data/TestSet1ProperlyFormatted", auto_rows=True,
                               features=Dysco.COLOR | Dysco.EDGE, shape=Dysco.FLAT)
    X = data[0]['X']
    plot_pca_variance(X, X.shape[-1], "Color and edge (no texture), 300 columns")



    # Test 4
    data = Dysco.generate_data(0, 350, "Data/TestSet1ProperlyFormatted", auto_rows=True,
                               features=Dysco.COLOR | Dysco.EDGE | Dysco.TEXTURE, shape=Dysco.FLAT)
    X = data[0]['X']
    plot_pca_variance(X, X.shape[-1], "Color, edge, and texture, 350 columns")


##################
# helper routine
##################
def visualize_pca_space(columns, features, blur, k=7):
    # retrieve/calculate data
    data = Dysco.generate_data(0, columns, "Data/TestSet1ProperlyFormatted", auto_rows=True,
                               features=features, shape=Dysco.FLAT)

    # retrieve sample to work with
    X = data[0]['X']
    om = data[0]['obstructionmask']
    rows = data[0]['rows']
    cols = data[0]['cols']

    #median blur?
    if blur:
        X = (X - X.min()) / (X.max() - X.min())
        X = (X * 255).astype(np.uint8)
        X = X.reshape((rows, cols, -1))
        channels = [cv2.medianBlur(X[:, :, i], k) for i in range(X.shape[-1])] # median
        #channels = [cv2.GaussianBlur(X[:, :, i], (k, k), 0) for i in range(X.shape[-1])]  # gaussian
        X = np.stack(channels, -1)
        X = X.reshape(-1, X.shape[2]).astype(np.float64)


    #create ground truth from obstruction mask
    gt = Verify.create_gt_array(om, rows, cols)
    gt = list(np.array(gt).flatten())

    #move into PCA space
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    # create title string
    title = f"{columns} columns"
    if features & Dysco.COLOR > 0:
        title += ", color"
    if features & Dysco.EDGE > 0:
        title += ", edge"
    if features & Dysco.TEXTURE > 0:
        title += ", texture"
    if blur:
        title += f", {k}x{k} median blur on feature diffs"

    # create colors from ground truth
    colors = ['g' if e else 'r' for e in gt]

    # make plot
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=1)
    plt.title(title)
    plt.xlabel('PC-1', fontsize=10)
    plt.ylabel('PC-2', fontsize=10)
    plt.grid()
    plt.show()


#main routine for visualizing the data in pca space
def pca_space_test():
    # Tests without blurring
    visualize_pca_space(columns=200, features=Dysco.COLOR | Dysco.EDGE | Dysco.TEXTURE, blur=False)
    visualize_pca_space(columns=200, features=Dysco.COLOR | Dysco.EDGE, blur=False)
    visualize_pca_space(columns=300, features=Dysco.COLOR | Dysco.EDGE, blur=False)
    visualize_pca_space(columns=350, features=Dysco.COLOR | Dysco.EDGE | Dysco.TEXTURE, blur=False)
    visualize_pca_space(columns=100, features=Dysco.COLOR | Dysco.EDGE, blur=False)
    visualize_pca_space(columns=100, features=Dysco.COLOR | Dysco.EDGE | Dysco.TEXTURE, blur=False)

    #Tests with blurring
    k = 13
    visualize_pca_space(columns=200, features=Dysco.COLOR | Dysco.EDGE, blur=True, k=k)
    visualize_pca_space(columns=300, features=Dysco.COLOR | Dysco.EDGE, blur=True, k=k)
    visualize_pca_space(columns=150, features=Dysco.COLOR | Dysco.EDGE, blur=True, k=k)
    visualize_pca_space(columns=100, features=Dysco.COLOR | Dysco.EDGE, blur=True, k=k)








visualize_pca_variance()
pca_space_test()













