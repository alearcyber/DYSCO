"""

"""
import Dysco
import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.filters import sobel


def interpolate_rainbow(t):
    """
    0 is red, 1 is violet
    Interpolates between red and violet along a line segment using barycentric coordinates.

    Parameters:
    - t: The interpolation parameter where 0 <= t <= 1.
         t = 0 corresponds to red, t = 1 corresponds to violet.

    Returns:
    - The interpolated RGB color as a tuple.
    """
    # RGB for red and violet
    red = np.array([255, 0, 0])
    violet = np.array([148, 0, 211])

    # Linear interpolation
    color = (1 - t) * red + t * violet
    return list(color.astype(int))




def show_kmeans():

    #generate the data
    data = Dysco.generate_data(10, 200, "Data/TestSet1ProperlyFormatted", auto_rows=True,
                               features=Dysco.COLOR | Dysco.EDGE | Dysco.TEXTURE)


    #retreive a sample to work with
    X = data[0]['X']
    rows = data[0]['rows']
    cols = data[0]['cols']

    #visualize current step
    for_vis = X.reshape((rows, cols, -1)).mean(2)
    for_vis = (for_vis - for_vis.min()) / (for_vis.max() - for_vis.min())
    for_vis = (for_vis * 255).astype(np.uint8)


    #Custer with kmeans
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X)
    cluster_centers = [np.linalg.norm(kmeans.cluster_centers_[i, :]) for i in range(n_clusters)]
    index_min = np.argmax(cluster_centers)
    prediction_matrix = kmeans.labels_ != index_min  # the seconds centroid is bigger
    pred_for_vis = prediction_matrix.reshape((rows, cols, -1))
    pred_for_vis = pred_for_vis.astype(np.uint8) * 255 # convert true and false to pixel values

    #red tint
    red_tint = np.stack((for_vis,) * 3, axis=-1)
    for r in range(rows):
        for c in range(cols):
            if pred_for_vis[r, c] == 0:
                original_intensity = for_vis[r, c]
                red_tint[r, c] = [0, 0, original_intensity]




    #visualize
    #pred_for_vis = cv2.resize(pred_for_vis, (1000, int((1000 / cols) * rows)), interpolation=cv2.INTER_NEAREST)
    #for_vis = cv2.resize(for_vis, (1000, int((1000 / cols) * rows)), interpolation=cv2.INTER_NEAREST)
    red_tint = cv2.resize(red_tint, (1000, int((1000 / cols) * rows)), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(f"KMeans Prediction ({n_clusters} clusters)", red_tint)
    cv2.waitKey()
    cv2.destroyAllWindows()





def test_dbscan():
    # generate the data
    data = Dysco.generate_data(10, 300, "Data/TestSet1ProperlyFormatted", auto_rows=True,
                               features=Dysco.COLOR | Dysco.EDGE, include_location=False)

    # retreive a sample to work with
    X = data[0]['X']
    rows = data[0]['rows']
    cols = data[0]['cols']

    #rescale X to be between 0 and 255
    X = (X - X.min()) / (X.max() - X.min())
    X = (X * 255).astype(np.uint8)

    print(f"min:{X.min()}, max:{X.max()}")
    print(X.dtype)
    #image histogram
    histogram = cv2.calcHist([X], [0], None, [256], [0, 256])
    plt.plot(histogram)
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

    #nearest Neighbor plot to determine proper epsilon value for DBSCAN
    # Assuming X is your dataset
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.xlabel('Points')
    plt.ylabel('Distance to k-th nearest neighbor')
    plt.title('k-Distance Plot')
    plt.show()


    #cluster with DBSCAN
    #dbscan = DBSCAN(eps=6, min_samples=10).fit(X)
    dbscan = DBSCAN(eps=4, min_samples=15).fit(X)

    #visulize labels as different colors
    for_vis = X.reshape((rows, cols, -1)).mean(2)
    for_vis = (for_vis - for_vis.min()) / (for_vis.max() - for_vis.min())
    for_vis = (for_vis * 255).astype(np.uint8)
    labels = dbscan.labels_.reshape((rows, cols, -1))
    max_label = labels.max()
    color_labels = np.stack((for_vis,) * 3, axis=-1).astype(np.uint8)
    for r in range(rows):
        for c in range(cols):
            if labels[r, c] == -1:
                color_labels[r, c] = [0, 0, 0]
            else:
                new_voxel = interpolate_rainbow(labels[r, c]/max_label)
                color_labels[r, c] = new_voxel

    color_labels = cv2.resize(color_labels, (1000, int((1000 / cols) * rows)), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("dbscan", color_labels)
    cv2.waitKey()

    #print out cluster labels and sizes
    y = list(dbscan.labels_)
    unique_values = sorted(list(np.unique(y)))
    for val in unique_values:
        print(f"label:{val}, count:{y.count(val)}")





def Spectral():
    data = Dysco.generate_data(10, 300, "Data/TestSet1ProperlyFormatted", auto_rows=True,
                               features=Dysco.COLOR | Dysco.EDGE, include_location=False)

    # retreive a sample to work with
    X = data[0]['X']
    rows = data[0]['rows']
    cols = data[0]['cols']

    #rescale X to be between 0 and 255
    X = (X - X.min()) / (X.max() - X.min())
    X = (X * 255).astype(np.uint8)

    # Spectral Clustering
    # clustering = SpectralClustering(n_clusters=2,
    # ...         assign_labels='discretize',
    # ...         random_state=0).fit(X)
    n_clusters = 5
    SC = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=0).fit(X)

    #visulize labels as different colors
    for_vis = X.reshape((rows, cols, -1)).mean(2)
    for_vis = (for_vis - for_vis.min()) / (for_vis.max() - for_vis.min())
    for_vis = (for_vis * 255).astype(np.uint8)
    labels = SC.labels_.reshape((rows, cols, -1))
    max_label = labels.max()
    color_labels = np.stack((for_vis,) * 3, axis=-1).astype(np.uint8)
    for r in range(rows):
        for c in range(cols):
            if labels[r, c] == -1:
                color_labels[r, c] = [0, 0, 0]
            else:
                new_voxel = interpolate_rainbow(labels[r, c]/max_label)
                color_labels[r, c] = new_voxel

    color_labels = cv2.resize(color_labels, (1000, int((1000 / cols) * rows)), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Spectral Clustering", color_labels)
    cv2.waitKey()


def some_others():
    # generate the data
    data = Dysco.generate_data(10, 200, "Data/TestSet1ProperlyFormatted", auto_rows=True,
                               features=Dysco.COLOR | Dysco.EDGE | Dysco.TEXTURE)

    # retreive a sample to work with
    X = data[0]['X']
    rows = data[0]['rows']
    cols = data[0]['cols']

    # visualize current step
    for_vis = X.reshape((rows, cols, -1)).mean(2)
    for_vis = (for_vis - for_vis.min()) / (for_vis.max() - for_vis.min())
    for_vis = (for_vis * 255).astype(np.uint8)


    img = X.reshape((rows, cols, -1))
    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    segments_slic = slic(img, n_segments=250, compactness=10, sigma=1,
                         start_label=1)
    #segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    gradient = sobel(img)
    segments_watershed = watershed(gradient, markers=250, compactness=0.001)

    print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')
    print(f'SLIC number of segments: {len(np.unique(segments_slic))}')
    #print(f'Quickshift number of segments: {len(np.unique(segments_quick))}')
    #print(f'Watershed number of segments: {len(np.unique(segments_watershed))}')

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    ax[0, 0].imshow(mark_boundaries(for_vis, segments_fz))
    ax[0, 0].set_title("Felzenszwalbs's method")
    ax[0, 1].imshow(mark_boundaries(for_vis, segments_slic))
    ax[0, 1].set_title('SLIC')
    #ax[1, 0].imshow(mark_boundaries(for_vis, segments_quick))
    ax[1, 0].set_title('Quickshift')
    #ax[1, 1].imshow(mark_boundaries(for_vis, segments_watershed))
    ax[1, 1].set_title('Compact watershed')

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()



def test_rag_from_skimage():
    """
    testing this
    https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rag_mean_color.html#sphx-glr-auto-examples-segmentation-plot-rag-mean-color-py

    """

    #retreive image to work with
    data = Dysco.generate_data(10, 400, "Data/TestSet1ProperlyFormatted", auto_rows=True,
                               features=Dysco.COLOR | Dysco.EDGE)
    X = data[0]['X']
    rows = data[0]['rows']
    cols = data[0]['cols']
    X = X.reshape((rows, cols, -1)).mean(2)

    cv2.imshow("Composite feature image", X)
    cv2.waitKey()


    #Code from the site
    from skimage import segmentation, color
    from skimage import graph
    img = X
    labels1 = segmentation.slic(img, compactness=10, n_segments=400, start_label=1, channel_axis=None)
    out1 = color.label2rgb(labels1, img, kind='avg', bg_label=0)

    g = graph.rag_mean_color(img, labels1)
    #labels2 = graph.cut_threshold(labels1, g, 29)
    labels2 = graph.cut_normalized(labels1, g, thresh=0.1)
    out2 = color.label2rgb(labels2, img, kind='avg', bg_label=0)

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,
                           figsize=(6, 8))

    ax[0].imshow(out1)
    ax[1].imshow(out2)

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()



def just_show_bunch_of_images():
    # retreive image to work with
    data = Dysco.generate_data(10, 400, "Data/TestSet1ProperlyFormatted", auto_rows=True,
                               features=Dysco.COLOR | Dysco.EDGE, shape=Dysco.RECT)

    images = [d['X'].mean(2) for d in data]
    Dysco.show_images(images)

    # descriptions = ["description here" for _ in data]
    # Dysco.show_images(images, descriptions)




def main():
    #show_kmeans()
    #test_dbscan()
    #Spectral()
    #some_others()
    test_rag_from_skimage()
    #just_show_bunch_of_images()



if __name__ == "__main__":
    main()