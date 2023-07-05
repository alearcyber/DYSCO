"""
This file explores law's texture masks and kernels.

TODO: finish writing out the details of this file


Quick Reminder about numpy array shapes as it relates to image dimensions:
 the first dimension in shape is the number of rows, aka the height
 the second dimension in shape is the number of columns, aka the width
 the third dimension in shape is the color channels, aka the depth or "bands"
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import DataProcessing
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from DataProcessing import plot_pca_variance


#seed for random states
RANDOM_STATE = 25



#filter vectors
level = np.array([[1, 4, 6, 4, 1]])
edge = np.array([[-1, -2, 0, 2, 1]])
spot = np.array([[-1, 0, 2, 0, -1]])
wave = np.array([[-1, 2, 0, -2, 1]])
ripple = np.array([[1, -4, 6, -4, 1]])

#edge kernels
el = np.dot(edge.reshape(-1, 1), level)
ee = np.dot(edge.reshape(-1, 1), edge)
es = np.dot(edge.reshape(-1, 1), spot)
ew = np.dot(edge.reshape(-1, 1), wave)
er = np.dot(edge.reshape(-1, 1), ripple)

#level kernels
ll = np.dot(level.reshape(-1, 1), level)
le = np.dot(level.reshape(-1, 1), edge)
ls = np.dot(level.reshape(-1, 1), spot)
lw = np.dot(level.reshape(-1, 1), wave)
lr = np.dot(level.reshape(-1, 1), ripple)

#spot kernels
sl = np.dot(spot.reshape(-1, 1), level)
se = np.dot(spot.reshape(-1, 1), edge)
ss = np.dot(spot.reshape(-1, 1), spot)
sw = np.dot(spot.reshape(-1, 1), wave)
sr = np.dot(spot.reshape(-1, 1), ripple)

#wave kernels
wl = np.dot(wave.reshape(-1, 1), level)
we = np.dot(wave.reshape(-1, 1), edge)
ws = np.dot(wave.reshape(-1, 1), spot)
ww = np.dot(wave.reshape(-1, 1), wave)
wr = np.dot(wave.reshape(-1, 1), ripple)

#ripple kernels
rl = np.dot(ripple.reshape(-1, 1), level)
re = np.dot(ripple.reshape(-1, 1), edge)
rs = np.dot(ripple.reshape(-1, 1), spot)
rw = np.dot(ripple.reshape(-1, 1), wave)
rr = np.dot(ripple.reshape(-1, 1), ripple)

texture_kernels = [ # list of 2-tuples: ('name', kernel)
    ('edge x level', el), ('edge x edge', ee), ('edge x spot', es), ('edge x wave', ew), ('edge x ripple', er),
    ('level x level', ll), ('level x edge', le), ('level x spot', ls), ('level x wave', lw), ('level x ripple', lr),
    ('spot x level', sl), ('spot x edge', se), ('spot x spot', ss), ('spot x wave', sw), ('spot x ripple', sr),
    ('wave x level', wl), ('wave x edge', we), ('wave x spot', ws), ('wave x wave', ww), ('wave x ripple', wr),
    ('ripple x level', rl), ('ripple x edge', re), ('ripple x spot', rs), ('ripple x wave', rw), ('ripple x ripple', rr),
]



#shows an image
def show(image, title=None):
    plt.imshow(image, cmap='gray')
    if not (title is None):
        plt.title(title)
    plt.show()



#First Step In Law's procedure: Remove illumination by subtracting mean blur from original.
#He used a 15x15 window size and a mean blur.
#I might try different window sizes and a gaussian blur perhaps.
def remove_illumination(img, blur_mode='mean', window_size=15):
    #input validation
    assert blur_mode in ['mean', 'gauss'], "invalid blur mode, options are 'mean' and 'gauss'"
    assert (window_size % 2) > 0, 'window size must be odd'
    assert window_size > 2, 'window size must be 3 or more, in addition to being odd'


    #blur the image
    if blur_mode == 'mean':
        blur = cv2.blur(img, (window_size, window_size))
    else:
        blur = cv2.GaussianBlur(img, (window_size, window_size), 0)


    #subtract blur
    return img - blur



def test_removing_illumination():
    image = cv2.imread('images/fail4.jpg')
    out = remove_illumination(image)
    show(image, 'original')
    show(out, 'illumination removed')

    r1 = cv2.filter2D(out, ddepth=-1, kernel=el)
    r2 = cv2.filter2D(image, ddepth=-1, kernel=el)
    show(r1, 'mask with illumination removed')
    show(r2, 'mask without preprocessing')


#Applies the 16 kernels to the image to create 16 bands of texture features,
# (each band has 3 colors).
#Returns a numpy array of the images all kind of stacked and mashed together into one image.
#Currently does not label the different bands in any significant way.
def generate_texture_features(img):
    if len(img.shape) == 2: #image is in grayscale
        return _generate_texture_features_gray(img)

    out = np.empty(shape=(img.shape[0], img.shape[1], 0), dtype='uint8')
    for descriptor, kernel in texture_kernels:
        layer = cv2.filter2D(img, ddepth=-1, kernel=kernel)
        out = np.concatenate((out, layer), axis=2)
    return out


#helper for generate_texture_features() to handle grayscale images
def _generate_texture_features_gray(img):
    out = np.empty(shape=(img.shape[0], img.shape[1], 0), dtype='uint8')
    for descriptor, kernel in texture_kernels:
        layer = cv2.filter2D(img, ddepth=-1, kernel=kernel)
        # expand dimensions of layer
        layer = np.expand_dims(layer, axis=2)
        out = np.concatenate((out, layer), axis=2)
    return out


#Applies the 16 kernels to the image to create 16 bands of texture features,
# (each band has 3 colors).
#Returns the 16 images as a list.
#Currently does not label the different bands in any significant way
def generate_texture_features_example(img):
    for descriptor, kernel in texture_kernels:
        out = cv2.filter2D(img, ddepth=-1, kernel=kernel)
        #print('dtype:', out.dtype)
        #print('shape:', out.shape)
        cv2.imwrite("TestResults/TextureMaskExamples/" + descriptor + ".png", out)  #this line saves the masks
        show(out, descriptor)



#will analyze the variance accounted for by the first n principle components for an example image
def pca_variance_test():
    #create the texture features
    X = generate_texture_features(cv2.imread('images/fail3.jpg'))

    #flatten the image
    original_dimensions = X.shape
    X = np.reshape(X, (-1, original_dimensions[2]))

    # data normalization
    """
    scaler = MinMaxScaler()
    scaler.fit(X.astype(float))
    X = scaler.transform(X)
    """

    #plot the PC variance
    DataProcessing.plot_pca_variance(X, n_components=15)


#I want to compare my output here to my output in image j.
#I will convolve the image the same way here and in image j as a sort of
# sanity check. They should look the same.
#Will use edge-level kernel
#
#NOTE: After comparing, they look like they are approximately the same. There are notable differences, e.i. they
# are not identical, but I think that is most likely do to scaling or something like that?
def test_a_texture():
    image = cv2.imread("images/fail4.jpg")
    filtered = cv2.filter2D(image, ddepth=-1, kernel=el)
    cv2.imshow('filtered with edge level kernel', filtered)
    cv2.waitKey()






#first attempt at clustering
def first_cluster_attempt(gray=False):
    # create the texture features
    print('Generating texture features...')

    if gray:
        image = cv2.imread('images/fail3.jpg', cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread('images/fail3.jpg')

    X = generate_texture_features(image)

    #TODO CONTINUE HERE, figure out what kind of clustering to do. Not just kmeans
    # but like how can I take into account pixels being near each other?
    # There is good example in Digital Image Processing book.


    #flatten the data
    print('Flattening image...')
    original_dimensions = X.shape
    X = np.reshape(X, (-1, original_dimensions[2]))
    print(X.shape)

    #extract first two principle components
    X_pca = (pca := PCA(n_components=2)).fit_transform(X)


    #Just regular kmeans clustering. I wont take into account any sort of
    # locality or spacial information from the image.
    km = KMeans(n_clusters=2, random_state=RANDOM_STATE)
    km.fit(X)

    # centroids
    centroids = pd.DataFrame(pca.transform(km.cluster_centers_), columns=['PC1', 'PC2'])


    # prediction label colors
    color1 = np.array(['green', 'red', 'blue', 'yellow', 'purple', 'brown'])
    pred_y = pd.DataFrame(km.labels_, columns=['Target'])

    #data visualization
    pc1, pc2 = X_pca[:, 0], X_pca[:, 1]
    plt.scatter(pc1, pc2, c=color1[pred_y['Target']], s=5)
    plt.scatter(centroids['PC1'], centroids['PC2'], c='k', s=10)
    plt.xlabel('PC 1', fontsize=14)
    plt.ylabel('PC 2', fontsize=14)
    plt.title('Kmeans Clustering')
    plt.show()



def test_macro_feature_extraction():
    out = cv2.filter2D(cv2.imread('images/fail3.jpg'), ddepth=-1, kernel=ll)
    sampling_grid = DataProcessing.grid_out_image(out, 15, 10)

    for row in range(10):
        for col in range(15):
            window = sampling_grid[row][col]
            print('avg:', np.average(window), '  std:', np.std(window))
            #cv2.imshow(f'window at row:{row}, col:{col}', window)


# this code will average all the texture bands into one grayscale image
def average_all_bands():
    X = generate_texture_features(cv2.imread('images/fail3.jpg'))
    #X = cv2.imread('images/fail3.jpg')
    print('shape of X:', X.shape)
    X_gray = np.mean(X, axis=2).astype(np.uint8)
    print('shape of X_gray:', X_gray.shape)
    #cv2.imwrite('TestResults/misc-images/texture-grayscale.png', X_gray)
    cv2.imshow('hyperspectral grayscale', X_gray)
    cv2.waitKey()


if __name__ == '__main__':
    """entry point"""
    #pca_variance_test()
    #generate_texture_features_example(cv2.imread('images/fail3.jpg'))
    #test_a_texture()
    #first_cluster_attempt(gray=True)
    #test_macro_feature_extraction()
    average_all_bands()



