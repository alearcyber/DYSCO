"""
This file will have helpful functions for data mining and whatnot
"""
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2
import numpy as np

###################################################
#Helper Function For Plotting The Variances FOR PCA
###################################################
def plot_pca_variance(data, n_components=None):
    # calculate principle components
    pc = PCA(n_components=n_components)
    pc.fit(data)  # try with SCALED data instead?

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
    plt.show()


#crops an image
def crop_image(image, mask):
    """ mask is (x1, y1, x2, y2)"""
    x1, y1, x2, y2 = mask  # unpack mask tuple
    image = image[y1:y2, x1:x2]
    return image


#split up an image into n x m,  n is across, m is up and down
def grid_out_image(image, n, m):
    """split the image into n x n sub-images, return in list, ordered like reading order"""
    height, width = image.shape[0], image.shape[1] # extract width and height of the image
    subheight = height//m
    subwidth = width//n

    x_crossections = []
    y_crossections = []

    #declare subimages sampling grid
    subimages = []
    rows, cols = m, n
    for i in range(rows):
        col = []
        for j in range(cols):
            col.append(0)
        subimages.append(col)



    for i in range(n):
        x_crossections.append(subwidth * i)

    for i in range(m):
        y_crossections.append(subheight * i)

    for j in range(len(y_crossections)):
        y = y_crossections[j]
        for i in range(len(x_crossections)):
            x = x_crossections[i]
            x1, y1, x2, y2 = x, y, x + subwidth, y + subheight

            if x2 > width:
                x2 = width
            if y2 > height:
                y2 = height

            mask = x1, y1, x2, y2
            cropped_section = crop_image(image, mask)
            subimages[j][i] = cropped_section

    return subimages

#helper function to extend grayscale images
def _extend_image_grayscale(a, b):
    # extend height
    dx = a.shape[0] - b.shape[0]
    if dx > 0:  # a is bigger, extend b
        new_pixels = np.zeros((abs(dx), b.shape[1]), dtype='uint8')
        b = np.concatenate((b, new_pixels), axis=0)
    elif dx < 0:  # b is bigger, extend a
        new_pixels = np.zeros((abs(dx), a.shape[1]), dtype='uint8')
        a = np.concatenate((a, new_pixels), axis=0)

    # extend width
    dy = a.shape[1] - b.shape[1]
    if dy > 0:  # a is bigger, extend b
        new_pixels = np.zeros((b.shape[0], abs(dy)), dtype='uint8')
        b = np.concatenate((b, new_pixels), axis=1)
    elif dy < 0:  # b is bigger, extend a
        new_pixels = np.zeros((a.shape[0], abs(dy)), dtype='uint8')
        a = np.concatenate((a, new_pixels), axis=1)

    # return output
    return a, b


#extend image bounds with blank pixels so they match.
def _extend_image(a, b):
    #check if the image is grayscale
    if len(a.shape) == 2:
        return _extend_image_grayscale(a, b)


    #extend height
    dx = a.shape[0] - b.shape[0]
    if dx > 0: #a is bigger, extend b
        new_pixels = np.zeros((abs(dx), b.shape[1], 3), dtype='uint8')
        b = np.concatenate((b, new_pixels), axis=0)
    elif dx < 0: #b is bigger, extend a
        new_pixels = np.zeros((abs(dx), a.shape[1], 3), dtype='uint8')
        a = np.concatenate((a, new_pixels), axis=0)

    #extend width
    dy = a.shape[1] - b.shape[1]
    if dy > 0:  # a is bigger, extend b
        new_pixels = np.zeros((b.shape[0], abs(dy), 3), dtype='uint8')
        b = np.concatenate((b, new_pixels), axis=1)
    elif dy < 0:  # b is bigger, extend a
        new_pixels = np.zeros((a.shape[0], abs(dy), 3), dtype='uint8')
        a = np.concatenate((a, new_pixels), axis=1)

    #return output
    return a, b


#Aligns query image with the train image based on their features and estimating the affine transformation.
#Returns a copy of the query image and the train image. This is because both images may be given padding so they
# have the same dimensions.
#Note: Does not currently consider the accuracy/strength of the matches.
def align_images(query, train):
    # extend bounds of images
    query, train = _extend_image(query, train)

    # keypoints and matching
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(query, None)
    kp2, des2 = sift.detectAndCompute(train, None)
    bf = cv2.BFMatcher(cv2.DIST_L2, crossCheck=True)
    matches = bf.match(des1, des2)  # first is query, second is train
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find the transformation between points, standard RANSAC
    transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    #Warp the image
    query = cv2.warpPerspective(query, transformation_matrix, (query.shape[1], query.shape[0]))

    # return the aligned images
    return query, train




#show an image
#shows an image
def show(image, title=None):
    plt.imshow(image, cmap='gray', interpolation='none', extent=[0, image.shape[1], image.shape[0], 0])
    if not (title is None):
        plt.title(title)
    plt.show()

def rescale(image):
    min_val = np.min(image)
    max_val = np.max(image)
    rescaled_image = (image - min_val) * (255.0 / (max_val - min_val))
    rescaled_image = rescaled_image.astype(np.uint8)
    return rescaled_image




#calculate the chi-squared distance
def chi_squared_distance(a, b):
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b) for (a, b) in zip(a, b)])
    return chi



#makes numpy array percentages along the given axis
def convert_to_percent(array, axis):
    s = np.sum(array, axis)
    cumulation = np.array([s for _ in range(array.shape[axis])])
    percents = (array / cumulation) * 100
    return percents


def test():
    print('Running DataProcessing tests...')
    sampling_grid = grid_out_image(cv2.imread('images/fail3.jpg'), 3, 2)
    for row in range(2):
        for col in range(3):
            cv2.imshow(f'row:{row}, col:{col}', sampling_grid[row][col])
    cv2.waitKey()

def test2():
    raw = cv2.imread("images/rawdisplay.png")
    cap = cv2.imread("images/displayoutput.png")

    one, two = align_images(raw, cap)
    cv2.imshow('one', one)
    cv2.imshow('two', two)


    composite = np.concatenate([one, two], axis=2)
    composite = np.mean(composite, axis=2).astype(np.uint8)
    cv2.imshow('composite', composite)

    cv2.waitKey()

    folder = "/Users/aidanlear/Desktop/"
    cv2.imwrite(folder + "one.png", one)
    cv2.imwrite(folder + "two.png", two)
    cv2.imwrite(folder + "composite.png", composite)

def test_chi2_dist():
    one = np.array([1, 2, 13, 5, 45, 23])
    two = np.array([67, 90, 18, 79, 24, 98])
    results = chi_squared_distance(one, two)
    print("chi squared distance:", results)





if __name__ == '__main__':
    test_chi2_dist()
